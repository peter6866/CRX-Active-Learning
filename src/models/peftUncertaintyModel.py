import pytorch_lightning as pl
import torchmetrics
import torch
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from mpra_tools import plot_utils
import math
import os
import numpy as np
import matplotlib.pyplot as plt


_LOG_2PI = math.log(2 * math.pi)


class PeftUncertaintyModel(pl.LightningModule):
    def __init__(self, model_name, train_dataloader_len, num_epochs, lr=3e-3):
        super(PeftUncertaintyModel, self).__init__()

        # Model
        self.model = model_name
        print(self.model.print_trainable_parameters())

        # Hyperparameters
        self.lr = lr
        self.train_dataloader_len = train_dataloader_len
        self.num_epochs = num_epochs

        # Regression Metrics
        self.train_pcc = torchmetrics.PearsonCorrCoef()
        self.val_pcc = torchmetrics.PearsonCorrCoef()
        self.test_pcc = torchmetrics.PearsonCorrCoef()
        # self.train_scc = torchmetrics.SpearmanCorrCoef()
        # self.val_scc = torchmetrics.SpearmanCorrCoef()
        # self.test_scc = torchmetrics.SpearmanCorrCoef()
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        
        self.retinopathy_preds = []
        self.retinopathy_truth = []
        self.test_preds = []
        self.test_truth = []

        self.std_values = []
        self.retinopathy_pcc = 0.0
        self.retinopathy_scc = 0.0
        self.count = 0
        self.save_dir = "ModelFitting/transformers_uncertainty"

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def gaussian_beta_loss(self, mean, std, target, beta=0.5):
        """Compute beta-NLL loss

        :param mean: Predicted mean of shape B x D
        :param variance: Predicted variance of shape B x D
        :param target: Target of shape B x D
        :param beta: Parameter from range [0, 1] controlling relative 
            weighting between data points, where `0` corresponds to 
            high weight on low error points and `1` to an equal weighting.
        :returns: Loss per batch element of shape B
        """
        std = torch.exp(std)
        var = std * std
        ll = -0.5 * ((target - mean) ** 2 / var + torch.log(var) + _LOG_2PI)
        weight = var.detach() ** beta
        res = -torch.mean(ll * weight)

        return res
    
    def gauss_loss(self, mean, std, target):
        std = torch.exp(std)
        variances = std * std

        log_p = (-torch.log(torch.sqrt(2 * math.pi * variances))
                - (target - mean) * (target - mean) / (2 * variances))

        return torch.mean(-log_p)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        mean, std = outputs.logits[:, 0], outputs.logits[:, 1]
        loss = self.gauss_loss(mean=mean, std=std, target=batch["labels"])

        self.log('train_nll_loss', loss, on_epoch=True, on_step=False)
        self.log('train_mse', self.train_mse(mean.squeeze(), batch["labels"]), on_epoch=True, on_step=False)
        self.log('train_pcc', self.train_pcc(mean.squeeze(), batch["labels"]), on_epoch=True, on_step=False)
        #self.log('train_scc', self.train_scc(outputs.logits.squeeze(), batch["labels"]), on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.06 * (self.train_dataloader_len * self.num_epochs),
            num_training_steps=(self.train_dataloader_len * self.num_epochs)
        )
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        mean, std = outputs.logits[:, 0], outputs.logits[:, 1]
        loss = self.gauss_loss(mean=mean, std=std, target=batch["labels"])

        self.log('val_nll_loss', loss, on_epoch=True, on_step=False)
        self.log("val_mse_loss", self.val_mse(mean.squeeze(), batch["labels"]), on_epoch=True, on_step=False)
        self.log('val_pcc', self.val_pcc(mean.squeeze().squeeze(), batch["labels"]), on_epoch=True, on_step=False)
        #self.log('val_scc', self.val_scc(outputs.logits.squeeze(), batch["labels"]), on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx, dataloader_idx):
        outputs = self.model(**batch)
        mean, std = outputs.logits[:, 0], outputs.logits[:, 1]

        pcc = self.test_pcc(mean.squeeze(), batch["labels"])
        scc = self.test_scc(mean.squeeze(), batch["labels"])
        if dataloader_idx == 0:
            self.log('test_retinopathy_pcc', pcc, on_epoch=True)
            #self.log('test_retinopathy_scc', scc, on_epoch=True)
            self.retinopathy_preds.append(mean.squeeze().detach().cpu().numpy())
            self.retinopathy_truth.append(batch["labels"].detach().cpu().numpy())
            self.std_values.aooend(torch.exp(std).detach().cpu().numpy())

            self.retinopathy_pcc += pcc.detach().item()
            self.retinopathy_scc += scc.detach().item()
            self.count += 1
        else:
            self.log('test_set_pcc', pcc, on_epoch=True)
            #self.log('test_set_scc', scc, on_epoch=True)
            self.test_preds.append(mean.squeeze().detach().cpu().numpy())
            self.test_truth.append(batch["labels"].detach().cpu().numpy())

    def on_test_end(self):

        preds = {
            "retinopathy": np.concatenate(self.retinopathy_preds),
            "test": np.concatenate(self.test_preds)
        }

        truth = {
            "retinopathy": np.concatenate(self.retinopathy_truth),
            "test": np.concatenate(self.test_truth)
        }

        for name in ["retinopathy", "test"]:
        
            fig, ax = plt.subplots(figsize=plot_utils.get_figsize())
            fig, ax, corrs = plot_utils.scatter_with_corr(
                truth[name],
                preds[name],
                "Observed",
                "Predicted",
                colors="density",
                loc="upper left",
                figax=(fig, ax),
                rasterize=True,
            )
            # Show y = x
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            vmin = max(xlim[0], ylim[0])
            vmax = min(xlim[1], ylim[1])
            ax.plot([vmin, vmax], [vmin, vmax], color="k", linestyle="--", lw=1)
            plot_utils.save_fig(fig, os.path.join(self.save_dir, f"{name}PredVsObs"))
            plt.close()

        means = np.concatenate(self.mean_values)
        stds = np.concatenate(self.std_values)

        mean_pcc = self.retinopathy_pcc / self.count
        mean_scc = self.retinopathy_scc / self.count

        # Create a new figure with a specified size and DPI
        plt.figure(figsize=(10, 6), dpi=200)

        # Scatter plot with customized marker style, edgecolors and size
        plt.scatter(means, stds, alpha=0.7, c='blue', edgecolors='w', s=50, linewidth=0.5)

        # Titles, labels with font size and weight adjustments
        plt.title("Mean vs Standard Deviation on Retinopathy test set", fontsize=18, fontweight='bold')
        plt.xlabel("Mean", fontsize=16)
        plt.ylabel("Standard deviation", fontsize=16)

        # Tweak the tick label size
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Optional: Add a grid for better readability
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.gca().set_facecolor("#f4f4f4")  # Set a background color

        # Annotate the number of samples
        num_samples = len(means)
        plt.annotate(f"n = {num_samples}",
                    xy=(0.95, 0.95),
                    xycoords='axes fraction',
                    fontsize=14,
                    fontweight='bold',
                    ha="right",
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="#ffffff", facecolor="#e1e1e1"))

        # Annotate the test_retinopathy_pcc and test_retinopathy_scc
        plt.annotate(f"PCC = {mean_pcc:.4f}",
                    xy=(0.05, 0.93),
                    xycoords='axes fraction',
                    fontsize=14,
                    fontweight='bold',
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="#ffffff", facecolor="#e1e1e1"))

        plt.annotate(f"SCC = {mean_scc:.4f}",
                    xy=(0.05, 0.85),
                    xycoords='axes fraction',
                    fontsize=14,
                    fontweight='bold',
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="#ffffff", facecolor="#e1e1e1"))

        # Make the plot tight layout
        plt.tight_layout()

        file_path = os.path.join(self.save_dir, "mean_vs_std_plot_on_retinopathy.png")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

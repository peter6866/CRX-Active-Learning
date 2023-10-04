import pytorch_lightning as pl
import torchmetrics
import torch
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from mpra_tools import plot_utils
import math
import os
import numpy as np
import matplotlib.pyplot as plt


class PeftRegressionModel(pl.LightningModule):
    def __init__(self, model_name, train_dataloader_len, num_epochs, lr=3e-3):
        super(PeftRegressionModel, self).__init__()

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

        self.loss_fn = torch.nn.MSELoss()
        
        self.retinopathy_preds = []
        self.retinopathy_truth = []
        self.test_preds = []
        self.test_truth = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = self.loss_fn(outputs.logits.squeeze(), batch["labels"])
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_pcc', self.train_pcc(outputs.logits.squeeze(), batch["labels"]), on_epoch=True, on_step=False)
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
        loss = self.loss_fn(outputs.logits.squeeze(), batch["labels"])
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log('val_pcc', self.val_pcc(outputs.logits.squeeze(), batch["labels"]), on_epoch=True, on_step=False)
        #self.log('val_scc', self.val_scc(outputs.logits.squeeze(), batch["labels"]), on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx, dataloader_idx):
        outputs = self.model(**batch)
        pred = outputs.logits.squeeze()
        pcc = self.test_pcc(pred, batch["labels"])
        #scc = self.test_scc(outputs.logits.squeeze(), batch["labels"])
        if dataloader_idx == 0:
            self.log('test_retinopathy_pcc', pcc, on_epoch=True)
            #self.log('test_retinopathy_scc', scc, on_epoch=True)
            self.retinopathy_preds.append(pred.detach().cpu().numpy())
            self.retinopathy_truth.append(batch["labels"].detach().cpu().numpy())
        else:
            self.log('test_set_pcc', pcc, on_epoch=True)
            #self.log('test_set_scc', scc, on_epoch=True)
            self.test_preds.append(pred.detach().cpu().numpy())
            self.test_truth.append(batch["labels"].detach().cpu().numpy())

    def on_test_end(self):
        OUTPUT_DIR = "ModelFitting/transformers"
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
            plot_utils.save_fig(fig, os.path.join(OUTPUT_DIR, f"{name}PredVsObs"))
            plt.close()
        

    # def on_test_epoch_end(self):
    #     self.save_pretrained("hfmodels/peftTest")
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(**batch)
        return outputs.logits

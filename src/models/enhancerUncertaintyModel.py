"""
A model to predict the activity of a cis-regulatory sequence and its uncertainty.
"""
import math
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from selene_sdk.utils import load_model_from_state_dict
import os
import matplotlib.pyplot as plt
from mpra_tools import plot_utils
import seaborn as sns
import pandas as pd


_LOG_2PI = math.log(2 * math.pi)


class EnhancerUncertaintyModel(pl.LightningModule):
    def __init__(self, sequence_length, learning_rate, sample_type=None):
        super().__init__()
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.sample_type = sample_type

        conv_dropout = 0.1
        resid_dropout = 0.25
        fc_neurons = 256
        fc_dropout = 0.25
        # [input_filters, output_filters, filter_size, activation, dilation_layers, pool_size]
        architecture = [
            [4, 256, 10, "exp", 4, 4],   # 164 --> 41
            [256, 256, 6, "relu", 3, 4], # 41 -->11
            [256, 64, 3, "relu", 2, 3],   # 11 --> 4
        ]

        layers = []
        for input_filters, output_filters, filter_size, activation, dilation_layers, pool_size in architecture:
            # Conv
            layers.append(nn.Conv1d(
                input_filters, output_filters, kernel_size=filter_size, padding=filter_size-1
            ))
            # BN
            layers.append(nn.BatchNorm1d(output_filters))
            # Activation
            layers.append(get_activation(activation))
            # Dropout
            layers.append(nn.Dropout(p=conv_dropout))
            # Residual dilation
            layers.append(DilationBlock(output_filters, dilation_layers))
            # Activation
            layers.append(nn.ReLU())
            # Dropout
            layers.append(nn.Dropout(p=resid_dropout))
            # Pooling
            layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))

        # Flatten + FC
        layers.append(nn.Flatten())
        layers.append(nn.LazyLinear(fc_neurons))
        # BN
        layers.append(nn.BatchNorm1d(fc_neurons))
        # Activation
        layers.append(nn.GELU())
        # Dropout
        layers.append(nn.Dropout(p=fc_dropout))

        self.conv_net = nn.Sequential(*layers)
        # Output mean and standard deviation
        self.output = nn.Linear(fc_neurons, 2)

        # Regression Metrics
        self.train_pcc = torchmetrics.PearsonCorrCoef()
        self.val_pcc = torchmetrics.PearsonCorrCoef()
        self.test_pcc = torchmetrics.PearsonCorrCoef()
        self.train_scc = torchmetrics.SpearmanCorrCoef()
        self.val_scc = torchmetrics.SpearmanCorrCoef()
        self.test_scc = torchmetrics.SpearmanCorrCoef()
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()

        self.mean_values = []
        self.std_values = []
        self.retinopathy_pcc = 0.0
        self.retinopathy_scc = 0.0
        self.count = 0

        self.muta_mean_values = []
        self.muta_std_values = []
        self.muta_pcc = 0.0
        self.muta_scc = 0.0
        self.muta_count = 0

        # metrics for generated sequences
        self.generated_mean_values = []
        self.generated_std_values = []
        self.generated_pcc = 0.0
        self.generated_scc = 0.0
        self.generated_count = 0

        # metrics for crx mutants
        self.crx_mean_values = []
        self.crx_std_values = []
        self.crx_pcc = 0.0
        self.crx_scc = 0.0
        self.crx_count = 0

        # Predicted vs. observed
        self.retinopathy_truth = []
        self.muta_truth = []
   
        self.save_dir = "ModelFitting/uncertainty"

    def forward(self, x):

        return self.output(self.conv_net(x))
    
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
        seq, target = batch
        outputs = self(seq)
        mean = outputs[:, 0]
        std = outputs[:, 1]
        beta_nll_loss = self.gaussian_beta_loss(mean=mean, std=std, target=target)
        loss = self.gauss_loss(mean=mean, std=std, target=target)
    
        self.log('train_nll_loss', loss, on_epoch=True, on_step=False)
        self.log('train_mse', self.train_mse(mean.squeeze(), target), on_epoch=True, on_step=False)
        self.log('train_pcc', self.train_pcc(mean.squeeze(), target), on_epoch=True, on_step=False)
        self.log('train_scc', self.train_scc(mean.squeeze(), target), on_epoch=True, on_step=False)
        return beta_nll_loss
    
    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        seq, target = batch
        outputs = self(seq)
        mean = outputs[:, 0]
        std = outputs[:, 1]
        # loss = self.gaussian_log_likelihood_loss(mean=mean, var=var, target=target)
        loss = self.gauss_loss(mean=mean, std=std, target=target)
    
        self.log('val_nll_loss', loss, on_epoch=True, on_step=False)
        self.log('val_mse', self.val_mse(mean.squeeze(), target), on_epoch=True, on_step=False)
        self.log('val_pcc', self.val_pcc(mean.squeeze(), target), on_epoch=True, on_step=False)
        self.log('val_scc', self.val_scc(mean.squeeze(), target), on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx, dataloader_idx):
        seq, target = batch
        outputs = self(seq)
        mean = outputs[:, 0]
        std = outputs[:, 1]

        pcc = self.test_pcc(mean.squeeze(), target)
        scc = self.test_scc(mean.squeeze(), target)

        if dataloader_idx == 0:
            self.mean_values.append(mean.detach().cpu().numpy())
            self.std_values.append(torch.exp(std).detach().cpu().numpy())
            self.retinopathy_truth.append(target.detach().cpu().numpy())

            self.retinopathy_pcc += pcc.detach().item()
            self.retinopathy_scc += scc.detach().item()
            self.count += 1
        elif dataloader_idx == 1:
            self.muta_mean_values.append(mean.detach().cpu().numpy())
            self.muta_std_values.append(torch.exp(std).detach().cpu().numpy())
            self.muta_truth.append(target.detach().cpu().numpy())

            self.muta_pcc += pcc.detach().item()
            self.muta_scc += scc.detach().item()
            self.muta_count += 1
        elif dataloader_idx == 2:
            self.generated_mean_values.append(mean.detach().cpu().numpy())
            self.generated_std_values.append(torch.exp(std).detach().cpu().numpy())

            self.generated_pcc += pcc.detach().item()
            self.generated_scc += scc.detach().item()
            self.generated_count += 1
        elif dataloader_idx == 3:
            self.crx_mean_values.append(mean.detach().cpu().numpy())
            self.crx_std_values.append(torch.exp(std).detach().cpu().numpy())

            self.crx_pcc += pcc.detach().item()
            self.crx_scc += scc.detach().item()
            self.crx_count += 1

    def on_test_end(self):
        means = np.concatenate(self.mean_values)
        stds = np.concatenate(self.std_values)

        # means = np.concatenate(self.generated_mean_values)
        # stds = np.concatenate(self.generated_std_values)
        
        std_indices = np.where(stds < 1.2)[0]
        certain_means = means[std_indices]
        df = pd.DataFrame(certain_means)
        df.to_csv(os.path.join(self.save_dir, "certain_means.csv"), index=False)
        # print(len(certain_means), min(certain_means), max(certain_means))

        # plt.figure(figsize=(10, 6), dpi=200)
        # sns.violinplot(data=certain_means)
        # plt.title("Distribution of predicted means on generated sequences")
        # plt.savefig(os.path.join(self.save_dir, "generated_means_distribution.png"))

        preds = {
            "retinopathy": means,
            "muta": np.concatenate(self.muta_mean_values)
        }

        truth = {
            "retinopathy": np.concatenate(self.retinopathy_truth),
            "muta": np.concatenate(self.muta_truth)
        }

        for name in ["retinopathy", "muta"]:
        
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

        mean_pcc = self.retinopathy_pcc / self.count
        mean_scc = self.retinopathy_scc / self.count

        muta_mean_pcc = self.muta_pcc / self.muta_count
        muta_mean_scc = self.muta_scc / self.muta_count

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
        
        plt.annotate(f"Muta: PCC = {muta_mean_pcc:.4f} SCC = {muta_mean_scc:.4f}", 
                    xy=(0.05, 0.77), 
                    xycoords='axes fraction',
                    fontsize=14,
                    fontweight='bold',
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="#ffffff", facecolor="#e1e1e1"))

        # Make the plot tight layout
        plt.tight_layout()
        
        file_path = os.path.join(self.save_dir, f"mean_vs_std_plot_on_retinopathy_{self.sample_type}.png")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close() 

    def predict_step(self, batch, batch_idx):
        seq, _ = batch
        outputs = self(seq)
        mean = outputs[:, 0]
        std = outputs[:, 1]

        return mean, std


class ResBlock(nn.Module):
    """
    Wrapper to make an arbitrary Sequence a residual block by adding a skip connection.
    https://towardsdatascience.com/building-a-residual-network-with-pytorch-df2f6937053b

    Attributes
    ----------
    block : nn.Sequential
        The arbitrary sequence to wrap in a skip connection.
    """
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x.clone() + self.block(x)


class DilationBlock(ResBlock):
    """
    A DilationBlock is a Sequence of convolutions with exponentially increasing dilations. Each convolution is
    followed by batch normalization, activation, and dropout. The final convolution is followed by a batch
    normalization step but not activation or dropout. The entire block is surrounded by a skip connection to create a
    ResBlock. The DilationBlock should be followed by an activation function and dropout after the skip connection.

    Parameters
    ----------
    nfilters : int
        Number of convolution filters to receive as input and use as output.
    nlayers : int
        Number of dilation layers to use.
    rate : int, optional
        Default is 2. Base to use for the dilation factor.
    width : int, optional
        Default is 3. Size of convolution filters.
    activation : str, optional
        Default is "relu". Activation function to use. Other options are "exp" for Exp, "lrelu" for LeakyReLU.
    dropout : float, optional
        Default is 0.1. Dropout rate to use. Must be in the range [0, 1). If 0, no dropout is used.
    """
    def __init__(self, nfilters, nlayers, rate=2, width=3, activation="relu", dropout=0.1):
        layers = []
        # Initial conv and BN
        layers.append(nn.Conv1d(
            nfilters, nfilters, kernel_size=width, padding=width//2, bias=False, dilation=1,
        ))
        layers.append(nn.BatchNorm1d(nfilters))
        for dilation in range(1, nlayers):
            # Activation function
            layers.append(get_activation(activation))
            # Dropout
            if dropout >= 1:
                raise ValueError("Dropout is a number greater than 1.")
            elif dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            # Conv
            layers.append(nn.Conv1d(
                nfilters, nfilters, kernel_size=width, padding=(width // 2 + 1) ** dilation, bias=False,
                dilation=rate ** dilation
            ))
            # BN
            layers.append(nn.BatchNorm1d(nfilters))
        super().__init__(nn.Sequential(*layers))


class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


def get_activation(function):
    """Given a string name of an activation function, return an object for the corresponding Module."""
    if function == "relu":
        return nn.ReLU()
    elif function == "exp":
        return Exp()
    elif function == "lrelu":
        return nn.LeakyReLU()
    else:
        raise ValueError("Did not recognize activation function name.")

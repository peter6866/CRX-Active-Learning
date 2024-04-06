import time
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import wandb

from datamodules.uncertaintyDataModule import UncertaintyDataModule
from models.enhancerUncertaintyModel import EnhancerUncertaintyModel

def fit_uncertainty_model(seed):
    # Global parameters
    LR = 0.001
    BATCH_SIZE = 64
    MAX_EPOCHS = 55
    # TRAINING_ON = "Round4b"
    SAMPLE_TYPE = None

    # data_dir = "Data/activity_summary_stats_and_metadata.txt"
    # retino_dir = "Data/retinopathy_reformatted.txt"
    data_dir = "Data/new_activity_all.csv"
    retino_dir = "Data/new_retinopathy.csv"

    pl.seed_everything(seed)

    wandb.login()

    wandb_logger = WandbLogger(
        project='BCLab-New',
        name=time.strftime('%Y-%m-%d-%H-%M'),
        # group=TRAINING_ON,
        )

    early_stop_callback = EarlyStopping(
        monitor="val_nll_loss", 
        min_delta=0.00,
        patience=10, 
        verbose=False,
        mode="min")

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator='gpu',
        devices=-1,
        max_epochs=MAX_EPOCHS,
        deterministic=True,
        fast_dev_run=False,
        callbacks=[early_stop_callback]
        )

    data_module = UncertaintyDataModule(
        data_path=data_dir,
        retinopathy_path=retino_dir,
        batch_size=BATCH_SIZE,
        sample_type=SAMPLE_TYPE
        )

    model = EnhancerUncertaintyModel(
        learning_rate=LR,
        sample_type=SAMPLE_TYPE,
        label=f"genomic_{SAMPLE_TYPE}_{seed}_noaug"
    )

    torch.set_float32_matmul_precision('high')
    trainer.fit(model, data_module)

    trainer.test(model, data_module)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Uncertainty Model with a specific seed")
    parser.add_argument("seed", type=int, help="The seed to use for training")
    args = parser.parse_args()

    fit_uncertainty_model(args.seed)
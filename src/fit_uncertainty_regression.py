import time
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import wandb

from datamodules.uncertaintyDataModule import UncertaintyDataModule
from datamodules.deepSTARRDataModule import DeepSTARRDataModule
from models.enhancerUncertaintyModel import EnhancerUncertaintyModel


# Global parameters
LR = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 50
TRAINING_ON = "Genomic"
SAMPLE_TYPE = None

data_dir = "Data/activity_summary_stats_and_metadata.txt"
retino_dir = "Data/retinopathy_reformatted.txt"

pl.seed_everything(7)

wandb.login()

wandb_logger = WandbLogger(
    project='BCLab-Uncertainty',
    name=time.strftime('%Y-%m-%d-%H-%M'),
    group=TRAINING_ON,
    )

early_stop_callback = EarlyStopping(
    monitor="val_mse", 
    min_delta=0.00,
    patience=5, 
    verbose=False,
    mode="min")

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator='gpu',
    devices=-1,
    max_epochs=MAX_EPOCHS,
    deterministic=True,
    fast_dev_run=False,
    # callbacks=[early_stop_callback]
    )

data_module = UncertaintyDataModule(
     data_path=data_dir,
     retinopathy_path=retino_dir,
     validate_type=TRAINING_ON,
     batch_size=BATCH_SIZE,
     sample_type=SAMPLE_TYPE
     )

# data_module = DeepSTARRDataModule(
#     batch_size=BATCH_SIZE
# )

model = EnhancerUncertaintyModel(
    learning_rate=LR
)

torch.set_float32_matmul_precision('high')
trainer.fit(model, data_module)

trainer.test(model, data_module)

wandb.finish()

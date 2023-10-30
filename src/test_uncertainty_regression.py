import time
import os

import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb

from datamodules.uncertaintyDataModule import UncertaintyDataModule
from models.enhancerUncertaintyModel import EnhancerUncertaintyModel


# Global parameters
SEQ_SIZE = 164
LR = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 1
TESTING_ON = "Genomic"
CHECKPOINT_DIR = "/scratch/bclab/jiayu/CRX-Active-Learning/BCLab-Uncertainty/8i9ny2vd/checkpoints/epoch=49-step=29950.ckpt"
SAMPLE_TYPE = "random"

data_dir = "Data/activity_summary_stats_and_metadata.txt"
retino_dir = "Data/retinopathy_reformatted.txt"
output_dir = "/scratch/bclab/jiayu/CRX-Active-Learning/ModelFitting/uncertainty"

pl.seed_everything(7)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=-1,
    max_epochs=MAX_EPOCHS,
    deterministic=True,
    fast_dev_run=False,
    )

data_module = UncertaintyDataModule(
     data_path=data_dir,
     retinopathy_path=retino_dir,
     validate_type=TESTING_ON,
     batch_size=BATCH_SIZE
     )

model = EnhancerUncertaintyModel.load_from_checkpoint(
    CHECKPOINT_DIR,
    sequence_length=SEQ_SIZE,
    learning_rate=LR,
    sample_type=SAMPLE_TYPE
)

trainer.test(model, data_module)

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from models.WGAN_Model import WGAN
import time
from datamodules.ganDataset import ganDataset
from torch.utils.data import DataLoader
import numpy as np
import selene_sdk


SEQ_LEN = 227
GEN_LR = 2e-4
CRITIC_LR = 8e-5
EPOCHS = 1500
BATCH_SIZE = 256

data_dir = "Data/atac_seq_data_trimed.csv"

pl.seed_everything(42, workers=True)

wandb_logger = WandbLogger(
    project='BCLab-WGAN',
    name=time.strftime('%m-%d-%H-%M') + f'_{GEN_LR}_{CRITIC_LR}_Improved',
    )

dataset = ganDataset(data_dir)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    num_workers=4, 
    shuffle=True,
    pin_memory=True,
    persistent_workers=True
)

model = WGAN(
    seq_len=SEQ_LEN, 
    gen_lr=GEN_LR, 
    critic_lr=CRITIC_LR,
    epochs = EPOCHS
)

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator='gpu',
    devices=-1,
    max_epochs=EPOCHS,
    deterministic=True
)

torch.set_float32_matmul_precision('high')
trainer.fit(model, dataloader)

wandb.finish()

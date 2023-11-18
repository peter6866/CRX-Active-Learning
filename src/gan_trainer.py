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


SEQ_LEN = 164
data_dir = "Data/activity_summary_stats_and_metadata.txt"

pl.seed_everything(7)

wandb_logger = WandbLogger(
    project='BCLab-WGAN',
    name=time.strftime('%Y-%m-%d-%H-%M'),
    )

dataset = ganDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=True)

model = WGAN(seq_len=SEQ_LEN)

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator='gpu',
    devices=-1,
    max_epochs=500,
    deterministic=True
)

torch.set_float32_matmul_precision('high')
trainer.fit(model, dataloader)

wandb.finish()

noise = torch.randn(100)
sample = model(noise)
_, indices = torch.max(sample, dim=1)
bases = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
dna_sequence = ''.join(bases[i.item()] for i in indices[0])

print(dna_sequence)

import time
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from evoaug import evoaug, augment

import wandb

from datamodules.uncertaintyDataModule import UncertaintyDataModule
from models.enhancerUncertaintyEvoAugModel import EnhancerUncertaintyEvoAugModel

def fit_uncertainty_model(seed):
    # Global parameters
    LR = 0.001
    BATCH_SIZE = 64
    MAX_EPOCHS = 50
    SAMPLE_TYPE = None

    data_dir = "Data/new_activity_all.csv"
    retino_dir = "Data/new_retinopathy.csv"

    augment_list = [
        augment.RandomDeletion(delete_min=0, delete_max=25),
    #     augment.RandomRC(rc_prob=0.5),
        augment.RandomInsertion(insert_min=0, insert_max=15),
        augment.RandomTranslocation(shift_min=0, shift_max=15),
        augment.RandomNoise(noise_mean=0, noise_std=0.3),
    ]

    pl.seed_everything(seed)

    early_stop_callback = EarlyStopping(
        monitor="val_nll_loss", 
        min_delta=0.00,
        patience=10, 
        verbose=False,
        mode="min")

    callback_topmodel = ModelCheckpoint(
        monitor="val_nll_loss",
        save_top_k=1,
        mode="min",
        dirpath="/scratch/bclab/jiayu/al/ModelFitting/evoaug",
        filename=f"topmodel-{seed}"
    )

    trainer = pl.Trainer(
        logger=None,
        accelerator='gpu',
        devices=-1,
        max_epochs=MAX_EPOCHS,
        deterministic=True,
        fast_dev_run=False,
        callbacks=[early_stop_callback, callback_topmodel],
        )

    data_module = UncertaintyDataModule(
        data_path=data_dir,
        retinopathy_path=retino_dir,
        batch_size=BATCH_SIZE,
        sample_type=SAMPLE_TYPE
        )

    model = EnhancerUncertaintyEvoAugModel(
        learning_rate=LR,
        sample_type=SAMPLE_TYPE,
        label=f"genomic_{SAMPLE_TYPE}_{seed}",
        augment_list=augment_list,
        max_augs_per_seq=2,
        hard_aug=True,
        inference_aug=False
    )

    torch.set_float32_matmul_precision('high')
    trainer.fit(model, data_module)

    trainer.test(model, data_module)

    model_finetune = EnhancerUncertaintyEvoAugModel.load_from_checkpoint(
        f"/scratch/bclab/jiayu/al/ModelFitting/evoaug/topmodel-{seed}.ckpt",
        learning_rate=0.0001,
        sample_type=SAMPLE_TYPE,
        label=f"genomic_{SAMPLE_TYPE}_{seed}_finetune",
        augment_list=augment_list,
        max_augs_per_seq=2,
        hard_aug=True,
        finetune=True,
        inference_aug=False
    )
    
    callback_topmodel = ModelCheckpoint(
        monitor="val_nll_loss",
        save_top_k=1,
        mode="min",
        dirpath="/scratch/bclab/jiayu/al/ModelFitting/uncertainty/genomic",
        filename=f"topmodel-{seed}"
    )

    trainer = pl.Trainer(
        logger=None,
        accelerator='gpu',
        devices=-1,
        max_epochs=10,
        deterministic=True,
        callbacks=[callback_topmodel],
    )

    trainer.fit(model_finetune, data_module)
    trainer.test(model_finetune, data_module)

    # wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Uncertainty Model with a specific seed")
    parser.add_argument("seed", type=int, help="The seed to use for training")
    args = parser.parse_args()

    fit_uncertainty_model(args.seed)
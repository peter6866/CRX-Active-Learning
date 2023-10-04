import time
import os

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
import peft
from peft import LoraConfig

from datamodules.activityDataModule import ActivityDataModule

from models.peftUncertaintyModel import PeftUncertaintyModel


BATCH_SIZE = 32
MAX_EPOCHS = 80
LEARNING_RATE = 5e-3
TRAINING_ON = "Genomic"
# Import the tokenizer and the model
pretrained_model_name = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model_O = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2, hidden_dropout_prob=0.3)

data_dir = "Data/activity_summary_stats_and_metadata.txt"
retino_dir = "Data/retinopathy_reformatted.txt"

# LORA config
lora_config = LoraConfig(
    task_type="SEQ_CLS",
    lora_alpha=32,
    lora_dropout=0.1,
    r=16,
    bias="none",
    target_modules=["query", "value", "dense"],
    inference_mode=False,
)
model_O = peft.get_peft_model(model_O, lora_config)

pl.seed_everything(42)

wandb.login()

wandb_logger = WandbLogger(
    project='BCLab_llm_uncertainty',
    name=time.strftime('%Y-%m-%d-%H-%M'),
    group=TRAINING_ON,
    job_type="fit"
    )

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    every_n_epochs=MAX_EPOCHS  # Save checkpoint on the last epoch
)

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator='gpu',
    devices=-1,
    max_epochs=MAX_EPOCHS,
    deterministic=True,
    fast_dev_run=False,
    strategy="ddp_find_unused_parameters_true",
    callbacks=[checkpoint_callback],
    # precision=16
    )

data_module = ActivityDataModule(
     data_path=data_dir,
     retinopathy_path=retino_dir,
     tokenizer=tokenizer,
     validate_type=TRAINING_ON,
     batch_size=BATCH_SIZE)

model = PeftUncertaintyModel(model_O, len(data_module.train_dataloader()), MAX_EPOCHS, lr=LEARNING_RATE)

torch.set_float32_matmul_precision('high')
trainer.fit(model, data_module)

wandb.finish()

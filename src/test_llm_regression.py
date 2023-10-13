import time
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
import peft
from peft import IA3Config, LoraConfig

from datamodules.activityDataset import ActivityDataset
from datamodules.activityDataModule import ActivityDataModule

from models.peftRegressionModel import PeftRegressionModel

from mpra_tools import plot_utils


CHECKPOINT_DIR = '/scratch/bclab/jiayu/CRX-Active-Learning/BCLab/3q3o1p0v/checkpoints/epoch=79-step=22160.ckpt'
OUTPUT_DIR = "/scratch/bclab/jiayu/CRX-Active-Learning/ModelFitting/transformers"
BATCH_SIZE = 32
MAX_EPOCHS = 1
TESTING_ON = "Round4b"

# Import the tokenizer and the model
pretrained_model_name = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model_O = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=1, hidden_dropout_prob=0.3)

pl.seed_everything(42)

wandb.login()

wandb_logger = WandbLogger(
    project='BCLab',
    name=time.strftime('%Y-%m-%d-%H-%M')+ "-Test",
    group=TESTING_ON,
    job_type="test"
    )

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator='gpu',
    devices=-1,
    max_epochs=1,
    deterministic=True,
    fast_dev_run=False,
    )

data_dir = "Data/activity_summary_stats_and_metadata.txt"
retino_dir = "Data/retinopathy_reformatted.txt"

train_dataset = ActivityDataset(data_path=data_dir, retinopathy_path=retino_dir, data_type="train", tokenizer=tokenizer)
val_dataset = ActivityDataset(data_path=data_dir, retinopathy_path=retino_dir, data_type="validate", tokenizer=tokenizer)

data_module = ActivityDataModule(
     data_path=data_dir,
     retinopathy_path=retino_dir,
     tokenizer=tokenizer,
     validate_type=TESTING_ON,
     batch_size=BATCH_SIZE)

# val_data_module = ActivityDataModule(
#      data_path=data_dir,
#      retinopathy_path=retino_dir,
#      tokenizer=tokenizer, 
#      validate_type=TESTING_ON,
#      batch_size=BATCH_SIZE,
#      pred_type="val")

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

model = PeftRegressionModel.load_from_checkpoint(
    CHECKPOINT_DIR, 
    model_name=model_O, 
    train_dataloader_len=len(data_module.train_dataloader()), 
    num_epochs=MAX_EPOCHS
)

torch.set_float32_matmul_precision('high')

trainer.test(model, data_module)

# train_preds_batch = trainer.predict(model, data_module)
# val_preds_batch = trainer.predict(model, val_data_module)

# train_preds = []
# for batch in train_preds_batch:
#     for i in range(len(batch)):
#         train_preds.append(batch[i][0].item())
# train_preds = np.array(train_preds)
# val_preds = []
# for batch in val_preds_batch:
#      for i in range(len(batch)):
#           val_preds.append(batch[i][0].item())
# val_preds = np.array(val_preds)

# preds = {
#      "train": train_preds,
#      "val": val_preds
# }

# train_truth = []
# for i in range(len(train_dataset)):
#     train_truth.append(train_dataset[i]["labels"])
# train_truth = np.array(train_truth)
# val_truth = []
# for i in range(len(val_dataset)):
#      val_truth.append(val_dataset[i]["labels"])
# val_truth = np.array(val_truth)

# truth = {
#     "train": train_truth,
#     "val": val_truth
# }

# for name in ["train", "val"]:
        
#         fig, ax = plt.subplots(figsize=plot_utils.get_figsize())
#         fig, ax, corrs = plot_utils.scatter_with_corr(
#             truth[name],
#             preds[name],
#             "Observed",
#             "Predicted",
#             colors="density",
#             loc="upper left",
#             figax=(fig, ax),
#             rasterize=True,
#         )
#         # Show y = x
#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         vmin = max(xlim[0], ylim[0])
#         vmax = min(xlim[1], ylim[1])
#         ax.plot([vmin, vmax], [vmin, vmax], color="k", linestyle="--", lw=1)
#         plot_utils.save_fig(fig, os.path.join(OUTPUT_DIR, f"{name}PredVsObs"))
#         plt.close()

wandb.finish()

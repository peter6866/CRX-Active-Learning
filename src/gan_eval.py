import torch
import pytorch_lightning as pl
from models.WGAN_Model import WGAN
import numpy as np
import os
import pandas as pd


def generate_one_seq(model):
    noise = torch.randn(1, 100).to("cuda")
    sample = model(noise)
    _, indices = torch.max(sample, dim=1)
    bases = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    dna_sequence = ''.join(bases[i.item()] for i in indices[0])
    return dna_sequence


SEQ_LEN = 200
CKPT_DIR = "/scratch/bclab/jiayu/al/BCLab-WGAN/5e7gdvnk/checkpoints/epoch=1499-step=6165000.ckpt"
SAVE_DIR = "ModelFitting/GAN"

model = WGAN.load_from_checkpoint(
    CKPT_DIR,
    seq_len=SEQ_LEN
)

model.eval()

generated_seqs = []
for i in range(10000):
    generated_seqs.append(generate_one_seq(model))
    
# Save generated sequences to a csv file
df = pd.DataFrame(generated_seqs)
df.to_csv(os.path.join(SAVE_DIR, "generated_seqs_1500.csv"), index=False, header=False)

print("Done!")

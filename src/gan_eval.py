import torch
import pytorch_lightning as pl
from models.WGAN_Model import WGAN
import numpy as np
import os
import pandas as pd

SEQ_LEN = 227
CKPT_DIR = "/scratch/bclab/jiayu/CRX-Active-Learning/BCLab-WGAN/doxfqs49/checkpoints/improved.ckpt"
SAVE_DIR = "ModelFitting/GAN"

model = WGAN.load_from_checkpoint(
    CKPT_DIR,
    seq_len=SEQ_LEN
)
model = model.to("cuda")
model.eval()

# Generate sequences in batches
batch_size = 100
num_sequences = 1000000
generated_seqs = []

# Handle full batches
num_full_batches = num_sequences // batch_size
remainder = num_sequences % batch_size

print("Generating sequences in batches...")

# Process full batches
for i in range(num_full_batches):
    if i % 100 == 0:  # Adjusted progress reporting
        print(f"Generating sequence {i * batch_size + 1}...")
    noise = torch.randn(batch_size, 100, device="cuda")
    with torch.no_grad():
        samples = model(noise)
    _, indices = torch.max(samples, dim=1)
    
    for idx in indices:
        dna_sequence = ''.join({0: 'A', 1: 'C', 2: 'G', 3: 'T'}[i.item()] for i in idx)
        generated_seqs.append(dna_sequence)

# Handle the remainder batch
if remainder > 0:
    noise = torch.randn(remainder, 100, device="cuda")
    with torch.no_grad():
        samples = model(noise)
    _, indices = torch.max(samples, dim=1)
    
    for idx in indices:
        dna_sequence = ''.join({0: 'A', 1: 'C', 2: 'G', 3: 'T'}[i.item()] for i in idx)
        generated_seqs.append(dna_sequence)

df = pd.DataFrame(generated_seqs)
df.to_csv(os.path.join(SAVE_DIR, "atac_gen_1m.csv"), index=False, header=False)

print("Done!")
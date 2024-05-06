import torch
from torch.utils.data import Dataset
import pandas as pd
import selene_sdk
import numpy as np


def load_data(file, index_col=0, **kwargs):
    """Wrapper for reading in an arbitrary tab-delimited file as a DataFrame/Series. Assumes the first column is the
    index column. Extra arguments for pd.read_csv."""
    return pd.read_csv(file, sep="\t", index_col=index_col, na_values="NaN", **kwargs)

def one_hot_encode(seqs):
    """Given a list of sequences, one-hot encode them.

    Parameters
    ----------
    seqs : list-like
        Each entry is a DNA sequence to be one-hot encoded

    Returns
    -------
    seqs_hot : ndarray, shape (number of sequences, 4, length of sequence)
    """
    seqs_hot = list()
    for seq in seqs:
        seqs_hot.append(
            selene_sdk.sequences.Genome.sequence_to_encoding(seq).T
        )
    seqs_hot = np.stack(seqs_hot)
    return seqs_hot


class ganDataset(Dataset):
    
    def __init__(self, data_path):
        # sequence_key = "sequence"
        sequence_key = "sequence"
        # activity_df = load_data(data_path)
        activity_df = pd.read_csv(data_path)

        batches_to_use = [
            "Genomic",
            # "CrxMotifMutant",
            # "Round2",
            # "Round3a",
            # "Round3b",
            # "Round4b",
        ]
        
        # data_df = activity_df[activity_df["cnn_train"] & activity_df["data_batch_name"].isin(batches_to_use)]
        self.sequences = activity_df[sequence_key]
        
        self.seqs_hot = one_hot_encode(self.sequences)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.seqs_hot[idx]

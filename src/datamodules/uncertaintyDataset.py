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

class UncertaintyDataset(Dataset):
    
    def __init__(self, data_path, retinopathy_path, data_type, sample_type=None):
        """
        Initialize the ActivityDataset.

        Parameters:
        - data_path: path to the activity data.
        - retinopathy_path: path to the retinopathy data.
        - data_type: type of data ("train", "validate", "test", etc.).
        - tokenizer: tokenizer instance for tokenization.
        """
        sequence_key = "sequence"
        activity_key = "expression_log2_new"
        # activity_df = load_data(data_path)
        activity_df = pd.read_csv(data_path)
        
        batches_to_use = [
            "Genomic",
            "CrxMotifMutant",
            "Round2",
            "Round3a",
            "Round3b",
            "Round4b",
        ]
        
        if data_type == "train":
            data_df = activity_df[activity_df["cnn_train"] & activity_df["data_batch_name"].isin(batches_to_use)]
        elif data_type == "validate":
            data_df = activity_df[activity_df["cnn_validation_set"]]
        elif data_type == "validate_genomic":
            data_df = activity_df[activity_df["cnn_validation_set"] & activity_df["original_genomic"]]
        elif data_type == "test":
            data_df = activity_df[activity_df["test_set"]]
        elif data_type == "test_retinopathy":
            # data_df = load_data(retinopathy_path)
            data_df = pd.read_csv(retinopathy_path)
        elif data_type == "test_generated":
            data_df = pd.read_csv("Data/generated_seqs_genomic_leaky.csv", header=None)
            data_df = data_df.rename(columns={0: "sequence"})
            data_df["expression_log2"] = list(np.zeros(len(data_df)))
        elif data_type == "test_mutant":
            data_df = activity_df[activity_df["cnn_train"] & activity_df["data_batch_name"].isin(["Genomic"])]
        else:
            raise ValueError(f"Invalid data_type provided: {data_type}")

        sequences = data_df[sequence_key]
        
        if sample_type is not None:
            if sample_type == "random":
                sample_df = pd.read_csv("Data/Sampling_Test/all_samples_round2.csv")
                round1_df = pd.read_csv("Data/Sampling_Test/all_samples.csv")
                sample_df = pd.concat([sample_df, round1_df], ignore_index=True).sample(n=15000)
            elif sample_type == "uncertainty":
                sample_df = pd.read_csv("Data/Sampling_Test/largest_samples_round2.csv")
                round1_df = pd.read_csv("Data/Sampling_Test/largest_samples.csv")
                sample_df = pd.concat([sample_df, round1_df], ignore_index=True)
            elif sample_type == "bin_uncertainty":
                sample_df = pd.read_csv("Data/Sampling_Test/largest_bin_samples_round2.csv")
                round1_df = pd.read_csv("Data/Sampling_Test/largest_bin_samples.csv")
                sample_df = pd.concat([sample_df, round1_df], ignore_index=True)
            elif sample_type == "distance_uncertainty":
                sample_df = pd.read_csv("Data/Sampling_Test/largest_samples_distance.csv")

            full_seqs = pd.concat([sequences, sample_df["sequence"]], ignore_index=True)
            full_target = pd.concat([data_df[activity_key], sample_df["expression_log2"]], ignore_index=True)

            self.seqs_hot = one_hot_encode(full_seqs)
            self.target = torch.tensor(full_target.values, dtype=torch.float)

        else:
            self.seqs_hot = one_hot_encode(sequences)
            self.target = torch.tensor(data_df[activity_key].values, dtype=torch.float)
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        x = self.seqs_hot[idx]
        y = self.target[idx]
        
        return x, y

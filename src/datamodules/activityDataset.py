import torch
from torch.utils.data import Dataset
import pandas as pd


def load_data(file, index_col=0, **kwargs):
    """Wrapper for reading in an arbitrary tab-delimited file as a DataFrame/Series. Assumes the first column is the
    index column. Extra arguments for pd.read_csv."""
    return pd.read_csv(file, sep="\t", index_col=index_col, na_values="NaN", **kwargs)


class ActivityDataset(Dataset):
    
    def __init__(self, data_path, retinopathy_path, data_type, tokenizer=None):
        """
        Initialize the ActivityDataset.

        Parameters:
        - data_path: path to the activity data.
        - retinopathy_path: path to the retinopathy data.
        - data_type: type of data ("train", "validate", "test", etc.).
        - tokenizer: tokenizer instance for tokenization.
        """
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided.")
        
        self.tokenizer = tokenizer
        self.sequence_key = "sequence"
        activity_key = "expression_log2"
        activity_df = load_data(data_path)
        
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
            data_df = load_data(retinopathy_path)
        else:
            raise ValueError(f"Invalid data_type provided: {data_type}")

        self.sequence = data_df[self.sequence_key]
        self.target = torch.tensor(data_df[activity_key].values, dtype=torch.float)

        # Tokenize the sequences during initialization
        self.encodings = self.tokenizer.batch_encode_plus(
            list(self.sequence),
            add_special_tokens=True,
            max_length=35,
            truncation=True,
            return_token_type_ids=False,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
    def __len__(self):
        return len(self.sequence)
    
    def __getitem__(self, idx):
        # Use the pre-tokenized sequences and attention masks
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]

        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': self.target[idx]
        }

        return res

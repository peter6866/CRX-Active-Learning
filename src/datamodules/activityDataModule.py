import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .activityDataset import ActivityDataset


class ActivityDataModule(pl.LightningDataModule):
    def __init__(self, data_path, retinopathy_path, tokenizer, validate_type, batch_size):
        super(ActivityDataModule, self).__init__()
        self.data_path = data_path
        self.retinopathy_path = retinopathy_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.validate_type = validate_type
     
        # Create datasets
        self.train_dataset = ActivityDataset(self.data_path, self.retinopathy_path, "train", self.tokenizer)
        self.validate_dataset = ActivityDataset(self.data_path, self.retinopathy_path, "validate", self.tokenizer)
        self.validate_genomic_dataset = ActivityDataset(self.data_path, self.retinopathy_path, "validate_genomic",
                                                        self.tokenizer)
        self.test_dataset = ActivityDataset(self.data_path, self.retinopathy_path, "test", self.tokenizer)
        self.test_retinopathy_dataset = ActivityDataset(self.data_path, self.retinopathy_path, "test_retinopathy",
                                                        self.tokenizer)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        # Return multiple validation loaders
        if self.validate_type == "Round4b":
            return DataLoader(self.validate_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False, drop_last=True, pin_memory=True)
        else:
            return DataLoader(self.validate_genomic_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False, drop_last=True, pin_memory=True)

    def test_dataloader(self):
       
        return [
            DataLoader(self.test_retinopathy_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False, pin_memory=True),
            DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False, pin_memory=True)
        ]
        

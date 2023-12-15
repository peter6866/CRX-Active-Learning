import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .uncertaintyDataset import UncertaintyDataset


class UncertaintyDataModule(pl.LightningDataModule):
    def __init__(self, data_path, retinopathy_path, validate_type, batch_size, pred_type="train", sample_type=None):
        super(UncertaintyDataModule, self).__init__()
        self.data_path = data_path
        self.retinopathy_path = retinopathy_path
        self.batch_size = batch_size
        self.validate_type = validate_type
        self.pred_type = pred_type
        # Create datasets
        self.train_dataset = UncertaintyDataset(self.data_path, self.retinopathy_path, "train", sample_type=sample_type)
        self.validate_dataset = UncertaintyDataset(self.data_path, self.retinopathy_path, "validate", sample_type=sample_type)
        self.validate_genomic_dataset = UncertaintyDataset(self.data_path, self.retinopathy_path, "validate_genomic", sample_type=sample_type)
        self.test_dataset = UncertaintyDataset(self.data_path, self.retinopathy_path, "test", sample_type=sample_type)
        self.test_retinopathy_dataset = UncertaintyDataset(self.data_path, self.retinopathy_path, "test_retinopathy", sample_type=sample_type)
        self.test_generated_dataset = UncertaintyDataset(self.data_path, self.retinopathy_path, "test_generated", sample_type=sample_type)
        self.test_mutant_dataset = UncertaintyDataset(self.data_path, self.retinopathy_path, "test_mutant", sample_type=sample_type)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    def val_dataloader(self):
        # Return multiple validation loaders
        if self.validate_type == "Round4b":
            return DataLoader(self.validate_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False, drop_last=True, pin_memory=True)
        else:
            return DataLoader(self.validate_genomic_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False, drop_last=True, pin_memory=True)

    def test_dataloader(self):
       
        return [
            DataLoader(self.test_retinopathy_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False, pin_memory=True),
            DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False, pin_memory=True),
            DataLoader(self.test_generated_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False, pin_memory=True),
            DataLoader(self.test_mutant_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False, pin_memory=True)
        ]
        
    def predict_dataloader(self):
        if self.pred_type == "train":
            return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        elif self.pred_type == "val" and self.validate_type == "Round4b":
            return DataLoader(self.validate_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False, pin_memory=True)
        else:
            return DataLoader(self.validate_genomic_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False)
    

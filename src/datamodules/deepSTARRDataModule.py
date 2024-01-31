import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .deepSTARRDataset import DeepSTARRDataset


class DeepSTARRDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super(DeepSTARRDataModule, self).__init__()
        self.batch_size = batch_size
        # Create datasets
        self.train_dataset = DeepSTARRDataset("train")
        self.val_dataset = DeepSTARRDataset("val")
        self.test_dataset = DeepSTARRDataset("test")
        
    def setup(self, stage=None):
        pass
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False, pin_memory=True)
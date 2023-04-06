import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class PreextractedFeatureDataset(Dataset):
    def __init__(self, manifest_path):
        pass

    def __len__(self):
        pass

    def __getitem__(self, ix):
        pass


class DataModule(pl.LightningDataModule):
    def __init__(self, train_manifest_path, val_manifest_path, test_manifest_path=None):
        pass

    def setup(self, stage="fit"):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

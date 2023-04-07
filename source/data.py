import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class PreextractedFeatureDataset(Dataset):
    def __init__(self, manifest_path, config):
        manifest = pd.read_csv(manifest_path)
        
        self.target = config['target'] if 'target' in config.keys() else 'label'
        self.data = manifest[['slide_id', self.target]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        case = self.data.iloc[ix]
        x,y = case['slide_id'], int(case[self.target])

        return torch.load(x), torch.tensor(y)


class DataModule(pl.LightningDataModule):
    def __init__(self, manifest_directory, config):
        pass

    def setup(self, stage="fit"):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

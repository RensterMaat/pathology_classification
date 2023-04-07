import torch
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
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
        self.manifest_directory = Path(manifest_directory)
        self.config = config

    def setup(self, stage="fit"):
        if stage == 'fit':
            self.train_dataset = PreextractedFeatureDataset(
                self.manifest_directory / 'train.csv',
                self.config
            )
            self.val_dataset = PreextractedFeatureDataset(
                self.manifest_directory / 'val.csv',
                self.config
            )

        if stage == 'test':
            self.test_dataset = PreextractedFeatureDataset(
                self.manifest_directory / 'test.csv',
                self.config
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=self.config['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.config['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.config['num_workers'])

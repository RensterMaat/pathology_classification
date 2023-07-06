import os
import torch
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, Dataset


class PreextractedFeatureDataset(Dataset):
    def __init__(self, manifest_path: os.PathLike, config: dict) -> None:
        manifest = pd.read_csv(manifest_path)
        self.config = config

        self.target = config["target"] if "target" in config.keys() else "label"
        self.data = manifest[["slide_id", self.target]]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, ix: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        case = self.data.iloc[ix]

        features_path = str(
            Path(self.config["features_dir"]) / (case["slide_id"] + ".pt")
        )

        x = torch.load(features_path).float()
        y = torch.zeros(self.config["n_classes"]).float()
        y[int(case[self.target])] = 1

        return x, y, features_path


class DataModule(pl.LightningDataModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.manifest_directory = (
            Path(config["manifest_dir"]) / f"fold_{config['fold']}"
        )
        self.config = config

    def setup(self, stage="fit") -> None:
        if stage == "fit":
            self.train_dataset = PreextractedFeatureDataset(
                self.manifest_directory / "train.csv", self.config
            )
            self.val_dataset = PreextractedFeatureDataset(
                self.manifest_directory / "tune.csv", self.config
            )

        if stage == "test":
            self.test_dataset = PreextractedFeatureDataset(
                self.manifest_directory / "test.csv", self.config
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.config["num_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config["num_workers"],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config["num_workers"],
        )

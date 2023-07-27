import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from source.utils.utils import (
    get_slide,
    get_coordinates,
    scale_coordinates,
    get_patch_coordinates_dir_name,
)
from pathlib import Path
from openslide import OpenSlide


class CrossSectionDataset(Dataset):
    def __init__(self, patch_coordinates_file_name, config) -> None:
        self.config = config
        self.patch_coordinates_file_name = patch_coordinates_file_name

        self.setup_patch_coordinates()
        self.setup_slide()

    def __len__(self) -> int:
        return len(self.patch_coordinates)
        # return 10

    def __getitem__(self, ix: int) -> torch.Tensor:
        img = self.slide.read_region(
            location=self.patch_coordinates[ix][1],
            level=self.config["level_during_feature_extraction"],
            size=self.patch_coordinates[ix][2],
        )

        out = torch.tensor(np.array(img)).float()[:, :, :-1].permute((2, 0, 1)) / 255
        return out

    def setup_patch_coordinates(self):
        patch_coordinates_dir = Path(
            self.config["patch_coordinates_dir"]
        ) / get_patch_coordinates_dir_name(self.config)
        patch_coordinates_file_path = (
            patch_coordinates_dir / self.patch_coordinates_file_name
        )

        with open(patch_coordinates_file_path) as f:
            self.patch_coordinates = json.load(f)

    def setup_slide(self):
        slide_file_name = (
            self.patch_coordinates_file_name.split("_cross_section")[0] + ".ndpi"
        )
        slide_file_path = Path(self.config["slides_dir"]) / slide_file_name
        self.slide = OpenSlide(slide_file_path)


class ExtractionDataModule(pl.LightningDataModule):
    def __init__(self, patch_coordinates_file_name: str, config: dict) -> None:
        super().__init__()
        self.patch_coordinates_file_name = patch_coordinates_file_name
        self.config = config

    def setup(self, stage: str = None) -> None:
        self.dataset = CrossSectionDataset(
            self.patch_coordinates_file_name, self.config
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config["extraction_batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )

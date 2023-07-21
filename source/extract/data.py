import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from source.utils.utils import get_slide, get_coordinates, scale_coordinates


class PatchSlideDataset(Dataset):
    def __init__(self, slide_id: str, config: dict) -> None:
        self.slide_id = slide_id
        self.config = config

        self.slide = get_slide(slide_id, config["slide_dir"])
        coordinates = get_coordinates(slide_id, config["patch_coordinate_dir"])
        self.scaled_coordinates = scale_coordinates(
            coordinates, self.slide, config["level_during_feature_extraction"]
        )

    def __len__(self) -> int:
        return len(self.scaled_coordinates)

    def __getitem__(self, ix: int) -> torch.Tensor:
        origin = self.scaled_coordinates[ix].astype(int)
        img = self.slide.read_region(
            location=origin,
            level=self.config["level_during_feature_extraction"],
            size=[self.config["patch_size_during_feature_extraction"]] * 2,
        )

        return torch.tensor(np.array(img)).float()[0, :, :-1]


class ExtractionDataModule(pl.LightningDataModule):
    def __init__(self, slide_id: str, config: dict) -> None:
        super().__init__()
        self.slide_id = slide_id
        self.config = config

    def setup(self, stage:str = None) -> None:
        self.dataset = PatchSlideDataset(self.slide_id, self.config)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config["extraction_batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )

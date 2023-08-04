import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pathlib import Path
from openslide import OpenSlide
from torchvision.io import read_image

from source.utils.utils import (
    get_patch_coordinates_dir_name,
    get_tiles_dir_name,
)


class CrossSectionDataset(Dataset):
    """
    Dataset for extracting features from a cross section of a slide.

    Attributes:
        config (dict): Configuration dictionary.
        patch_coordinates (list): List of coordinates of the patches.
        tile_images_folder_path (OpenSlide): Path to the folder containing the
            pre-extracted images of the cross section
    """

    def __init__(self, patch_coordinates_file_name: str, config: dict) -> None:
        """
        Initialize the dataset.

        Patches are indexed based on the order in which they are stored in the file
        containing the coordinates of the patches. This way, output features can be
        related to their location in the cross section.

        Args:
            patch_coordinates_file_name (os.PathLike): Name of the file containing
                the coordinates of the patches.
            config (dict): Configuration dictionary.

        Returns:
            None
        """
        self.config = config

        patch_coordinates_dir = get_patch_coordinates_dir_name(self.config)
        patch_coordinates_file_path = (
            patch_coordinates_dir / patch_coordinates_file_name
        )

        with open(patch_coordinates_file_path) as f:
            self.patch_coordinates = json.load(f)

        self.tile_images_folder_path = get_tiles_dir_name(self.config) / ".".join(
            patch_coordinates_file_name.split(".")[:-1]
        )

    def __len__(self) -> int:
        """
        Get the number of patches.

        Returns:
            int: Number of patches.
        """
        return len(self.patch_coordinates)

    def __getitem__(self, ix: int) -> torch.Tensor:
        """
        Get a patch.

        Args:
            ix (int): Index of the patch.

        Returns:
            torch.Tensor: Patch of dimensions (3, patch_size, patch_size) and type float with values in [0, 1].
        """
        x, y = self.patch_coordinates[ix][1]
        patch_image_file_name = self.tile_images_folder_path.name + f"_{x}_{y}.png"

        img = read_image(str(self.tile_images_folder_path / patch_image_file_name))

        out = (img[:3] / 255.0).float()

        return out


class ExtractionDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning compatible data module for extracting features from a cross section of a slide.

    The data module is used by the trainer during extraction.

    Attributes:
        patch_coordinates_file_name (str): Name of the file containing the coordinates of the patches.
        config (dict): Configuration dictionary.
        dataset (CrossSectionDataset): Dataset for extracting features from a cross section of a slide.
    """

    def __init__(self, patch_coordinates_file_name: str, config: dict) -> None:
        """
        Initialize the data module.

        Args:
            patch_coordinates_file_name (str): Name of the file containing the coordinates of the patches.
            config (dict): Configuration dictionary.

        Returns:
            None
        """
        super().__init__()
        self.patch_coordinates_file_name = patch_coordinates_file_name
        self.config = config

    def setup(self, stage: str = None) -> None:
        """
        Setup the data module.

        Args:
            stage (str): Stage of the training process. Can be "fit" or "test", but is always
            "test" for this data module.
        """
        self.dataset = CrossSectionDataset(
            self.patch_coordinates_file_name, self.config
        )

    def test_dataloader(self) -> DataLoader:
        """
        Get the data loader for the test set.

        Returns:
            DataLoader: Data loader for the test set.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.config["extraction_batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )

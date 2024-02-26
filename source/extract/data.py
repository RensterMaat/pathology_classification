import os
import json
import torch
import numpy as np
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pathlib import Path
from PIL import Image

from source.utils.utils import (
    get_patch_coordinates_dir_name,
)


class CrossSectionDataset(Dataset):
    """
    Dataset for extracting features from a cross section of a slide.

    Attributes:
        config (dict): Configuration dictionary.
        patch_coordinates_file_name (os.PathLike): Name of the file containing the coordinates of the patches.
        patch_coordinates (list): List of coordinates of the patches.
        slide (OpenSlide): Slide from which the patches were extracted.
    """

    def __init__(self, patch_coordinates_file_name: os.PathLike, config: dict) -> None:
        """
        Initialize the dataset.

        Args:
            patch_coordinates_file_name (os.PathLike): Name of the file containing the coordinates of the patches.
            config (dict): Configuration dictionary.

        Returns:
            None
        """
        self.config = config
        self.patch_coordinates_file_name = patch_coordinates_file_name

        self.setup_patch_coordinates()
        self.setup_patches()

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
        patch_file_name = f"{x}_{y}.jpg"

        patch_image = Image.open(self.patch_images_dir / patch_file_name)

        out = torch.tensor(np.array(patch_image)).float().permute((2, 0, 1)) / 255
        return out

    def setup_patch_coordinates(self):
        """
        Load the patch coordinates based on the provided patch coordinate file name.

        Returns:
            None
        """
        patch_coordinates_dir = (
            Path(self.config["output_dir"])
            / "patch_coordinates"
            / get_patch_coordinates_dir_name(self.config)
        )

        patch_coordinates_file_path = (
            patch_coordinates_dir / self.patch_coordinates_file_name
        )

        with open(patch_coordinates_file_path) as f:
            self.patch_coordinates = json.load(f)

    def setup_patches(self):
        """
        Load the patches based on the provided patch coordinate file name.

        Returns:
            None
        """

        self.patch_images_dir = (
            Path(self.config["output_dir"])
            / "tiles"
            / get_patch_coordinates_dir_name(self.config).name
            / self.patch_coordinates_file_name.split(".json")[0]
        )


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
            num_workers=(
                self.config["num_workers"]
                if "num_workers" in self.config
                and self.config["num_workers"] is not None
                else mp.cpu_count()
            ),
        )

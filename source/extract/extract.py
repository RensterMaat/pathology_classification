import os
import torch
import argparse
import pytorch_lightning as pl
from tqdm import tqdm

from source.extract.extractor_models import (
    RandomExtractor,
    ResNet50ImagenetExtractor,
    PatchLevelHIPTFeatureExtractor,
    RegionLevelHIPTFeatureExtractor,
    PLIPFeatureExtractor,
)
from source.extract.data import ExtractionDataModule
from source.utils.utils import (
    load_general_config,
    load_specific_config,
    get_features_dir_name,
    get_patch_coordinates_dir_name,
)


class ExtractorFramework(pl.LightningModule):
    """
    Framework for extracting features from patches.

    This class implements the pytorch lightning LightningModule interface
    and contains boilerplate code for extracting patch level features from
    histopathology slides given an extractor model.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the ExtractorFramework.

        Args:
            config (dict): Configuration dictionary. See load_config() in
                source/utils/utils.py for more information.
        """
        super().__init__()

        self.config = config

        if config["extractor_model"] == "random":
            self.extractor = RandomExtractor(config)
        elif config["extractor_model"] == "resnet50_imagenet":
            self.extractor = ResNet50ImagenetExtractor(config)
        elif config["extractor_model"] == "patch_hipt":
            self.extractor = PatchLevelHIPTFeatureExtractor(config)
        elif config["extractor_model"] == "region_hipt":
            self.extractor = RegionLevelHIPTFeatureExtractor(config)
        elif config["extractor_model"] == "plip":
            self.extractor = PLIPFeatureExtractor(config)
        else:
            raise ValueError(
                f"Extractor model {config['extractor_model']} not recognized."
            )

        self.save_dir = get_features_dir_name(self.config)
        self.all_features = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the extractor model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_features).

        """
        x_transformed = self.extractor.transform(x)
        batch_features = self.extractor.forward(x_transformed)

        return batch_features

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Test step for pytorch lightning.

        Args:
            batch (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_features).
        """
        batch_features = self.forward(batch).to("cpu").detach()

        self.all_features.append(batch_features)

        return batch_features

    def save_features(self, features_file_name: os.PathLike) -> None:
        """
        Save the extracted features to disk.

        Args:
            features_file_name (str): Name of the file to save the features to.

        Returns:
            None
        """

        torch.save(
            torch.cat(self.all_features), self.save_dir / (features_file_name + ".pt")
        )

        self.all_features = []


def main(config):
    """
    Extract features from patches.

    Args:
        config (dict): Configuration dictionary. See load_config() in
            source/utils/utils.py for more information.

    Returns:
        None
    """
    trainer = pl.Trainer(accelerator="gpu", enable_progress_bar=True)
    extractor = ExtractorFramework(config)

    if not extractor.save_dir.exists():
        extractor.save_dir.mkdir(parents=True)

    coordinates_dir = get_patch_coordinates_dir_name(config)

    not_yet_processed = [
        cross_section
        for cross_section in coordinates_dir.glob("*.json")
        if not (extractor.save_dir / (cross_section.stem + ".pt")).exists()
    ]

    for cross_section in tqdm(not_yet_processed):
        datamodule = ExtractionDataModule(cross_section.name, config)
        trainer.test(extractor, datamodule)
        extractor.save_features(cross_section.stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from patches.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/general/umcu.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    general_config = load_general_config(args.config)
    specific_config = load_specific_config(args.config, "extract")

    config = general_config | specific_config

    main(config)

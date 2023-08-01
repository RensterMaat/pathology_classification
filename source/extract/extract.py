import torch
import argparse
import pytorch_lightning as pl
from source.extract.extractor_models import (
    RandomExtractor,
    ResNet50ImagenetExtractor,
    PatchLevelHIPTFeatureExtractor,
    RegionLevelHIPTFeatureExtractor,
)
from source.extract.data import ExtractionDataModule
from source.utils.utils import (
    load_config,
    get_features_dir_name,
    get_patch_coordinates_dir_name,
)
from pathlib import Path
from tqdm import tqdm


class ExtractorFramework(pl.LightningModule):
    def __init__(self, config: dict) -> None:
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
        else:
            raise ValueError(
                f"Extractor model {config['extractor_model']} not recognized."
            )

        self.all_features = []

    def forward(self, x):
        x_transformed = self.extractor.transform(x)
        batch_features = self.extractor.forward(x_transformed)

        return batch_features

    def test_step(self, batch, batch_idx):
        batch_features = self.forward(batch)

        self.all_features.append(batch_features)

        return batch_features

    def save_features(self, features_file_name):
        save_dir = get_features_dir_name(self.config)

        if not save_dir.exists():
            save_dir.mkdir()

        torch.save(
            torch.cat(self.all_features), save_dir / (features_file_name + ".pt")
        )


def main(config):
    trainer = pl.Trainer(accelerator="cpu", enable_progress_bar=True)

    extractor = ExtractorFramework(config)

    coordinates_dir = get_patch_coordinates_dir_name(config)

    for cross_section in tqdm(list(coordinates_dir.glob("*.json"))):
        datamodule = ExtractionDataModule(cross_section.name, config)
        trainer.test(extractor, datamodule)
        extractor.save_features(cross_section.stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from patches.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    main(config)

import yaml
from pytorch_lightning import Trainer
from source.extract.extract import ExtractorFramework
from source.extract.data import ExtractionDataModule
from pathlib import Path
from source.utils.utils import get_patch_coordinates_dir_name, load_config


config = load_config("config/default.yaml")


def test_extract():
    trainer = Trainer(
        accelerator="cpu",
    )

    patch_coordinates_dir = Path(
        config["patch_coordinates_dir"]
    ) / get_patch_coordinates_dir_name(config)
    patch_coordinates_file_name = list(patch_coordinates_dir.iterdir())[0].name

    datamodule = ExtractionDataModule(patch_coordinates_file_name, config)
    extractor = ExtractorFramework(config)

    trainer.test(extractor, datamodule)

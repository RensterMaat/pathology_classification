import yaml
from pytorch_lightning import Trainer
from source.extract.extractor_framework import ExtractorFramework
from source.extract.data import ExtractionDataModule
from pathlib import Path

with open("config/classify.yaml", "r") as f:
    config = yaml.safe_load(f)


def test_extract():
    trainer = Trainer(
        accelerator="cpu",
    )

    slide_dir = Path(config["slide_dir"]) / "primary" / "vumc"

    slide_path = list(slide_dir.iterdir())[0]
    slide_id = slide_path.stem

    datamodule = ExtractionDataModule(slide_id, config)
    extractor = ExtractorFramework(config)

    trainer.test(extractor, datamodule)

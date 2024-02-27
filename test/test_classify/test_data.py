import yaml
from pathlib import Path
from source.classify.classify_data import PreextractedFeatureDataset, ClassificationDataModule

with open("config/classify.yaml", "r") as f:
    config = yaml.safe_load(f)

config["fold"] = 0


def test_dataset():
    dataset = PreextractedFeatureDataset(
        Path(config["cross_val_splits_dir"]) / f'fold_{config["fold"]}' / "train.csv",
        config,
    )

    x, y, _ = dataset[0]


def test_datamodule():
    datamodule = ClassificationDataModule(config)

    datamodule.setup(stage="fit")
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()

    datamodule.setup(stage="test")
    test_dl = datamodule.test_dataloader()

    x, y, _ = next(iter(train_dl))

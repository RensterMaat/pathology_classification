import yaml
from source.data import PreextractedFeatureDataset, DataModule

with open("config/test.yaml", "r") as f:
    config = yaml.safe_load(f)


def test_dataset():
    dataset = PreextractedFeatureDataset(
        "/mnt/hpc/rens/hipt/data/fold_dir_dcb/fold_0/train.csv", config
    )

    x, y, _ = dataset[0]


def test_datamodule():
    datamodule = DataModule(config)

    datamodule.setup(stage="fit")
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()

    datamodule.setup(stage="test")
    test_dl = datamodule.test_dataloader()

    x, y, _ = next(iter(train_dl))

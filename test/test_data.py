from source.data import PreextractedFeatureDataset, DataModule


def test_dataset():
    config = {"target": "label", 'n_classes':3}

    dataset = PreextractedFeatureDataset(
        "/mnt/hpc/rens/hipt/data/fold_dir_test/fold_0/train.csv", config
    )

    x, y = dataset[0]


def test_datamodule():
    config = {"target": "label", 'n_classes':3, "num_workers": 1}

    datamodule = DataModule("/mnt/hpc/rens/hipt/data/fold_dir_test/fold_0", config)

    datamodule.setup(stage="fit")
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()

    datamodule.setup(stage="test")
    test_dl = datamodule.test_dataloader()

    x, y = next(iter(train_dl))

from source.data import PreextractedFeatureDataset

def test_dataset():
    config = {'target':'label'}

    dataset = PreextractedFeatureDataset(
        '/mnt/hpc/rens/hipt/data/fold_dir_test/fold_0/train.csv',
        config
    )

    x, y = dataset[0]

def test_datamodule():
    
from source.data import PreextractedFeatureDataset

def test_initialize_dataset():
    config = {'target':'label'}

    dataset = PreextractedFeatureDataset(
        '/mnt/hpc/rens/hipt/data/fold_dir_primary_vs_metastasis/fold_0/train.csv',
        config
    )

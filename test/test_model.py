from source.model import (
    NaivePoolingClassifier,
    AttentionClassifier,
    TransformerClassifier,
    Model,
)
from source.data import PreextractedFeatureDataset, DataModule


def test_naive_pooling_classifier():
    config = {
        "target": "label",
        "n_features": 192,
        "n_classes": 3,
        "pooling_function": "max",
        "final_activation": "softmax",
    }
    dataset = PreextractedFeatureDataset(
        "/mnt/hpc/rens/hipt/data/fold_dir_test/fold_0/train.csv", config
    )
    x, _ = dataset[0]

    model = NaivePoolingClassifier(config)

    y_hat, heatmap = model.forward(x, return_heatmap=True)

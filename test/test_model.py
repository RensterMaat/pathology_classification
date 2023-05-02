from pathlib import Path
import yaml
from source.model import (
    NaivePoolingClassifier,
    AttentionClassifier,
    TransformerClassifier,
    Model,
)
from source.data import PreextractedFeatureDataset, DataModule


with open("config/test.yaml", "r") as f:
    config = yaml.safe_load(f)


def test_naive_pooling_classifier():
    dataset = PreextractedFeatureDataset(
        Path(config["manifest_dir"]) / "train.csv", config
    )
    x, _ = dataset[0]

    model = NaivePoolingClassifier(config)

    y_hat, heatmap = model.forward(x, return_heatmap=True)
    print()

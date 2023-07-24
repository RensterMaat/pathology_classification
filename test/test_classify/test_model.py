import yaml
from pathlib import Path
from source.classify.classifiers import (
    AttentionClassifier,
    NaivePoolingClassifier,
    TransformerClassifier,
)
from source.classify.data import PreextractedFeatureDataset


with open("config/classify.yaml", "r") as f:
    config = yaml.safe_load(f)
config["fold"] = 0


def test_naive_pooling_classifier():
    dataset = PreextractedFeatureDataset(
        Path(config["manifest_dir"]) / f'fold_{config["fold"]}' / "train.csv", config
    )
    x, _, _ = dataset[0]

    model = NaivePoolingClassifier(config)

    y_hat, heatmap = model.forward(x.unsqueeze(0), return_heatmap_vector=True)
    print()


def test_attention_classifier():
    dataset = PreextractedFeatureDataset(
        Path(config["manifest_dir"]) / f'fold_{config["fold"]}' / "train.csv", config
    )
    x, _, _ = dataset[0]

    model = AttentionClassifier(config)

    y_hat, attention = model.forward(x.unsqueeze(0), return_heatmap_vector=True)

    print()


def test_transformer_classifier():
    dataset = PreextractedFeatureDataset(
        Path(config["manifest_dir"]) / f'fold_{config["fold"]}' / "train.csv", config
    )
    x, _, _ = dataset[0]
    model = TransformerClassifier(config)

    y_hat, attention = model.forward(x.unsqueeze(0), return_heatmap_vector=True)
    print()

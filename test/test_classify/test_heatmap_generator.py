import yaml
from pathlib import Path
from source.classify.heatmap import HeatmapGenerator
from source.classify.classify_data import PreextractedFeatureDataset

with open("config/classify.yaml", "r") as f:
    config = yaml.safe_load(f)
config["fold"] = 0


def test_heatmap():
    dataset = PreextractedFeatureDataset(
        Path(config["cross_val_splits_dir"]) / f'fold_{config["fold"]}' / "train.csv",
        config,
    )

    x, y, features_path = dataset[0]
    heatmap_vector = x[
        :, :2
    ]  # mock heatmap vector with same length as x (number of patches)

    slide_id = Path(features_path).stem

    hmg = HeatmapGenerator(config)

    figure = hmg(heatmap_vector, slide_id)

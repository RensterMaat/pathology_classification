import yaml
from pathlib import Path
from source.classify.heatmap import HeatmapGenerator
from source.classify.data import PreextractedFeatureDataset

with open("config/test.yaml", "r") as f:
    config = yaml.safe_load(f)


def test_heatmap():
    dataset = PreextractedFeatureDataset(
        Path(config["manifest_dir"]) / "train.csv", config
    )

    x, y, features_path = dataset[0]
    heatmap_vector = x[
        :, :2
    ]  # mock heatmap vector with same length as x (number of patches)

    slide_id = Path(features_path).stem

    hmg = HeatmapGenerator(config)

    figure = hmg(heatmap_vector, slide_id)

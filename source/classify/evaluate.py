import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split

from source.classify import train_and_test_loop
from source.classify.heatmap import HeatmapGenerator
from source.utils.plot_utils import plot_roc, plot_calibration_curve
from source.utils.utils import (
    load_config,
    get_cross_val_splits_dir_path,
)


def make_cross_val_splits(config):
    manifest = pd.read_csv(config["manifest_file_path"]).set_index("slide_id")

    for characteristic, groups in config["subgroups"].items():
        manifest = manifest[manifest[characteristic].isin(groups)]

    manifest = manifest[config["targets"]]
    manifest = manifest.dropna()

    save_dir = get_cross_val_splits_dir_path(config)
    save_dir.mkdir(exist_ok=True, parents=True)

    skf = StratifiedKFold(
        n_splits=config["n_folds"], shuffle=True, random_state=config["seed"]
    )

    for fold, (train_tune_index, test_index) in enumerate(
        skf.split(manifest.index, manifest[config["targets"][0]])
    ):
        fold_dir = save_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True, parents=True)

        train_tune_manifest = manifest.iloc[train_tune_index]
        train_manifest, tune_manifest = train_test_split(
            train_tune_manifest,
            test_size=0.25,
            random_state=config["seed"],
            stratify=train_tune_manifest[config["targets"][0]],
        )

        test_manifest = manifest.iloc[test_index]

        train_manifest.to_csv(fold_dir / "train.csv")
        tune_manifest.to_csv(fold_dir / "tune.csv")
        test_manifest.to_csv(fold_dir / "test.csv")


def make_plots(config):
    all_results = []
    for ix, fold_csv in enumerate(
        (Path(config["experiment_log_dir"]) / "results").glob("*.csv")
    ):
        df = pd.read_csv(fold_csv)
        df["fold"] = ix

        all_results.append(df)

    all_results = pd.concat(all_results).set_index("slide_id")

    for target in config["targets"]:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_roc(
            all_results,
            target=f"{target}_true",
            prediction=f"{target}_prediction",
            ax=ax,
        )
        fig.savefig(
            Path(config["experiment_log_dir"]) / "results" / f"{target}_roc_curve.png"
        )

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_calibration_curve(
            all_results,
            target=f"{target}_true",
            prediction=f"{target}_prediction",
            ax=ax,
        )
        fig.savefig(
            Path(config["experiment_log_dir"])
            / "results"
            / f"{target}_calibration_curve.png"
        )


def make_heatmaps(config):
    heatmap_vectors_dir = config["experiment_log_dir"] / "heatmap_vectors"
    heatmap_dir = config["experiment_log_dir"] / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True, parents=True)

    hmg = HeatmapGenerator(config)

    for heatmap_vector_file in tqdm(
        list(heatmap_vectors_dir.glob("*.pt")),
        desc="Generating heatmaps",
        unit="slide",
    ):
        fig = hmg(heatmap_vector_file)
        fig.savefig(heatmap_dir / f"{heatmap_vector_file.stem}.png")


def main(config):
    # prepare config for evaluation
    config["mode"] = "evaluate"
    config.update(config["evaluate"])
    config["experiment_log_dir"] = str(
        Path(config["output_dir"], "output", datetime.now().strftime("%Y%m%d_%H:%M:%S"))
    )

    # split data into folds
    make_cross_val_splits(config)

    # perform evaluation
    train_and_test_loop.main(config)

    # make plots
    make_plots(config)

    # generate heatmaps (optional)
    if config["generate_heatmaps"]:
        make_heatmaps(config)

    # save configuration file for reproducibility
    with open(Path(config["experiment_log_dir"]) / "config.yaml", "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/umcu.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    main(config)
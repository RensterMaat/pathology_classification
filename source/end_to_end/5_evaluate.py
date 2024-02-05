import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split

from source.utils.utils import load_config, get_cross_val_splits_dir_path
from source.classify import train_and_test
from source.utils.plot_utils import plot_roc, plot_calibration_curve


def make_cross_val_splits(config):
    manifest = pd.read_csv(config["manifest_file_path"]).set_index("slide_id")

    for characteristic, groups in config["subgroups"].items():
        manifest = manifest[manifest[characteristic].isin(groups)]

    manifest = manifest[[config["target"]]]
    manifest = manifest.dropna()

    save_dir = get_cross_val_splits_dir_path(config)
    save_dir.mkdir(exist_ok=True, parents=True)

    skf = StratifiedKFold(
        n_splits=config["n_folds"], shuffle=True, random_state=config["seed"]
    )

    for fold, (train_tune_index, test_index) in enumerate(
        skf.split(manifest.index, manifest[config["target"]])
    ):
        fold_dir = save_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True, parents=True)

        train_tune_manifest = manifest.iloc[train_tune_index]
        train_manifest, tune_manifest = train_test_split(
            train_tune_manifest,
            test_size=0.25,
            random_state=config["seed"],
            stratify=train_tune_manifest[config["target"]],
        )

        test_manifest = manifest.iloc[test_index]

        train_manifest.to_csv(fold_dir / "train.csv")
        tune_manifest.to_csv(fold_dir / "tune.csv")
        test_manifest.to_csv(fold_dir / "test.csv")


def make_plots(config):
    all_results = []
    for ix, fold_csv in enumerate(
        (config["experiment_log_dir"] / "results").glob("*.csv")
    ):
        df = pd.read_csv(fold_csv)
        df["fold"] = ix

        all_results.append(df)

    all_results = pd.concat(all_results).set_index("slide_id")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_roc(all_results, target=config["evaluate"]["target"], ax=ax)
    fig.savefig(config["experiment_log_dir"] / "results" / "roc_curve.png")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_calibration_curve(all_results, target=config["evaluate"]["target"], ax=ax)
    fig.savefig(config["experiment_log_dir"] / "results" / "calibration_curve.png")


def main(config):
    config["mode"] = "evaluate"
    config.update(config["evaluate"])

    config["experiment_log_dir"] = Path(
        config["output_dir"], "output", datetime.now().strftime("%Y%m%d_%H:%M:%S")
    )

    make_cross_val_splits(config)

    train_and_test.main(config)

    make_plots(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/end_to_end/umcu.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    main(config)

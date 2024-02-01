import wandb
import argparse
import pandas as pd
from source.utils.utils import load_config, get_cross_val_splits_dir_path
from source.classify import train_and_test
from sklearn.model_selection import train_test_split


def make_train_tune_split(config):
    config.update(config["sweep"]["general_settings"])
    config["mode"] = "sweep"

    manifest = pd.read_csv(config["manifest_file_path"]).set_index("slide_id")
    manifest = manifest[[config["target"]]]
    manifest = manifest.dropna()

    train_manifest, tune_manifest = train_test_split(
        manifest,
        test_size=0.25,
        random_state=config["seed"],
        stratify=manifest[config["target"]],
    )

    save_dir = get_cross_val_splits_dir_path(config) / "fold_0"
    save_dir.mkdir(exist_ok=True, parents=True)

    train_manifest.to_csv(save_dir / "train.csv")
    tune_manifest.to_csv(save_dir / "tune.csv")


def make_wandb_sweep_config(config):
    sweep_config = {
        "method": config["method"],
        "metric": {"name": "fold_0/val_auc", "goal": "maximize"},
    }

    parameters = {"output_dir": {"values": [config["output_dir"]]}}

    for parameter, value in config["sweep"]["general_settings"].items():
        parameters[parameter] = {"values": [value]}

    for parameter, value in config["sweep"]["fixed_hyperparameters"].items():
        parameters[parameter] = {"values": [value]}

    for parameter, values in config["sweep"]["variable_hyperparameters"].items():
        parameters[parameter] = {"values": values}

    sweep_config["parameters"] = parameters

    return sweep_config


def evaluate_configuration():
    wandb.init()

    wandb.config["mode"] = "sweep"
    wandb.config["n_folds"] = 1

    train_and_test.main(wandb.config)


def main(config):
    make_train_tune_split(config)
    wandb_sweep_config = make_wandb_sweep_config(config)

    sweep_id = wandb.sweep(
        wandb_sweep_config, project=config["sweep"]["general_settings"]["project_name"]
    )

    wandb.agent(sweep_id, evaluate_configuration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conduct a hyperparameter sweep based on the supplied configuration file."
    )
    parser.add_argument("--config", default="config/end_to_end/umcu.yaml")

    args = parser.parse_args()
    config = load_config(args.config)

    main(config)

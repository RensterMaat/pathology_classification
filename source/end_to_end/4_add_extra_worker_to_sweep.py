import wandb
import argparse
import importlib


sweep = importlib.import_module("source.end_to_end.4_hyperparameter_sweep")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add an extra worker to the sweep")
    parser.add_argument("--sweep_id", required=True)

    args = parser.parse_args()

    wandb.agent(args.sweep_id, sweep.evaluate_configuration, project='wsi_classification')

import argparse
import numpy as np
from pathlib import Path
from source.classify.classifier_framework import ClassifierFramework
from source.classify.data import ClassificationDataModule
from source.utils.utils import load_config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime


def main(config):
    seed_everything(config["seed"])

    if not "mode" in config:
        config["mode"] = "evaluate"

    if not "experiment_log_dir" in config:
        config["experiment_log_dir"] = Path(
            config["output_dir"], "output", datetime.now().strftime("%Y%m%d_%H:%M:%S")
        )

    logger = WandbLogger(
        save_dir=config["experiment_log_dir"], project="wsi_classification_dev"
    )
    logger.experiment.config.update(config)

    for fold in range(config["n_folds"]):
        config["fold"] = fold

        trainer = Trainer(
            accelerator="gpu",
            max_epochs=config["max_epochs"],
            accumulate_grad_batches=config["accumulate_grad_batches"],
            logger=logger,
            callbacks=[
                EarlyStopping(
                    monitor=f"fold_{fold}/val_auc",
                    patience=config["patience"],
                    mode="max",
                ),
                ModelCheckpoint(
                    Path(config["experiment_log_dir"]) / "checkpoints",
                    monitor=f"fold_{fold}/val_auc",
                    mode="max",
                    filename=f"fold={fold}_" + "{epoch:02d}",
                ),
            ],
        )

        datamodule = ClassificationDataModule(config)
        model = ClassifierFramework(config)

        trainer.fit(model, datamodule)

        if (datamodule.cross_val_splits_directory / "test.csv").exists():
            trainer.test(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross validate the classifier on preextracted features."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/rens/repos/pathology_classification/config/classify/umcu.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    main(config)

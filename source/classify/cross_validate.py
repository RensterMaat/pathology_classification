import torch
import argparse
from pathlib import Path
from source.classify.classifier_framework import ClassifierFramework
from source.classify.data import ClassificationDataModule
from source.utils.utils import load_config
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def main(config):
    logger = WandbLogger(
        save_dir=config["output_dir"], project="wsi_classification_dev"
    )

    config["experiment_log_dir"] = logger.experiment._settings.sync_dir
    log_dir = Path(config["experiment_log_dir"])

    logger.experiment.config.update(config)

    n_folds = len(list(Path(config["dataset_dir"], 'cross_val_splits').iterdir()))
    for fold in range(n_folds):
        config["fold"] = fold

        trainer = Trainer(
            accelerator="gpu",
            max_epochs=config["max_epochs"],
            logger=logger,
            callbacks=[
                EarlyStopping(
                    monitor=f"fold_{fold}/val_auc",
                    patience=config["patience"],
                    mode="max",
                ),
                ModelCheckpoint(
                    log_dir / "checkpoints", monitor=f"fold_{fold}/val_auc", mode="max"
                ),
            ],
        )

        datamodule = ClassificationDataModule(config)
        model = ClassifierFramework(config)

        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross validate the classifier on preextracted features."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/hpc/umcu.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # torch.multiprocessing.set_start_method("spawn")

    main(config)

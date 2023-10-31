import yaml
import argparse
from pathlib import Path
from source.classify.classifier_framework import ClassifierFramework
from source.classify.data import ClassificationDataModule
from source.utils.utils import load_config
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime


def main(config):
    config["experiment_log_dir"] = Path(
        config["dataset_dir"], "output", datetime.now().strftime("%Y%m%d_%H:%M:%S")
    )

    logger = WandbLogger(
        save_dir=config["experiment_log_dir"], project="wsi_classification_dev"
    )
    logger.experiment.config.update(config)

    n_folds = len(list(Path(config["dataset_dir"], "cross_val_splits").iterdir()))
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
                    config["experiment_log_dir"] / "checkpoints",
                    monitor=f"fold_{fold}/val_auc",
                    mode="max",
                    filename=f"fold={fold}_" + "{epoch:02d}",
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
        default="/home/rens/repos/pathology_classification/config/classify/umcu.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    main(config)

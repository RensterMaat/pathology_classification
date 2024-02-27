from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime

from source.classify.classifier_framework import ClassifierFramework
from source.classify.classify_data import ClassificationDataModule


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
                    monitor=f"fold_{fold}/{config['targets'][0]}_val_auc",
                    patience=config["patience"],
                    mode="max",
                ),
                ModelCheckpoint(
                    Path(config["experiment_log_dir"]) / "checkpoints",
                    monitor=f"fold_{fold}/{config['targets'][0]}_val_auc",
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

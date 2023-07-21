import yaml
from pathlib import Path
from source.classify.classifier_framework import ClassifierFramework
from source.classify.data import ClassificationDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

with open("config/classify.yaml", "r") as f:
    config = yaml.safe_load(f)

logger = WandbLogger(save_dir=config["output_dir"], project="wsi_classification_dev")

config["experiment_log_dir"] = logger.experiment._settings.sync_dir
log_dir = Path(config["experiment_log_dir"])

logger.experiment.config.update(config)

n_folds = len(list(Path(config["manifest_dir"]).iterdir()))
for fold in range(n_folds):
    config["fold"] = fold

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=config["max_epochs"],
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor=f"fold_{fold}/val_auc", patience=config["patience"], mode="max"
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

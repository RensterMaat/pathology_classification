import yaml
from pytorch_lightning import Trainer
from source.classify.classify_data import ClassificationDataModule
from source.classify.classifier_framework import ClassifierFramework


with open("config/classify.yaml", "r") as f:
    config = yaml.safe_load(f)
config["fold"] = 0
config["generate_heatmaps"] = False

datamodule = ClassificationDataModule(config)
model = ClassifierFramework(config)


def test_fast_dev_run():
    trainer = Trainer(
        accelerator="cpu",
        fast_dev_run=True,
    )
    trainer.fit(model, datamodule)


def test_overfit():
    trainer = Trainer(
        max_epochs=10,
        accelerator="cpu",
        overfit_batches=2,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule)


def test_testing_loop():
    trainer = Trainer(accelerator="cpu")
    config["experiment_log_dir"] = trainer.logger.log_dir

    datamodule = ClassificationDataModule(config)
    model = ClassifierFramework(config)

    trainer.test(model, datamodule)

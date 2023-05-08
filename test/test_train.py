import yaml
from pytorch_lightning import Trainer
from source.data import DataModule
from source.model import Model


with open("config/test.yaml", "r") as f:
    config = yaml.safe_load(f)


datamodule = DataModule(config)
model = Model(config)

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
        num_sanity_val_steps=0
    )
    trainer.fit(model, datamodule)


def test_testing_loop():
    trainer = Trainer(accelerator='cpu')
    config['experiment_log_dir'] = trainer.logger.log_dir

    datamodule = DataModule(config)
    model = Model(config)

    trainer.test(model, datamodule)

import yaml
import torch
from model import Model
from data import DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


with open("config/test.yaml", "r") as f:
    config = yaml.safe_load(f)

logger = WandbLogger(save_dir=config["output_dir"], project="wsi_classification_dev")

trainer = Trainer(logger=logger)
config["experiment_log_dir"] = logger.experiment._settings.sync_dir

model = Model(config)
ckpt = torch.load(
    "/home/rens/repos/pathology_classification/output/wandb/run-20230706_092001-lhn8npk3/checkpoints/epoch=8-step=4896.ckpt"
)
model.load_state_dict(ckpt["state_dict"])
model.eval()

datamodule = DataModule(config)
datamodule.setup(stage="test")

trainer.test(model, datamodule)

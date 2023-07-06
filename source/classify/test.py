import yaml
import torch
from model import Model
from data import ClassificationDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


with open("config/classify.yaml", "r") as f:
    config = yaml.safe_load(f)

logger = WandbLogger(save_dir=config["output_dir"], project="wsi_classification_dev")

trainer = Trainer(logger=logger)
config["experiment_log_dir"] = logger.experiment._settings.sync_dir
config["fold"] = 0

model = Model(config)
ckpt = torch.load(
    "/home/rens/repos/pathology_classification/output/wandb/run-20230706_135215-5x9qil9d/checkpoints/epoch=0-step=544.ckpt"
)
model.load_state_dict(ckpt["state_dict"])
model.eval()

datamodule = ClassificationDataModule(config)
datamodule.setup(stage="test")

trainer.test(model, datamodule)

import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryAUROC
from source.classifiers import (
    AttentionClassifier,
    NaivePoolingClassifier,
    TransformerClassifier,
)
from source.heatmap import HeatmapGenerator
from pathlib import Path


class Model(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

        if config["model"] == "NaivePoolingClassifier":
            self.model = NaivePoolingClassifier(config)
        elif config["model"] == "AttentionClassifier":
            self.model = AttentionClassifier(config)
        elif config["model"] == "TransformerClassifier":
            self.model = TransformerClassifier(config)
        else:
            raise NotImplementedError

        self.criterion = nn.BCELoss()

        self.train_auc = BinaryAUROC(pos_label=1)
        self.val_auc = BinaryAUROC(pos_label=1)
        self.test_auc = BinaryAUROC(pos_label=1)

        self.test_outputs = []

        if config["generate_heatmaps"]:
            self.heatmap_generator = HeatmapGenerator(config)

    def forward(
        self, x: torch.Tensor, return_heatmap_vector: bool = False
    ) -> torch.Tensor:
        return self.model(x, return_heatmap_vector)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y, _ = batch
        y_hat = self.model(x)

        loss = self.criterion(y_hat, y)
        self.train_auc.update(y_hat.squeeze(), y.squeeze().int())

        self.log_dict(
            {
                f"fold_{self.config['fold']}/train_loss": loss,
                f"fold_{self.config['fold']}/train_auc": self.train_auc.compute(),
            }
        )

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y, _ = batch
        y_hat = self.model(x)

        loss = self.criterion(y_hat, y)
        self.val_auc.update(y_hat.squeeze(), y.squeeze().int())

        self.log_dict(
            {
                f"fold_{self.config['fold']}/val_loss": loss,
                f"fold_{self.config['fold']}/val_auc": self.val_auc.compute(),
            }
        )

        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y, features_path = batch
        slide_id = Path(features_path[0]).stem

        if self.config["generate_heatmaps"]:
            y_hat, heatmap_vector = self.model.forward(
                x, return_heatmap_vector=self.config["generate_heatmaps"]
            )
            heatmap = self.heatmap_generator(heatmap_vector, slide_id)
            heatmap.axes[0].set_title(
                f"Label: {y.argmax().cpu()} - Prediction: {y_hat[0,1].cpu():.3f}"
            )

            heatmap_dir = Path(self.config["experiment_log_dir"]) / "heatmaps"
            heatmap_dir.mkdir(exist_ok=True)

            save_path = heatmap_dir / (slide_id + ".jpg")
            heatmap.savefig(save_path)
            plt.close("all")
        else:
            y_hat = self.model.forward(
                x, return_heatmap_vector=self.config["generate_heatmaps"]
            )

        self.test_auc.update(y_hat.squeeze(), y.squeeze().int())

        self.test_outputs.append(
            [slide_id, int(y[0, 1].detach().cpu()), float(y_hat[0, 1].detach().cpu())]
        )
        self.log_dict(
            {f"test/fold_{self.config['fold']}_auc": self.test_auc.compute()},
            on_step=False,
            on_epoch=True,
        )

    def on_test_epoch_end(self):
        results = pd.DataFrame(
            self.test_outputs, columns=["slide_id", self.config["target"], "prediction"]
        )
        results_dir = Path(self.config["experiment_log_dir"]) / "results"
        results_dir.mkdir(exist_ok=True)
        results.to_csv(
            results_dir / f"fold_{self.config['fold']}_test_output.csv", index=False
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        return optimizer

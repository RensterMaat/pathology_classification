import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryAUROC
from source.classify.classifiers import (
    AttentionClassifier,
    NaivePoolingClassifier,
    TransformerClassifier,
)
from source.classify.heatmap import HeatmapGenerator
from pathlib import Path


class ClassifierFramework(pl.LightningModule):
    """
    Classifier framework for training and testing.

    Implements the pytorch-lightning LightningModule interface.

    Methods:
        forward: Forward pass through the classifier.
        training_step: Training step for pytorch-lightning.
        validation_step: Validation step for pytorch-lightning.
        test_step: Test step for pytorch-lightning.
        on_test_epoch_end: Called at the end of the test epoch.
        configure_optimizers: Configures the optimizer.

    Attributes:
        config: Dictionary containing the configuration.
        classifier: The classifier, one of:
            - NaivePoolingClassifier
            - AttentionClassifier
            - TransformerClassifier
        criterion: The loss function.
        train_auc: The training AUROC metric.
        val_auc: The validation AUROC metric.
        test_auc: The test AUROC metric.
        test_outputs: List containing the test outputs.
        heatmap_generator: The heatmap generator.
    """

    def __init__(self, config: dict) -> None:
        """
        Initializes the ClassifierFramework.

        Args:
            config: Dictionary containing the configuration. See also load_config in source/utils/utils.py.
        """
        super().__init__()

        self.config = config

        if config["classifier"] == "NaivePoolingClassifier":
            self.classifier = NaivePoolingClassifier(config)
        elif config["classifier"] == "AttentionClassifier":
            self.classifier = AttentionClassifier(config)
        elif config["classifier"] == "TransformerClassifier":
            self.classifier = TransformerClassifier(config)
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
        """
        Forward pass through the classifier.

        Args:
            x: The input tensor of shape (batch_size, num_patches, num_features).
            return_heatmap_vector: Whether to return the heatmap vector.

        Returns:
            The output tensor of shape (batch_size, num_classes).
        """
        return self.classifier(x, return_heatmap_vector)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step for pytorch-lightning.

        Args:
            batch: The batch of data.
            batch_idx: The batch index.

        Returns:
            The loss.
        """
        x, y, _ = batch
        y_hat = self.classifier(x)

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
        """
        Validation step for pytorch-lightning.

        Args:
            batch: The batch of data.
            batch_idx: The batch index.

        Returns:
            The loss.
        """
        x, y, _ = batch
        y_hat = self.classifier(x)

        loss = self.criterion(y_hat, y)
        self.val_auc.update(y_hat.squeeze(), y.squeeze().int())

        self.log_dict(
            {
                f"fold_{self.config['fold']}/val_loss": loss,
                f"fold_{self.config['fold']}/val_auc": self.val_auc.compute(),
            }
        )

        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Testing step for pytorch-lightning.

        Optionally generates heatmaps. Set this option in the config file.

        Args:
            batch: The batch of data.
            batch_idx: The batch index.

        """
        x, y, features_path = batch
        slide_id = Path(features_path[0]).stem

        if self.config["generate_heatmaps"]:
            y_hat, heatmap_vector = self.classifier.forward(
                x, return_heatmap_vector=True
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
            y_hat = self.classifier.forward(x, return_heatmap_vector=False)

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
        """
        Saves the test outputs per slide of the current fold to a csv file.
        """
        results = pd.DataFrame(
            self.test_outputs, columns=["slide_id", self.config["target"], "prediction"]
        )
        results_dir = Path(self.config["experiment_log_dir"]) / "results"
        results_dir.mkdir(exist_ok=True)
        results.to_csv(
            results_dir / f"fold_{self.config['fold']}_test_output.csv", index=False
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer.

        Returns:
            The optimizer.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        return optimizer

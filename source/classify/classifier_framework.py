import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from source.classify.classifiers import (
    AttentionClassifier,
    NaivePoolingClassifier,
    TransformerClassifier,
)
from pathlib import Path
from collections import defaultdict


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

        self.metrics = {
            target: {
                "train_auc": BinaryAUROC(),
                "val_auc": BinaryAUROC(),
                "test_auc": BinaryAUROC(),
            }
            for target in config["targets"]
        }

        self.test_outputs = defaultdict(list)

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

        for ix, target in enumerate(self.config["targets"]):
            self.metrics[target]["train_auc"].update(y_hat[0, ix], y[0, ix].int())

        self.log(f"fold_{self.config['fold']}/train_loss", loss)

        return loss

    def on_train_epoch_end(self) -> None:
        for target in self.config["targets"]:
            self.log(
                f"fold_{self.config['fold']}/{target}_train_auc",
                self.metrics[target]["train_auc"].compute(),
            )
            self.metrics[target]["train_auc"].reset()

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

        for ix, target in enumerate(self.config["targets"]):
            self.metrics[target]["val_auc"].update(y_hat[0, ix], y[0, ix].int())

        self.log(f"fold_{self.config['fold']}/val_loss", loss)

        return loss

    def on_validation_epoch_end(self) -> None:
        for target in self.config["targets"]:
            self.log(
                f"fold_{self.config['fold']}/{target}_val_auc",
                self.metrics[target]["val_auc"].compute(),
            )
            self.metrics[target]["val_auc"].reset()

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

            heatmap_vector_dir = (
                Path(self.config["experiment_log_dir"]) / "heatmap_vectors"
            )
            heatmap_vector_dir.mkdir(exist_ok=True, parents=True)

            save_path = heatmap_vector_dir / (slide_id + ".pt")
            torch.save(heatmap_vector, save_path)
        else:
            y_hat = self.classifier.forward(x, return_heatmap_vector=False)

        for ix, target in enumerate(self.config["targets"]):
            self.metrics[target]["test_auc"].update(y_hat[0, ix], y[0, ix].int())

        self.test_outputs["slide_id"].append(slide_id)
        for ix, target in enumerate(self.config["targets"]):
            self.test_outputs[f"{target}_true"].append(int(y[0, ix].detach().cpu()))
            self.test_outputs[f"{target}_prediction"].append(
                float(y_hat[0, ix].detach().cpu())
            )

    def on_test_epoch_end(self):
        """
        Saves the test outputs per slide of the current fold to a csv file.
        """
        results = pd.DataFrame(self.test_outputs)
        results_dir = Path(self.config["experiment_log_dir"]) / "results"
        results_dir.mkdir(exist_ok=True, parents=True)
        results.to_csv(
            results_dir / f"fold_{self.config['fold']}_test_output.csv", index=False
        )

        for target in self.config["targets"]:
            self.log(
                f"test_{target}/fold_{self.config['fold']}_auc",
                self.metrics[target]["test_auc"].compute(),
            )
            self.metrics[target]["test_auc"].reset()

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

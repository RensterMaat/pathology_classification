import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC


class Model(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

        if config["model"] == "NaivePoolingClassifier":
            self.model = NaivePoolingClassifier(config)
        else:
            raise NotImplementedError

        self.criterion = nn.BCELoss()

        self.train_auc = BinaryAUROC(pos_label=1)
        self.val_auc = BinaryAUROC(pos_label=1)

    def forward(self, x: torch.Tensor, return_heatmap: bool = False) -> torch.Tensor:
        return self.model(x, return_heatmap)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y, _ = batch
        y_hat = self.model(x)

        loss = self.criterion(y_hat, y)
        self.train_auc.update(y_hat.squeeze(), y.squeeze().int())

        self.log_dict({"train/loss": loss, "train/auc": self.train_auc.compute()})

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y, _ = batch
        y_hat = self.model(x)

        loss = self.criterion(y_hat, y)
        self.val_auc.update(y_hat.squeeze(), y.squeeze().int())

        self.log_dict({"val/loss": loss, "val/auc": self.val_auc.compute()})

        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        return optimizer


class Classifier(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        if config["final_activation"] == "softmax":
            self.final_activation = nn.Softmax(dim=1)
        elif config["final_activation"] == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            raise NotImplementedError


class NaivePoolingClassifier(Classifier):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self.classifier = nn.Linear(config["n_features"], config["n_classes"])

        if config["pooling_function"] == "max":
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif config["pooling_function"] == "mean":
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor, return_heatmap: bool = False) -> torch.Tensor:
        per_patch_logits = self.classifier(x[0])
        per_slide_logits = torch.t(self.pooling(torch.t(per_patch_logits)))
        slide_prediction = self.final_activation(per_slide_logits)

        if return_heatmap:
            return slide_prediction, self.final_activation(per_patch_logits)

        return slide_prediction


class AttentionClassifier(Classifier):
    def __init__(self, config: dict) -> None:
        super().__init__()

    def forward(self, x):
        pass

    def get_heatmap(self, x):
        pass


class TransformerClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def get_heatmap(self, x):
        pass

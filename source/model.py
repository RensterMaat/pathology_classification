import torch
import torch.nn as nn
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config["model"] == "NaivePoolingClassifier":
            self.model = NaivePoolingClassifier(config)
        else:
            raise NotImplementedError

        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.criterion(y_hat, y[0])

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.criterion(y_hat, y[0])

        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
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
            self.final_activation = nn.Softmax()
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
        per_patch_prediction = self.classifier(x[0])
        logits = torch.t(self.pooling(torch.t(per_patch_prediction)))
        slide_prediction = self.final_activation(logits)

        if return_heatmap:
            return slide_prediction, per_patch_prediction

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

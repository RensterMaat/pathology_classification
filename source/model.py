import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
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
        x, y, features_path = batch
        slide_id = Path(features_path[0]).stem

        if self.config["generate_heatmaps"]:
            y_hat, heatmap_vector = self.model.forward(
                x, return_heatmap_vector=self.config["generate_heatmaps"]
            )
            heatmap = self.heatmap_generator(heatmap_vector, slide_id)
            save_path = Path(self.config["experiment_log_dir"]) / (slide_id + ".jpg")
            heatmap.savefig(save_path)
            heatmap.close()
        else:
            y_hat = self.model.forward(
                x, return_heatmap_vector=self.config["generate_heatmaps"]
            )

        self.test_auc.update(y_hat.squeeze(), y.squeeze().int())

        self.test_outputs.append(
            [slide_id, int(y[0, 1].detach().cpu()), float(y_hat[0, 1].detach().cpu())]
        )
        self.log_dict(
            {"test/auc": self.test_auc.compute()}, on_step=False, on_epoch=True
        )

    def on_test_epoch_end(self):
        results = pd.DataFrame(
            self.test_outputs, columns=["slide_id", self.config["target"], "prediction"]
        )
        results.to_csv(
            Path(self.config["experiment_log_dir"]) / "test_output.csv", index=False
        )

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

    def forward(
        self, x: torch.Tensor, return_heatmap_vector: bool = False
    ) -> torch.Tensor:
        per_patch_logits = self.classifier(x[0])
        per_slide_logits = torch.t(self.pooling(torch.t(per_patch_logits)))
        slide_prediction = self.final_activation(per_slide_logits)

        if return_heatmap_vector:
            return slide_prediction, self.final_activation(per_patch_logits)

        return slide_prediction


class AttentionClassifier(Classifier):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self.attention_layer = GatedAttentionLayer(
            config["n_features"], config["attention_dim"], config["n_classes"]
        )
        self.classifier = nn.Linear(config["n_features"], 1)

    def forward(self, x, return_heatmap_vector=False):
        attention = self.attention_layer(x)[0]

        slide_representation = torch.matmul(torch.transpose(attention, 0, 1), x)
        per_slide_logits = self.classifier(slide_representation)
        slide_prediction = torch.transpose(
            self.final_activation(per_slide_logits)[0], 0, 1
        )

        if return_heatmap_vector:
            return slide_prediction, attention

        return slide_prediction


class GatedAttentionLayer(nn.Module):
    def __init__(self, n_features, attention_dim, n_classes):
        super().__init__()

        self.attention_a = nn.Sequential(
            *[nn.Linear(n_features, attention_dim), nn.Tanh()]
        )
        self.attention_b = nn.Sequential(
            *[nn.Linear(n_features, attention_dim), nn.Sigmoid()]
        )
        self.attention_c = nn.Linear(*[attention_dim, n_classes])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        attention = self.softmax(A)
        return attention


class TransformerClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def get_heatmap(self, x):
        pass

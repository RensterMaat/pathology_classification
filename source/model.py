import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
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

        self.phi = nn.Sequential(
            nn.Linear(config["n_features"], config["n_features"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
        )

        self.attention_pooling = GlobalGatedAttentionPooling(
            input_dim=config["n_features"],
            hidden_dim=config["attention_dim"],
            output_dim=1,
            dropout=config["dropout"],
        )
        self.classifier = nn.Linear(config["n_features"], config["n_classes"])

    def forward(self, x, return_heatmap_vector=False):
        x = x[0]

        x = self.phi(x)

        attention = self.attention_pooling(x)
        slide_representation = torch.matmul(attention.transpose(0, 1), x)

        slide_logits = self.classifier(slide_representation)
        slide_prediction = self.final_activation(slide_logits)

        if return_heatmap_vector:
            return slide_prediction, attention

        return slide_prediction


class GlobalGatedAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()

        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout)
        )
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout)
        )

        self.attention_c = nn.Linear(*[hidden_dim, output_dim])
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)

        A = self.attention_c(A)
        attention = self.softmax(A)

        return attention


class TransformerClassifier(Classifier):
    """
    Slide level classifier using patch level features based on HIPT's last layer with the following steps:
    1. Linear layer: n_patches x n_features -> n_patches x n_features
    2. Transformer: n_patches x n_features -> n_patches x n_features
    3. Attention pooling: n_patches x n_features -> n_features
    4. Linear layer: n_features -> n_features
    5. Classifier: n_features -> n_classes
    """

    def __init__(self, config):
        super().__init__(config)

        # make hidden dim a parameter
        self.phi = nn.Sequential(
            nn.Linear(config["n_features"], config["n_features"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config["n_features"],
                nhead=config["n_heads"],
                dim_feedforward=config["n_features"],
                dropout=config["dropout"],
                activation=config["activation_function"],
            ),
            num_layers=config["n_layers"],
        )

        self.attention_pooling = GlobalGatedAttentionPooling(
            input_dim=config["n_features"],
            hidden_dim=config["attention_dim"],
            output_dim=1,
            dropout=config["dropout"],
        )

        self.rho = nn.Sequential(
            nn.Linear(config["n_features"], config["n_features"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
        )

        self.classifier = nn.Linear(config["n_features"], config["n_classes"])

    def forward(self, x, return_heatmap_vector=False):
        x = x[0]

        x = self.phi(x)
        x = self.transformer(x)

        attention = self.attention_pooling(x)
        slide_representation = torch.matmul(attention.transpose(0, 1), x)

        slide_representation = self.rho(slide_representation)
        slide_logits = self.classifier(slide_representation)
        slide_prediction = self.final_activation(slide_logits)

        if return_heatmap_vector:
            return slide_prediction, attention

        return slide_prediction

    def get_heatmap(self, x):
        pass

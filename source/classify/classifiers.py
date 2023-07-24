import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from source.classify.model_components import GlobalGatedAttentionPooling


class Classifier(nn.Module, ABC):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

        if config["final_activation"] == "softmax":
            self.final_activation = nn.Softmax(dim=1)
        elif config["final_activation"] == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            raise NotImplementedError

    @abstractmethod
    def forward(
        self, x: torch.Tensor, return_heatmap_vector: bool = False
    ) -> torch.Tensor:
        pass


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
            output_dim=config["n_classes"],
            dropout=config["dropout"],
        )

        self.classifiers = nn.ModuleList(
            [nn.Linear(config["n_features"], 1) for _ in range(config["n_classes"])]
        )

        self.attention_softmax = nn.Softmax(dim=0)

    def forward(self, x, return_heatmap_vector=False):
        x = x[0]

        x = self.phi(x)

        attention_logits = self.attention_pooling(x)
        attention = self.attention_softmax(attention_logits)
        slide_representation = torch.matmul(attention.transpose(0, 1), x)

        slide_logits = []
        for class_ix in range(self.config["n_classes"]):
            class_logit = self.classifiers[class_ix](slide_representation[class_ix])[0]
            slide_logits.append(class_logit)
        slide_logits = torch.stack(slide_logits).unsqueeze(dim=0)

        slide_prediction = self.final_activation(slide_logits)

        if return_heatmap_vector:
            return slide_prediction, attention_logits

        return slide_prediction


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

        self.attention_softmax = nn.Softmax(dim=0)

    def forward(self, x, return_heatmap_vector=False):
        x = x[0]

        x = self.phi(x)
        x = self.transformer(x)

        attention_logits = self.attention_pooling(x)
        attention = self.attention_softmax(attention_logits)
        slide_representation = torch.matmul(attention.transpose(0, 1), x)

        slide_representation = self.rho(slide_representation)
        slide_logits = self.classifier(slide_representation)
        slide_prediction = self.final_activation(slide_logits)

        if return_heatmap_vector:
            return slide_prediction, attention_logits

        return slide_prediction

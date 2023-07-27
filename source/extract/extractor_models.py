import torch
import torch.nn as nn
import torchvision
import numpy as np
from abc import ABC, abstractmethod


class Extractor(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def transform(self, x: np.array) -> torch.Tensor:
        pass


class RandomExtractor(Extractor):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(self, x):
        return torch.tensor(np.random.normal(size=(x.shape[0], 32)))

    def transform(self, x):
        return x


class ResNet50ImagenetExtractor(Extractor):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        resnet50 = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )

        # only use the first three residual blocks of resnet50
        self.resnet50 = nn.Sequential(
            *list(resnet50)[:-3],
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        out = self.resnet50(x)
        out = out.view(out.size(0), -1)
        return out

    def transform(self, x):
        # normalize using mean and std of imagenet dataset
        return torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(x)

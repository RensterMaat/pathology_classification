import torch
import torch.nn as nn
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
        return np.random.normal(size=(x.shape[0], 32))
    
    def transform(self, x):
        return x


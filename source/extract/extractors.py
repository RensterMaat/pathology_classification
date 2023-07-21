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


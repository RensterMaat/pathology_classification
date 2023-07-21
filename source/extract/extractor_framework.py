from typing import Any
import torch
import pytorch_lightning as pl


class ExtractorFramework(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

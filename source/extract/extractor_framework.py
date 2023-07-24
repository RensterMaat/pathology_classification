import torch
import pytorch_lightning as pl
from source.extract.extractors import RandomExtractor


class ExtractorFramework(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config
        self.extractor = RandomExtractor(config)

        self.all_features = []

    def forward(self, x):
        x_transformed = self.extractor.transform(x)
        batch_features = self.extractor.forward(x_transformed)

        return batch_features

    def test_step(self, batch, batch_idx):
        batch_features = self.forward(batch)
        self.all_features.append(batch_features)

    def on_test_epoch_end(self):
        print(self.all_features)

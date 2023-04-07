import torch.nn as nn
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self, config):
        pass

    def forward(self, x):
        return self.model(x)
    
    def setup(self, stage='fit'):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers():
        pass

    def get_heatmap(self, x):
        return self.model.get_heatmap(x)


class NaivePoolingClassifier(nn.Module):
    def __init__(self, config):
        pass

    def forward(self, x):
        pass

    def get_heatmap(self, x):
        pass


class AttentionClassifier(nn.Module):
    def __init__(self, config):
        pass

    def forward(self, x):
        pass

    def get_heatmap(self, x):
        pass


class TransformerClassifier(nn.Module):
    def __init__(self, config):
        pass

    def forward(self, x):
        pass

    def get_heatmap(self, x):
        pass
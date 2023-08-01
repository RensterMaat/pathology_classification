import torch
import torch.nn as nn
import torchvision
import numpy as np
from abc import ABC, abstractmethod
from torchvision.models.vision_transformer import VisionTransformer
from einops import rearrange

from source.utils.model_components import BaseHIPT, Lambda


class Extractor(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    def __post_init__(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

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
            *list(resnet50.children())[:-3],
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


class PatchLevelHIPTFeatureExtractor(Extractor, BaseHIPT):
    def __init__(self, config):
        super().__init__()

        self.model = VisionTransformer(
            image_size=256,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
        )
        self.model.heads = nn.Identity()

        weights = self.load_and_convert_hipt_checkpoint(
            config["patch_level_hipt_checkpoint_path"]
        )
        self.model.load_state_dict(weights)

    def forward(self, x):
        return self.model(x)

    def transform(self, x):
        return torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )(x)


class RegionLevelHIPTFeatureExtractor(Extractor, BaseHIPT):
    def __init__(self, config) -> None:
        super().__init__()

        self.patch_level_extractor = PatchLevelHIPTFeatureExtractor(config)

        self.region_level_extractor = VisionTransformer(
            image_size=16,
            patch_size=1,
            num_layers=6,
            num_heads=6,
            hidden_dim=192,
            mlp_dim=768,
        )

        phi = nn.Sequential(
            Lambda(lambda x: x.flatten(2, 3).transpose(1, 2)),
            nn.Linear(384, 192),
            nn.GELU(),
            Lambda(lambda x: x.transpose(1, 2)),
        )

        self.region_level_extractor.conv_proj = phi
        self.region_level_extractor.heads = nn.Identity()

        weights = self.load_and_convert_hipt_checkpoint(
            config["region_level_hipt_checkpoint_path"]
        )
        self.region_level_extractor.load_state_dict(weights)

    def forward(self, x):
        x = x.unfold(2, 256, 256).unfold(3, 256, 256)
        x = rearrange(x, "b c h w p q -> (b h w) c p q")

        patch_features = self.patch_level_extractor(x)
        patch_features = patch_features.unsqueeze(0)
        patch_features = patch_features.unfold(1, 16, 16).transpose(1, 2)

        region_features = self.region_level_extractor(patch_features)

        return region_features

    def transform(self, x):
        return torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )(x)

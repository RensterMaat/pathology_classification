import torch
import torch.nn as nn
import torchvision
import numpy as np
from abc import ABC, abstractmethod
from torchvision.models.vision_transformer import VisionTransformer
from einops import rearrange

from source.utils.model_components import BaseHIPT, Lambda


class Extractor(nn.Module, ABC):
    """
    Abstract base class for extractors.

    Should be subclassed when implementing a new extractor model.
    """

    def __init__(self) -> None:
        super().__init__()

    def __post_init__(self):
        """
        Disable gradient computation and set the model to evaluation mode, since
        extractor models are only used for inference/feature extraction.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        pass


class RandomExtractor(Extractor):
    """
    Extractor that returns random features.

    Used for testing purposes.

    Attributes:
        config (dict): Configuration dictionary. See load_config() in
            source/utils/utils.py for more information.

    Methods:
        forward: Forward pass through the extractor.
        transform: Transform the input to the format expected by the extractor.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(np.random.normal(size=(x.shape[0], 32)))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResNet50ImagenetExtractor(Extractor):
    """
    Extractor that returns the output of the first three residual blocks of a ResNet50 model
    pretrained on ImageNet.

    This is the same model as used for feature extraction in the CLAM paper "Data Efficient and
    Weakly Supervised Computational Pathology on Whole Slide Images" by Lu et al. (2020).

    Attributes:
        config (dict): Configuration dictionary. See load_config() in
            source/utils/utils.py for more information.

    Methods:
        forward: Forward pass through the extractor.
        transform: Transform the input to the format expected by the extractor.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the ResNet50ImagenetExtractor.

        Args:
            config (dict): Configuration dictionary. See load_config() in
                source/utils/utils.py for more information.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the extractor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 256, 256).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1024).
        """
        out = self.resnet50(x)
        out = out.view(out.size(0), -1)
        return out

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # normalize using mean and std of imagenet dataset
        return torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(x)


class PatchLevelHIPTFeatureExtractor(Extractor, BaseHIPT):
    """
    Extractor that returns the output of the patch-level HIPT model.

    This is the first part of the HIPT model, which extracts features from
    the input image at the patch level (256 x 256 at magnification level 1).
    Further details can be found in the HIPT paper "Scaling Vision Transformers
    to Gigapixel Images via Hierarchical Self-Supervised Learning" by Chen et al.
    (2022).

    Uses the pretrained weights from the official implementation of the HIPT model,
    which can be found at https://github.com/mahmoodlab/HIPT/tree/master/HIPT_4K/Checkpoints.

    Attributes:
        model: The patch-level HIPT model.

    Methods:
        forward: Forward pass through the extractor.
        transform: Transform the input to the format expected by the extractor.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the patch level HIPT feature extractor.

        Args:
            config (dict): Configuration dictionary. See load_config() in
            source/utils/utils.py for more information.

        Returns:
            None
        """
        super().__init__()

        self.model = VisionTransformer(
            image_size=256,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
        )

        # remove the head of the model
        self.model.heads = nn.Identity()

        weights = self.load_and_convert_hipt_checkpoint(
            config["patch_level_hipt_checkpoint_path"]
        )
        self.model.load_state_dict(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the extractor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 256, 256).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 384).
        """
        return self.model(x)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform the input to the format expected by the extractor.

        HIPT expects the input to be normalized to mean 0.5 and standard deviation 0.5.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 256, 256).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3, 256, 256).
        """
        return torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )(x)


class RegionLevelHIPTFeatureExtractor(Extractor, BaseHIPT):
    """
    Extractor that returns the output of the region-level HIPT model.

    This extractor consists of the first and second part of the HIPT model,
    which first extracts features at the patch level (256 x 256 at magnification level 1)
    and then aggregates them to extract features at the region level (4096 x 4096 at magnification
    level 1). Further details can be found in the HIPT paper "Scaling Vision Transformers
    to Gigapixel Images via Hierarchical Self-Supervised Learning" by Chen et al.
    (2022).

    Uses the pretrained weights from the official implementation of the HIPT model,
    which can be found at https://github.com/mahmoodlab/HIPT/tree/master/HIPT_4K/Checkpoints.

    Attributes:
        patch_level_extractor: The patch-level HIPT model.
        region_level_extractor: The region-level HIPT model.

    Methods:
        forward: Forward pass through the extractor.
        transform: Transform the input to the format expected by the extractor.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the patch+region level HIPT feature extractor.

        Uses the PatchLevelHIPTFeatureExtractor to extract features at the patch level, which
        are subsequently used by the region level feature extractor to extract region level
        features.

        Args:
            config (dict): Configuration dictionary. See load_config() in
            source/utils/utils.py for more information.

        Returns:
            None
        """
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

        # The region level extractor does not have a convolutional patch projection
        # layer, so we replace the convolutional patch projection layer of the
        # vision transformer with a linear projection layer as in the original
        # implementation.
        phi = nn.Sequential(
            Lambda(lambda x: x.flatten(2, 3).transpose(1, 2)),
            nn.Linear(384, 192),
            nn.GELU(),
            Lambda(lambda x: x.transpose(1, 2)),
        )

        self.region_level_extractor.conv_proj = phi

        # remove the head of the model
        self.region_level_extractor.heads = nn.Identity()

        weights = self.load_and_convert_hipt_checkpoint(
            config["region_level_hipt_checkpoint_path"]
        )
        self.region_level_extractor.load_state_dict(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the extractor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 4096, 4096).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 192).
        """
        x = x.unfold(2, 256, 256).unfold(3, 256, 256)
        x = rearrange(x, "b c h w p q -> (b h w) c p q")

        patch_features = self.patch_level_extractor(x)
        patch_features = rearrange(patch_features, "(b h w) c -> b c h w", h=16, w=16)

        region_features = self.region_level_extractor(patch_features)

        return region_features

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform the input to the format expected by the extractor.

        HIPT expects the input to be normalized to mean 0.5 and standard deviation 0.5.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 4096, 4096).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3, 4096, 4096).
        """
        # return torchvision.transforms.Normalize(
        #     mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        # )(x)
        return x

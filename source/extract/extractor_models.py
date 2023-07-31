import torch
import torch.nn as nn
import torchvision
import numpy as np
from abc import ABC, abstractmethod
from torchvision.models.vision_transformer import VisionTransformer
from einops import rearrange


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


class PatchLevelHIPTFeatureExtractor(nn.Module):
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
        return x

    def load_and_convert_hipt_checkpoint(self, checkpoint_path):
        weights = torch.load(checkpoint_path)["teacher"]
        weights = self.rename_checkpoint_weights_to_match_model(weights)
        weights = self.interpolate_positional_encoding_weights(weights)
        weights = self.drop_unnecessary_weights(weights)
        return weights

    def rename_checkpoint_weights_to_match_model(self, weights):
        renamed_weights = {}
        for k, v in weights.items():
            renamed_key = k.replace("backbone.", "")
            renamed_key = renamed_key.replace("cls_token", "class_token")
            renamed_key = renamed_key.replace("pos_embed", "encoder.pos_embedding")
            renamed_key = renamed_key.replace("patch_embed.proj", "conv_proj")
            renamed_key = renamed_key.replace(
                "blocks.", "encoder.layers.encoder_layer_"
            )
            renamed_key = renamed_key.replace("norm", "ln_")
            renamed_key = renamed_key.replace("ln_.", "encoder.ln.")
            renamed_key = renamed_key.replace("attn.qkv.", "self_attention.in_proj_")
            renamed_key = renamed_key.replace("attn.proj", "self_attention.out_proj")
            renamed_key = renamed_key.replace("mlp.fc1", "mlp.0")
            renamed_key = renamed_key.replace("mlp.fc2", "mlp.3")

            renamed_weights[renamed_key] = v

        return renamed_weights

    def interpolate_positional_encoding_weights(self, weights):
        class_positional_embedding = weights["encoder.pos_embedding"][:, 0]
        patch_positional_embedding = weights["encoder.pos_embedding"][:, 1:]

        patch_positional_embedding = patch_positional_embedding.reshape(1, 14, 14, 384)
        patch_positional_embedding = patch_positional_embedding.permute((0, 3, 1, 2))
        upsampled_patch_positional_embedding = nn.functional.interpolate(
            input=patch_positional_embedding,
            size=(16, 16),
            mode="bicubic",
            align_corners=False,
        )
        upsampled_patch_positional_embedding = (
            upsampled_patch_positional_embedding.permute((0, 2, 3, 1))
        )
        upsampled_patch_positional_embedding = (
            upsampled_patch_positional_embedding.reshape(1, 256, 384)
        )

        weights["encoder.pos_embedding"] = torch.cat(
            (
                class_positional_embedding.unsqueeze(0),
                upsampled_patch_positional_embedding,
            ),
            dim=1,
        )

        return weights

    def drop_unnecessary_weights(self, weights):
        return {k: v for k, v in weights.items() if not k.startswith("head")}

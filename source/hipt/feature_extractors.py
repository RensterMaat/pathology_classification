import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path
from source.hipt.vision_transformer import vit_small, vit4k_xs
from source.hipt.utils import update_state_dict


class LocalFeatureExtractor(nn.Module):
    def __init__(
        self,
        patch_size: int = 256,
        mini_patch_size: int = 16,
        pretrain_vit_patch: str = "path/to/pretrained/vit_patch/weights.pth",
        embed_dim_patch: int = 384,
        verbose: bool = True,
    ):
        super(LocalFeatureExtractor, self).__init__()
        checkpoint_key = "teacher"

        self.ps = patch_size

        self.vit_patch = vit_small(
            img_size=patch_size,
            patch_size=mini_patch_size,
            embed_dim=embed_dim_patch,
        )

        if Path(pretrain_vit_patch).is_file():
            if verbose:
                print("Loading pretrained weights for patch-level Transformer")
            state_dict = torch.load(pretrain_vit_patch, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                if verbose:
                    print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_patch.state_dict(), state_dict)
            self.vit_patch.load_state_dict(state_dict, strict=False)
            if verbose:
                print(f"Pretrained weights found at {pretrain_vit_patch}")
                print(msg)

        elif verbose:
            print(
                f"{pretrain_vit_patch} doesnt exist ; please provide path to existing file"
            )

        if verbose:
            print("Freezing pretrained patch-level Transformer")
        for param in self.vit_patch.parameters():
            param.requires_grad = False
        if verbose:
            print("Done")

    def forward(self, x):
        # x = [1, 3, region_size, region_size]
        # TODO: add prepare_img_tensor method
        x = x.unfold(2, self.ps, self.ps).unfold(
            3, self.ps, self.ps
        )  # [1, 3, npatch, region_size, ps] -> [1, 3, npatch, npatch, ps, ps]
        x = rearrange(x, "b c p1 p2 w h -> (b p1 p2) c w h")  # [num_patches, 3, ps, ps]

        patch_feature = self.vit_patch(x).detach().cpu()  # [num_patches, 384]

        return patch_feature


class GlobalFeatureExtractor(nn.Module):
    def __init__(
        self,
        region_size: int = 4096,
        patch_size: int = 256,
        mini_patch_size: int = 16,
        pretrain_vit_patch: str = "path/to/pretrained/vit_patch/weights.pth",
        pretrain_vit_region: str = "path/to/pretrained/vit_region/weights.pth",
        embed_dim_patch: int = 384,
        embed_dim_region: int = 192,
        split_across_gpus: bool = False,
        verbose: bool = True,
    ):
        super(GlobalFeatureExtractor, self).__init__()
        checkpoint_key = "teacher"

        self.npatch = int(region_size // patch_size)
        self.ps = patch_size
        self.split_across_gpus = split_across_gpus

        if split_across_gpus:
            self.device_patch = torch.device("cuda:0")
            self.device_region = torch.device("cuda:1")

        self.vit_patch = vit_small(
            img_size=patch_size,
            patch_size=mini_patch_size,
            embed_dim=embed_dim_patch,
        )

        if Path(pretrain_vit_patch).is_file():
            if verbose:
                print("Loading pretrained weights for patch-level Transformer...")
            state_dict = torch.load(pretrain_vit_patch, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                if verbose:
                    print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_patch.state_dict(), state_dict)
            self.vit_patch.load_state_dict(state_dict, strict=False)
            if verbose:
                print(f"Pretrained weights found at {pretrain_vit_patch}")
                print(msg)

        elif verbose:
            print(
                f"{pretrain_vit_patch} doesnt exist ; please provide path to existing file"
            )

        if verbose:
            print("Freezing pretrained patch-level Transformer")
        for param in self.vit_patch.parameters():
            param.requires_grad = False
        if verbose:
            print("Done")

        if split_across_gpus:
            self.vit_patch.to(self.device_patch)

        self.vit_region = vit4k_xs(
            img_size=region_size,
            patch_size=patch_size,
            input_embed_dim=embed_dim_patch,
            output_embed_dim=embed_dim_region,
        )

        if Path(pretrain_vit_region).is_file():
            if verbose:
                print("Loading pretrained weights for region-level Transformer...")
            state_dict = torch.load(pretrain_vit_region, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                if verbose:
                    print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(
                self.vit_region.state_dict(), state_dict
            )
            self.vit_region.load_state_dict(state_dict, strict=False)
            if verbose:
                print(f"Pretrained weights found at {pretrain_vit_region}")
                print(msg)

        elif verbose:
            print(
                f"{pretrain_vit_region} doesnt exist ; please provide path to existing file"
            )

        if verbose:
            print("Freezing pretrained region-level Transformer")
        for param in self.vit_region.parameters():
            param.requires_grad = False
        if verbose:
            print("Done")

        if split_across_gpus:
            self.vit_region.to(self.device_region)

    def forward(self, x):
        # x = [1, 3, region_size, region_size]
        # TODO: add prepare_img_tensor method
        x = x.unfold(2, self.ps, self.ps).unfold(
            3, self.ps, self.ps
        )  # [1, 3, npatch, npatch, ps, ps]
        x = rearrange(
            x, "b c p1 p2 w h -> (b p1 p2) c w h"
        )  # [1*npatch*npatch, 3, ps, ps]
        if self.split_across_gpus:
            x = x.to(self.device_patch, non_blocking=True)  # [num_patches, 3, ps, ps]

        patch_features = self.vit_patch(x)  # [num_patches, 384]
        patch_features = patch_features.unsqueeze(0)  # [1, num_patches, 384]
        patch_features = patch_features.unfold(1, self.npatch, self.npatch).transpose(
            1, 2
        )  # [1, 384, npatch, npatch]
        if self.split_across_gpus:
            patch_features = patch_features.to(self.device_region, non_blocking=True)

        region_feature = self.vit_region(patch_features).cpu()  # [1, 192]

        return region_feature

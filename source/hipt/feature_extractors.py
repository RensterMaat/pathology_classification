import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path
from source.hipt.vision_transformer import vit_small
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

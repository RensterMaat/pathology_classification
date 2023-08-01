import torch
import torch.nn as nn


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class GlobalGatedAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()

        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout)
        )
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout)
        )

        self.attention_c = nn.Linear(*[hidden_dim, output_dim])
        # self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)

        A = self.attention_c(A)

        return A
        # attention = self.softmax(A)

        # return attention


class BaseHIPT:
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
            renamed_key = renamed_key.replace("phi.0", "conv_proj.1")

            renamed_weights[renamed_key] = v

        return renamed_weights

    def interpolate_positional_encoding_weights(self, weights):
        class_positional_embedding = weights["encoder.pos_embedding"][:, 0]
        patch_positional_embedding = weights["encoder.pos_embedding"][:, 1:]

        patch_positional_embedding = patch_positional_embedding.reshape(1, 14, 14, -1)
        patch_positional_embedding = patch_positional_embedding.permute((0, 3, 1, 2))
        upsampled_patch_positional_embedding = nn.functional.interpolate(
            input=patch_positional_embedding,
            scale_factor=(16.1 / 14, 16.1 / 14),
            mode="bicubic",
        )
        upsampled_patch_positional_embedding = (
            upsampled_patch_positional_embedding.permute((0, 2, 3, 1))
        )
        upsampled_patch_positional_embedding = (
            upsampled_patch_positional_embedding.reshape(1, 256, -1)
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

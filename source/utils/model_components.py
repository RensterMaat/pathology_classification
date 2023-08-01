import torch
import torch.nn as nn


class Lambda(nn.Module):
    """
    Lambda layer for wrapping a function as a torch.nn.Module.

    Attributes:
        func (callable): Function to wrap.

    Methods:
        forward: Forward pass through the layer.

    """

    def __init__(self, func: callable) -> None:
        super().__init__()
        self.func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)


class GlobalGatedAttentionPooling(nn.Module):
    """
    Global Gated Attention Pooling layer.

    Passes the input through 2 linear layers (attention_a and attention_b). The final 
    attention is computed as the element-wise multiplication of the outputs of the two
    linear layers. Softmax normalization is not applied; this should be done manually. 

    Implementation of Attention-based Deep Multiple Instance Learning, Maximilian Ilse, 
    Jakub Tomczak, Max Welling, Proceedings of the 35th International Conference on \
    Machine Learning, PMLR 80:2127-2136, 2018.

    Attributes:
        attention_a (torch.nn.Sequential): First linear layer.
        attention_b (torch.nn.Sequential): Second linear layer.
        attention_c (torch.nn.Linear): Final linear layer.

    Methods:
        forward: Forward pass through the layer.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:
        """
        Initialize the layer.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            dropout (float): Dropout probability.
        """
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
        """
        Forward pass through the layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_instances, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)

        A = self.attention_c(A)

        return A
        # attention = self.softmax(A)

        # return attention


class BaseHIPT:
    """
    Base class for HIPT models.

    Attributes:
        None

    Methods:
        load_and_convert_hipt_checkpoint: Load and convert a HIPT checkpoint.
        rename_checkpoint_weights_to_match_model: Rename the weights of a HIPT checkpoint
            to match the model.
        interpolate_positional_encoding_weights: Interpolate the positional encoding
            weights of a HIPT checkpoint to match the model.
        drop_unnecessary_weights: Drop weights which are present in the HIPT checkpoint,
            but not in the model.

    """

    def load_and_convert_hipt_checkpoint(self, checkpoint_path):
        """
        Load and convert a HIPT checkpoint.

        Works in three steps:
            1. Rename the weights of the checkpoint to match the model.
            2. Interpolate the positional encoding weights of the checkpoint to match the
                model.
            3. Drop weights which are present in the checkpoint, but not in the model.

        Args:
            checkpoint_path (str): Path to the checkpoint.

        Returns:
            dict: Converted weights.


        """
        weights = torch.load(checkpoint_path)["teacher"]
        weights = self.rename_checkpoint_weights_to_match_model(weights)
        weights = self.interpolate_positional_encoding_weights(weights)
        weights = self.drop_unnecessary_weights(weights)
        return weights

    def rename_checkpoint_weights_to_match_model(self, weights):
        """
        Rename the weights of a HIPT checkpoint to match the model.

        Args:
            weights (dict): Weights of the checkpoint.

        Returns:
            dict: Renamed weights.
        """
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
        """
        Interpolate the positional encoding weights of a HIPT checkpoint to match the
        model.

        The weights for the positional encoding of the HIPT checkpoint are of shape
        (1, 14 x 14, 768). The weights for the positional encoding of the model are of shape
        (1, 16 x 16, 768). The checkpoint weights have this shape as the model was trained
        using DINO; during DINO training, images are cropped with a factor 0.875 (and 16 * 0.875
        = 14).

        This method interpolates the weights of the checkpoint to match the model. In the original
        implementation, the positional encodings are calculated at every forward pass. However,
        here we interpolate the weights to match the model, as this is more efficient.

        The same method is used for interpolating the weights as in the original implementation:
        1. Class and patch positional encodings are separated.
        2. The patch positional encodings are reshaped to (1, 14, 14, 768).
        3. The patch positional encodings are transposed to (1, 768, 14, 14).
        4. The patch positional encodings are upsampled to (1, 768, 16.1, 16.1) using bicubic
            interpolation. A small factor of 0.1 is added to the upsampled shape to prevent
            rounding errors (see original implementation and discussion at
            https://github.com/facebookresearch/dino/issues/8).
        5. The patch positional encodings are transposed to (1, 16, 16, 768).
        6. The class positional encodings are concatenated with the patch positional encodings.

        Args:
            weights (dict): Weights of the checkpoint.

        Returns:
            dict: Interpolated weights.
        """
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
        """
        Drop weights which are present in the checkpoint, but not in the model.

        Args:
            weights (dict): Weights of the checkpoint.

        Returns:
            dict: Weights without the unnecessary weights.
        """
        return {k: v for k, v in weights.items() if not k.startswith("head")}

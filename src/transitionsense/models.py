"""Small supervised baselines for six-channel 50 Hz IMU windows.

``ResNet1D`` is a compact 1-D ResNet; ``PatchTransformer1D`` is a patch-embedding
Transformer over non-overlapping 8-sample patches. Both take ``[B, 6, 128]`` inputs.
"""

from __future__ import annotations

import torch
from torch import nn

from transitionsense.labels import N_CLASSES
from transitionsense.data import N_CHANNELS, WINDOW


class BasicBlock1D(nn.Module):
    """A pre-activation-style residual block with two convolutions."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Build the block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            stride: Convolution stride of the first layer.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block."""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet1D(nn.Module):
    """A small 1-D ResNet for IMU window classification."""

    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        num_classes: int = N_CLASSES,
        window: int = WINDOW,
        width: int = 32,
    ) -> None:
        """Build the network.

        Args:
            in_channels: Input channel count.
            num_classes: Number of output classes.
            window: Input window length (kept for config symmetry).
            width: Base channel width; stages use ``width``, ``2*width``, ``4*width``.
        """
        super().__init__()
        self.window = window
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlock1D(width, width), BasicBlock1D(width, width))
        self.layer2 = nn.Sequential(BasicBlock1D(width, width * 2, stride=2), BasicBlock1D(width * 2, width * 2))
        self.layer3 = nn.Sequential(
            BasicBlock1D(width * 2, width * 4, stride=2), BasicBlock1D(width * 4, width * 4)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network on ``[B, C, T]`` inputs."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


class PatchTransformer1D(nn.Module):
    """A patch-embedding Transformer over non-overlapping windows of each channel mix."""

    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        num_classes: int = N_CLASSES,
        window: int = WINDOW,
        patch_size: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """Build the network.

        Args:
            in_channels: Input channel count.
            num_classes: Number of output classes.
            window: Input window length; must be divisible by ``patch_size``.
            patch_size: Patch length in samples.
            d_model: Token dimension.
            nhead: Attention heads.
            num_layers: Encoder layers.
            dim_feedforward: Feed-forward width.
            dropout: Dropout rate.
        """
        super().__init__()
        if window % patch_size:
            raise ValueError(f"window {window} is not divisible by patch_size {patch_size}")
        self.window = window
        self.patch_size = patch_size
        n_patches = window // patch_size
        self.patch_embed = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network on ``[B, C, T]`` inputs."""
        tokens = self.patch_embed(x).transpose(1, 2) + self.pos_embed
        tokens = self.encoder(tokens)
        pooled = self.norm(tokens).mean(dim=1)
        return self.head(pooled)


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name.

    Args:
        name: ``"resnet"`` or ``"transformer"``.
        **kwargs: Forwarded to the model constructor.

    Returns:
        The model.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name == "resnet":
        return ResNet1D(**kwargs)
    if name == "transformer":
        return PatchTransformer1D(**kwargs)
    raise ValueError(f"unknown model {name!r}; expected 'resnet' or 'transformer'")


def parameter_count(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

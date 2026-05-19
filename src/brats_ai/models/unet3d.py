"""Compact 3D UNet baseline."""

from __future__ import annotations

from brats_ai.exceptions import MissingDependencyError

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - allows syntax/import in minimal envs
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise MissingDependencyError("Install PyTorch to instantiate UNet3D.")


class ConvBlock(nn.Module if nn is not None else object):
    """Two Conv3D layers with normalization and GELU activation."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        _require_torch()
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module if nn is not None else object):
    """Memory-conscious 3D UNet for BraTS-style four-channel inputs."""

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 24, dropout: float = 0.1):
        _require_torch()
        super().__init__()
        ch = base_channels
        self.enc1 = ConvBlock(in_channels, ch, dropout)
        self.enc2 = ConvBlock(ch, ch * 2, dropout)
        self.enc3 = ConvBlock(ch * 2, ch * 4, dropout)
        self.bottleneck = ConvBlock(ch * 4, ch * 8, dropout)
        self.pool = nn.MaxPool3d(2)
        self.up3 = nn.ConvTranspose3d(ch * 8, ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(ch * 8, ch * 4, dropout)
        self.up2 = nn.ConvTranspose3d(ch * 4, ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(ch * 4, ch * 2, dropout)
        self.up1 = nn.ConvTranspose3d(ch * 2, ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(ch * 2, ch, dropout)
        self.head = nn.Conv3d(ch, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


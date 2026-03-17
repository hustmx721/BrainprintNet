"""BrainprintNet family models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import TEMPORAL_LAYERS, subband_split


class BrainprintNet(nn.Module):
    """Main BrainprintNet architecture."""

    def __init__(
        self,
        kernels: list[int],
        sampling_rate: int,
        temporal_layer: str = "LogVarLayer",
        stride_factor: int = 5,
        in_channels: int = 22,
        n_bands: int = 12,
        num_classes: int = 9,
        radix: int = 8,
    ) -> None:
        """Initialise BrainprintNet."""

        super().__init__()
        self.sampling_rate = sampling_rate
        self.kernels = kernels
        self.stride_factor = stride_factor
        self.parallel_conv = nn.ModuleList(
            [
                nn.Conv2d(n_bands, n_bands, kernel_size=(1, kernel_size), stride=1, padding=0, bias=False, groups=n_bands)
                for kernel_size in kernels
            ]
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(n_bands),
            nn.ReLU(inplace=False),
            nn.Conv2d(n_bands, n_bands * radix, kernel_size=(in_channels, 1), stride=1, padding=0, bias=False),
        )
        self.temporal_layer = TEMPORAL_LAYERS[temporal_layer](dim=-1)
        self.classifier = nn.Sequential(
            nn.Linear(n_bands * radix * stride_factor, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run BrainprintNet and return features plus logits."""

        signal = torch.squeeze(input_tensor)
        device = signal.device
        signal = torch.from_numpy(subband_split(signal.cpu().numpy(), 8, 32, 2, self.sampling_rate)).to(device)
        outputs = [conv(signal) for conv in self.parallel_conv]
        features = torch.cat(outputs, dim=-1)
        features = self.conv_block(features)
        features = torch.squeeze(features)
        pad_length = self.stride_factor - (features.shape[-1] % self.stride_factor)
        if pad_length != 0:
            features = F.pad(features, (0, pad_length))
        features = features.reshape([*features.shape[0:2], self.stride_factor, int(features.shape[-1] / self.stride_factor)])
        features = self.temporal_layer(features)
        features = torch.flatten(features, start_dim=1)
        return features, self.classifier(features)

    def _get_fea_dim(self, input_tensor: torch.Tensor) -> torch.Size:
        """Return the emitted feature shape."""

        with torch.no_grad():
            features, _ = self.forward(input_tensor)
        return features.shape


class MSNet(nn.Module):
    """Multi-scale temporal CNN used in BrainprintNet ablations."""

    def __init__(
        self,
        kernels: list[int],
        hidden_chans: int = 64,
        temporal_layer: str = "LogVarLayer",
        stride_factor: int = 5,
        in_channels: int = 22,
        num_classes: int = 9,
    ) -> None:
        """Initialise MSNet."""

        super().__init__()
        self.kernels = kernels
        self.hidden_chans = hidden_chans
        self.stride_factor = stride_factor
        self.parallel_conv = nn.ModuleList(
            [nn.Conv1d(in_channels, hidden_chans, kernel_size=kernel_size, stride=1, padding=0, bias=False) for kernel_size in kernels]
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(hidden_chans),
            nn.ReLU(inplace=False),
            nn.Conv1d(hidden_chans, hidden_chans, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(hidden_chans),
            nn.ReLU(inplace=False),
        )
        self.temporal_layer = TEMPORAL_LAYERS[temporal_layer](dim=-1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_chans * stride_factor, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run MSNet and return features plus logits."""

        signal = input_tensor.squeeze(1) if input_tensor.dim() == 4 else input_tensor
        outputs = [conv(signal) for conv in self.parallel_conv]
        features = torch.cat(outputs, dim=-1)
        features = self.conv_block(features)
        pad_length = self.stride_factor - (features.shape[-1] % self.stride_factor)
        if pad_length != 0:
            features = F.pad(features, (0, pad_length))
        features = features.reshape([*features.shape[0:2], self.stride_factor, int(features.shape[-1] / self.stride_factor)])
        features = self.temporal_layer(features)
        features = torch.flatten(features, start_dim=1)
        return features, self.classifier(features)

    def _get_fea_dim(self, input_tensor: torch.Tensor) -> torch.Size:
        """Return the emitted feature shape."""

        with torch.no_grad():
            features, _ = self.forward(input_tensor)
        return features.shape


class CBFNet(BrainprintNet):
    """Single-kernel ablation of BrainprintNet."""


class BrainprintNetFeatureExtractor(BrainprintNet):
    """Feature extractor half of BrainprintNet for transfer-learning style workflows."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return features without a classifier head."""

        features, _ = super().forward(input_tensor)
        return features


class BrainprintNetClassifier(nn.Module):
    """Classifier head paired with :class:`BrainprintNetFeatureExtractor`."""

    def __init__(self, n_bands: int = 12, radix: int = 8, stride_factor: int = 5, num_classes: int = 9) -> None:
        """Initialise the classifier head."""

        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_bands * radix * stride_factor, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return features and logits."""

        return input_tensor, self.classifier(input_tensor)


class MSNetFeatureExtractor(MSNet):
    """Feature extractor half of MSNet."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return features without a classifier head."""

        features, _ = super().forward(input_tensor)
        return features


class MSNetClassifier(nn.Module):
    """Classifier head paired with :class:`MSNetFeatureExtractor`."""

    def __init__(self, hidden_chans: int = 64, stride_factor: int = 5, num_classes: int = 9) -> None:
        """Initialise the classifier head."""

        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_chans * stride_factor, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return features and logits."""

        return input_tensor, self.classifier(input_tensor)

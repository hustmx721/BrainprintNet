"""Filter-bank based EEG backbones."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from .common import (
    Conv2dWithConstraint,
    LinearWithConstraint,
    Swish,
    TEMPORAL_LAYERS,
    subband_split,
    subband_split_edges,
)


class FBCNet(nn.Module):
    """Filter Bank Convolutional Network."""

    def __init__(
        self,
        n_channels: int,
        sampling_rate: int = 250,
        n_classes: int = 2,
        n_bands: int = 12,
        multiplier: int = 32,
        temporal_layer: str = "LogVarLayer",
        stride_factor: int = 4,
        use_weight_norm: bool = True,
    ) -> None:
        """Initialise FBCNet."""

        super().__init__()
        self.sampling_rate = sampling_rate
        self.stride_factor = stride_factor
        self.scb = nn.Sequential(
            Conv2dWithConstraint(
                n_bands,
                multiplier * n_bands,
                (n_channels, 1),
                groups=n_bands,
                max_norm=2,
                do_weight_norm=use_weight_norm,
                padding=0,
            ),
            nn.BatchNorm2d(multiplier * n_bands),
            Swish(),
        )
        self.temporal_layer = TEMPORAL_LAYERS[temporal_layer](dim=3)
        self.classifier = nn.Sequential(
            LinearWithConstraint(multiplier * n_bands * stride_factor, n_classes, max_norm=0.5, do_weight_norm=use_weight_norm),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run FBCNet and return features plus logits."""

        signal = torch.squeeze(input_tensor.squeeze(1) if input_tensor.dim() == 5 else input_tensor)
        device = signal.device
        signal = torch.from_numpy(subband_split(signal.cpu().numpy(), 8, 32, 2, self.sampling_rate)).to(device)
        features = self.scb(signal)
        pad_length = features.shape[3] % self.stride_factor
        if pad_length != 0:
            features = F.pad(features, (0, pad_length))
        features = features.reshape([*features.shape[0:2], self.stride_factor, int(features.shape[3] / self.stride_factor)])
        features = self.temporal_layer(features)
        features = torch.flatten(features, start_dim=1)
        return features, self.classifier(features)

    def _get_fea_dim(self, input_tensor: torch.Tensor) -> torch.Size:
        """Return the emitted feature shape."""

        with torch.no_grad():
            features, _ = self.forward(input_tensor)
        return features.shape


def _is_static_pad(kernel_size: int | tuple[int, int], stride: int = 1, dilation: int = 1) -> bool:
    """Return whether symmetric padding can be resolved statically."""

    value = max(kernel_size) if isinstance(kernel_size, tuple) else kernel_size
    return stride == 1 and (dilation * (value - 1)) % 2 == 0


def _get_padding(kernel_size: int | tuple[int, int], stride: int = 1, dilation: int = 1) -> int:
    """Compute symmetric convolution padding."""

    value = max(kernel_size) if isinstance(kernel_size, tuple) else kernel_size
    return ((stride - 1) + dilation * (value - 1)) // 2


def _calc_same_pad(size: int, kernel_size: int, stride: int, dilation: int) -> int:
    """Compute TensorFlow-style same padding size."""

    return max(((-(size // -stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - size), 0)


def _same_pad_arg(
    input_size: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
) -> list[int]:
    """Build a torch ``F.pad`` argument list."""

    input_h, input_w = input_size
    kernel_h, kernel_w = kernel_size
    pad_h = _calc_same_pad(input_h, kernel_h, stride[0], dilation[0])
    pad_w = _calc_same_pad(input_w, kernel_w, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]


def _split_channels(num_channels: int, num_groups: int) -> list[int]:
    """Split channels as evenly as possible between mixed convolutions."""

    split = [num_channels // num_groups for _ in range(num_groups)]
    split[0] += num_channels - sum(split)
    return split


def conv2d_same(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
) -> torch.Tensor:
    """Apply TensorFlow-style same padding before convolution."""

    input_h, input_w = input_tensor.size()[-2:]
    kernel_h, kernel_w = weight.size()[-2:]
    pad_h = _calc_same_pad(input_h, kernel_h, stride[0], dilation[0])
    pad_w = _calc_same_pad(input_w, kernel_w, stride[1], dilation[1])
    padded = F.pad(input_tensor, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(padded, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """Conv2D module with TensorFlow-style same padding."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply same-padded convolution."""

        return conv2d_same(input_tensor, self.weight, self.bias, self.stride, self.dilation, self.groups)


def create_conv2d_pad(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int = 1,
    padding: str = "",
    dilation: int = 1,
    groups: int = 1,
    bias: bool = False,
) -> nn.Module:
    """Create a Conv2D module with optional same padding."""

    dynamic = False
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == "same":
            if _is_static_pad(kernel_size, stride=stride, dilation=dilation):
                padding = _get_padding(kernel_size, stride=stride, dilation=dilation)
            else:
                padding = 0
                dynamic = True
        elif padding == "valid":
            padding = 0
        else:
            padding = _get_padding(kernel_size, stride=stride, dilation=dilation)
    if dynamic:
        return Conv2dSame(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
    if isinstance(kernel_size, tuple):
        padding = (0, padding)
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)


class MixedConv2d(nn.ModuleDict):
    """Grouped mixed-kernel convolution used by FBMSNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[tuple[int, int]] | list[int],
        stride: int = 1,
        padding: str = "",
        dilation: int = 1,
        depthwise: bool = False,
    ) -> None:
        """Create mixed-kernel branches."""

        super().__init__()
        in_splits = _split_channels(in_channels, len(kernel_sizes))
        out_splits = _split_channels(out_channels, len(kernel_sizes))
        self.splits = in_splits
        for index, (kernel_size, in_split, out_split) in enumerate(zip(kernel_sizes, in_splits, out_splits)):
            groups = out_split if depthwise else 1
            self.add_module(
                str(index),
                create_conv2d_pad(in_split, out_split, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups),
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply mixed convolutions and concatenate outputs."""

        split_input = torch.split(input_tensor, self.splits, dim=1)
        outputs = [conv(split_input[index]) for index, conv in enumerate(self.values())]
        return torch.cat(outputs, dim=1)


class FBMSNet(nn.Module):
    """Filter Bank Mixed-Scale Network."""

    def __init__(
        self,
        sampling_rate: int,
        n_channels: int,
        n_time: int,
        n_classes: int = 4,
        temporal_layer: str = "LogVarLayer",
        num_features: int = 36,
        dilatability: int = 8,
    ) -> None:
        """Initialise FBMSNet."""

        super().__init__()
        self.sampling_rate = sampling_rate
        self.stride_factor = 4
        self.mix_conv2d = nn.Sequential(
            MixedConv2d(12, num_features, kernel_sizes=[(1, 15), (1, 31), (1, 63), (1, 125)], stride=1, padding=""),
            nn.BatchNorm2d(num_features),
        )
        self.scb = nn.Sequential(
            Conv2dWithConstraint(num_features, num_features * dilatability, (n_channels, 1), groups=num_features, max_norm=2, padding=0),
            nn.BatchNorm2d(num_features * dilatability),
            Swish(),
        )
        self.temporal_layer = TEMPORAL_LAYERS[temporal_layer](dim=3)
        size = self._get_size(n_channels, n_time)
        self.classifier = nn.Sequential(
            LinearWithConstraint(size[1], n_classes, max_norm=0.5),
            nn.LogSoftmax(dim=1),
        )

    def _get_size(self, n_channels: int, n_time: int) -> torch.Size:
        """Infer the flattened feature size."""

        dummy = torch.ones((1, 12, n_channels, n_time))
        features = self.mix_conv2d(dummy)
        features = self.scb(features)
        features = features.reshape([*features.shape[0:2], self.stride_factor, int(features.shape[3] / self.stride_factor)])
        features = self.temporal_layer(features)
        features = torch.flatten(features, start_dim=1)
        return features.size()

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run FBMSNet and return features plus logits."""

        signal = torch.squeeze(input_tensor.squeeze(1) if input_tensor.dim() == 5 else input_tensor)
        device = signal.device
        signal = torch.from_numpy(subband_split(signal.cpu().numpy(), 8, 32, 2, self.sampling_rate)).to(device)
        features = self.mix_conv2d(signal)
        features = self.scb(features)
        features = features.reshape([*features.shape[0:2], self.stride_factor, int(features.shape[3] / self.stride_factor)])
        features = self.temporal_layer(features)
        features = torch.flatten(features, start_dim=1)
        return features, self.classifier(features)

    def _get_fea_dim(self, input_tensor: torch.Tensor) -> torch.Size:
        """Return the emitted feature shape."""

        with torch.no_grad():
            features, _ = self.forward(input_tensor)
        return features.shape


class ConvBlock(nn.Module):
    """Convenience wrapper around convolution, batch norm, and activation."""

    def __init__(self, conv: nn.Module, activation: Optional[nn.Module] = None, batch_norm: Optional[nn.Module] = None) -> None:
        """Store layer components."""

        super().__init__()
        self.conv = conv
        self.activation = activation
        self.batch_norm = batch_norm
        if batch_norm is not None:
            self.conv.bias = None

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped convolution block."""

        output = self.conv(input_tensor)
        if self.batch_norm is not None:
            output = self.batch_norm(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


class LogPowerLayer(nn.Module):
    """Compute log-power along the patch dimension."""

    def __init__(self, dim: int) -> None:
        """Store the reduction dimension."""

        super().__init__()
        self.dim = dim

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return the clamped log-power."""

        return torch.log(torch.clamp(torch.mean(input_tensor**2, dim=self.dim), 1e-4, 1e4))


class InterFrequency(nn.Module):
    """Fuse multiple frequency branches."""

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Sum the branches and apply GELU."""

        return F.gelu(sum(inputs))


class IFStem(nn.Module):
    """Stem block used by IFNet."""

    def __init__(self, in_planes: int, out_planes: int = 64, kernel_size: int = 63, patch_size: int = 125, radix: int = 2) -> None:
        """Initialise the interactive frequency stem."""

        super().__init__()
        self.out_planes = out_planes
        self.patch_size = patch_size
        self.radix = radix
        self.sconv = ConvBlock(nn.Conv1d(in_planes, out_planes * radix, 1, bias=False, groups=radix), batch_norm=nn.BatchNorm1d(out_planes * radix))
        temporal_convs = []
        current_kernel = kernel_size
        for _ in range(radix):
            temporal_convs.append(
                ConvBlock(
                    nn.Conv1d(out_planes, out_planes, current_kernel, 1, groups=out_planes, padding=current_kernel // 2, bias=False),
                    batch_norm=nn.BatchNorm1d(out_planes),
                )
            )
            current_kernel = current_kernel // 2 if current_kernel == 63 else current_kernel // 2 + 1
        self.temporal_convs = nn.ModuleList(temporal_convs)
        self.fusion = InterFrequency()
        self.power = LogPowerLayer(dim=3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run the IFNet stem."""

        batch_size, _, time_points = input_tensor.shape
        output = self.sconv(input_tensor)
        split_output = torch.split(output, self.out_planes, dim=1)
        output = [conv(branch) for branch, conv in zip(split_output, self.temporal_convs)]
        output = self.fusion(output)
        output = output.reshape(batch_size, self.out_planes, time_points // self.patch_size, self.patch_size)
        output = self.power(output)
        return self.dropout(output)


class IFNet(nn.Module):
    """Interactive Frequency Convolutional Neural Network."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        radix: int,
        patch_size: int,
        time_points: int,
        num_classes: int,
        sampling_rate: int,
    ) -> None:
        """Initialise IFNet."""

        super().__init__()
        self.sampling_rate = sampling_rate
        self.stem = IFStem(in_planes * radix, out_planes, kernel_size, patch_size=patch_size, radix=radix)
        self.classifier = nn.Sequential(LinearWithConstraint(out_planes * (time_points // patch_size), num_classes, do_weight_norm=True))
        self.apply(self._init_parameters)

    def _init_parameters(self, module: nn.Module) -> None:
        """Initialise module parameters."""

        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run IFNet and return features plus logits."""

        signal = input_tensor if input_tensor.dim() == 3 else input_tensor.squeeze(1)
        device = signal.device
        split_signal = subband_split_edges(signal.cpu().numpy(), [(8, 16), (16, 32)], self.sampling_rate)
        signal = torch.from_numpy(split_signal).to(torch.float32).to(device)
        signal = torch.squeeze(signal)
        features = self.stem(signal)
        flattened = features.flatten(1)
        return features, self.classifier(flattened)

    def _get_fea_dim(self, input_tensor: torch.Tensor) -> torch.Size:
        """Return the emitted feature shape."""

        with torch.no_grad():
            features, _ = self.forward(input_tensor)
        return features.shape

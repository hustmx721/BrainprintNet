"""Classical CNN and recurrent EEG backbones."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import calculate_output_size


class MLPClassifier(nn.Module):
    """Two-layer multilayer perceptron classifier."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 50) -> None:
        """Build the classifier head."""

        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Project features to class logits."""

        return self.network(input_tensor)


class EEGNet(nn.Module):
    """EEGNet backbone used for compact EEG classification."""

    def __init__(
        self,
        classes_num: int,
        chans: int,
        samples: int,
        kernel_size: int = 64,
        f1: int = 8,
        f2: int = 16,
        depth_multiplier: int = 2,
        dropout: float = 0.5,
    ) -> None:
        """Initialise EEGNet layers."""

        super().__init__()
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((kernel_size // 2 - 1, kernel_size - kernel_size // 2, 0, 0)),
            nn.Conv2d(1, f1, kernel_size=(1, kernel_size), stride=1, bias=False),
            nn.BatchNorm2d(f1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, f1 * depth_multiplier, kernel_size=(chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(f1 * depth_multiplier, f1 * depth_multiplier, kernel_size=(1, 16), groups=f1 * depth_multiplier, bias=False),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.Conv2d(f1 * depth_multiplier, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        feature_size = f2 * (samples // 32)
        self.classifier = MLPClassifier(feature_size, classes_num)

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run EEGNet and return features plus logits."""

        features = self.block1(input_tensor)
        features = self.block2(features)
        features = self.block3(features)
        features = features.view(features.size(0), -1)
        return features, self.classifier(features)

    def _get_fea_dim(self, input_tensor: torch.Tensor) -> torch.Size:
        """Return the emitted feature shape."""

        with torch.no_grad():
            features, _ = self.forward(input_tensor)
        return features.shape


class DeepConvNet(nn.Module):
    """DeepConvNet baseline for EEG classification."""

    def __init__(
        self,
        chans: int,
        samples: int,
        classes_num: int,
        dropout: float = 0.5,
        d1: int = 25,
        d2: int = 50,
        d3: int = 100,
    ) -> None:
        """Initialise DeepConvNet."""

        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, d1, kernel_size=(1, 5)),
            nn.Conv2d(d1, d1, kernel_size=(chans, 1)),
            nn.BatchNorm2d(d1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(d1, d2, kernel_size=(1, 5)),
            nn.BatchNorm2d(d2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(d2, d3, kernel_size=(1, 5)),
            nn.BatchNorm2d(d3),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout),
        )
        feature_size = calculate_output_size(nn.Sequential(self.block1, self.block2, self.block3), chans, samples)
        self.classifier = MLPClassifier(feature_size, classes_num)

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run DeepConvNet and return features plus logits."""

        features = self.block1(input_tensor)
        features = self.block2(features)
        features = self.block3(features)
        features = features.reshape(features.size(0), -1)
        return features, self.classifier(features)

    def _get_fea_dim(self, input_tensor: torch.Tensor) -> torch.Size:
        """Return the emitted feature shape."""

        with torch.no_grad():
            features, _ = self.forward(input_tensor)
        return features.shape


class ShallowConvNet(nn.Module):
    """ShallowConvNet baseline for EEG classification."""

    def __init__(
        self,
        classes_num: int,
        chans: int,
        samples: int,
        dropout: float = 0.5,
        mid_dim: int = 40,
    ) -> None:
        """Initialise ShallowConvNet."""

        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, mid_dim, kernel_size=(1, 13)),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=(chans, 1)),
            nn.BatchNorm2d(mid_dim),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        feature_size = calculate_output_size(self.block1, chans, samples)
        self.classifier = MLPClassifier(feature_size, classes_num)

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ShallowConvNet and return features plus logits."""

        features = self.block1(input_tensor)
        features = features.reshape(features.size(0), -1)
        return features, self.classifier(features)

    def _get_fea_dim(self, input_tensor: torch.Tensor) -> torch.Size:
        """Return the emitted feature shape."""

        with torch.no_grad():
            features, _ = self.forward(input_tensor)
        return features.shape


class CNNLSTM(nn.Module):
    """CNN-LSTM hybrid for EEG classification."""

    def __init__(
        self,
        channels: int,
        time_points: int,
        hidden_size: int,
        n_classes: int,
        num_layers: int,
        spatial_num: int = 32,
        dropout: float = 0.25,
    ) -> None:
        """Initialise CNN-LSTM."""

        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, spatial_num, (channels, 1), bias=False),
            nn.BatchNorm2d(spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(spatial_num, 2 * spatial_num, (1, 1), bias=False),
            nn.BatchNorm2d(2 * spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(2 * spatial_num, 4 * spatial_num, (1, 1), bias=False),
            nn.BatchNorm2d(4 * spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout),
        )
        conv_block = nn.Sequential(self.block1, self.block2, self.block3)
        conv_t = calculate_output_size(conv_block, channels, time_points)
        self.lstm = nn.LSTM(8 * conv_t, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * spatial_num // 2, hidden_size),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run CNN-LSTM and return features plus logits."""

        features = self.block1(input_tensor)
        features = self.block2(features)
        features = self.block3(features)
        features = features.reshape(features.shape[0], -1, features.shape[-1] * 8)
        features, _ = self.lstm(features)
        features = features.reshape(features.shape[0], -1)
        return features, self.classifier(features)

    def _get_fea_dim(self, input_tensor: torch.Tensor) -> torch.Size:
        """Return the emitted feature shape."""

        with torch.no_grad():
            features, _ = self.forward(input_tensor)
        return features.shape


class ResNet1DBlock(nn.Module):
    """Residual one-dimensional convolution block used by GWNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        downsampling: nn.Module,
    ) -> None:
        """Build the residual block."""

        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.25, inplace=False)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.downsampling = downsampling

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply the residual block."""

        identity = self.downsampling(input_tensor)
        output = self.bn1(input_tensor)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.conv1(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.conv2(output)
        output = self.avgpool(output)
        return output + identity


class GWNet(nn.Module):
    """CNN-GRU hybrid baseline used throughout the original project."""

    def __init__(
        self,
        kernels: list[int],
        samples: int,
        res_blocks: int = 2,
        hidden_chans: int = 32,
        in_channels: int = 22,
        fixed_kernel_size: int = 5,
        num_classes: int = 9,
    ) -> None:
        """Initialise GWNet."""

        super().__init__()
        self.kernels = kernels
        self.planes = hidden_chans
        self.parallel_conv = nn.ModuleList(
            [
                nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=kernel_size, stride=1, padding=0, bias=False)
                for kernel_size in kernels
            ]
        )
        self.bn1 = nn.BatchNorm1d(self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(self.planes, self.planes, kernel_size=fixed_kernel_size, stride=2, padding=2, bias=False)
        self.block = self._make_resnet_layer(fixed_kernel_size, padding=fixed_kernel_size // 2, blocks=res_blocks)
        self.bn2 = nn.BatchNorm1d(self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=8, stride=8)

        parallel_out = len(kernels) * samples - sum(kernels) + len(kernels)
        conv_out = parallel_out // 2
        block_out = conv_out // (2**res_blocks)
        avgpool_out = block_out // 8
        classifier_in = self.planes * avgpool_out + 2 * 4 * self.planes
        self.rnn = nn.GRU(input_size=in_channels, hidden_size=4 * self.planes, num_layers=2, bidirectional=True)
        self.classifier = nn.Linear(classifier_in, num_classes)

    def _make_resnet_layer(self, kernel_size: int, padding: int, blocks: int) -> nn.Sequential:
        """Build stacked residual blocks."""

        layers = []
        for _ in range(blocks):
            downsampling = nn.Sequential(nn.AvgPool1d(kernel_size=2, stride=2))
            layers.append(
                ResNet1DBlock(
                    in_channels=self.planes,
                    out_channels=self.planes,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    downsampling=downsampling,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GWNet and return features plus logits."""

        if input_tensor.dim() == 4:
            input_tensor = input_tensor.squeeze(1)
        outputs = [conv(input_tensor) for conv in self.parallel_conv]
        features = torch.cat(outputs, dim=-1)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.conv1(features)
        features = self.block(features)
        features = self.bn2(features)
        features = self.relu(features)
        features = self.avgpool(features)
        features = features.reshape(features.shape[0], -1)
        recurrent_features, _ = self.rnn(input_tensor.permute(0, 2, 1))
        last_state = recurrent_features[:, -1, :]
        features = torch.cat([features, last_state], dim=1)
        return features, self.classifier(features)

    def _get_fea_dim(self, input_tensor: torch.Tensor) -> torch.Size:
        """Return the emitted feature shape."""

        with torch.no_grad():
            features, _ = self.forward(input_tensor)
        return features.shape


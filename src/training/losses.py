"""Loss functions used by BrainprintNet."""

from __future__ import annotations

import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss from Wen et al., ECCV 2016."""

    def __init__(self, num_classes: int, feat_dim: int, use_gpu: bool = True) -> None:
        """Initialise learnable class centres."""

        super().__init__()
        centers = torch.randn(num_classes, feat_dim)
        if use_gpu and torch.cuda.is_available():
            centers = centers.cuda()
        self.centers = nn.Parameter(centers)
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute center loss for one batch."""

        batch_size = features.size(0)
        distance_matrix = (
            torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        )
        distance_matrix.addmm_(features, self.centers.t(), beta=1.0, alpha=-2.0)
        classes = torch.arange(self.num_classes, device=features.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        distance = distance_matrix * mask.float()
        return distance.clamp(min=1e-12, max=1e12).sum() / batch_size

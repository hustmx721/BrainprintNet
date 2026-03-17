"""Validation and test-time evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_curve
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from .losses import CenterLoss


@dataclass
class EvaluationResult:
    """Aggregated metrics from one evaluation pass."""

    loss: float
    accuracy: float
    f1: float
    eer: float


def calculate_equal_error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list[float]]:
    """Compute one-vs-rest equal error rate across classes."""

    classes = np.unique(np.concatenate([y_true, y_pred]))
    class_to_index = {label: index for index, label in enumerate(classes)}
    true_indices = np.vectorize(class_to_index.get)(y_true)
    pred_indices = np.vectorize(class_to_index.get)(y_pred)
    true_one_hot = np.eye(len(classes))[true_indices]
    pred_one_hot = np.eye(len(classes))[pred_indices]

    class_eer: list[float] = []
    for class_index in range(len(classes)):
        fpr, tpr, _ = roc_curve(true_one_hot[:, class_index], pred_one_hot[:, class_index])
        fnr = 1 - tpr
        min_index = int(np.nanargmin(np.abs(fnr - fpr)))
        class_eer.append(float(np.mean([fpr[min_index], fnr[min_index]])))
    return float(np.mean(class_eer)), class_eer


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> EvaluationResult:
    """Evaluate a standard classifier."""

    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    total_samples = len(dataloader.dataset)
    total_correct = 0.0
    total_loss = 0.0
    predictions: list[int] = []
    references: list[int] = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.no_grad():
            _, logits = model(batch_x)
            predicted = torch.argmax(logits, dim=1)
            loss = criterion(logits, batch_y.long())
        total_correct += float((predicted == batch_y).sum())
        total_loss += float(loss.item())
        predictions.extend(predicted.detach().cpu().tolist())
        references.extend(batch_y.detach().cpu().tolist())

    f1 = float(f1_score(np.asarray(references), np.asarray(predictions), average="weighted"))
    eer, _ = calculate_equal_error_rate(np.asarray(references), np.asarray(predictions))
    return EvaluationResult(
        loss=total_loss / len(dataloader),
        accuracy=total_correct / total_samples,
        f1=f1,
        eer=eer,
    )


def evaluate_center_loss(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    config: ExperimentConfig,
) -> EvaluationResult:
    """Evaluate a model trained with center loss."""

    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    dummy = torch.randn(config.batch_size, 1, config.channels, config.sample_points, device=device)
    feature_dim = model._get_fea_dim(dummy)[-1]
    center_loss = CenterLoss(num_classes=config.n_classes, feat_dim=feature_dim, use_gpu=device.type == "cuda")

    total_samples = len(dataloader.dataset)
    total_correct = 0.0
    total_loss = 0.0
    predictions: list[int] = []
    references: list[int] = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.no_grad():
            features, logits = model(batch_x)
            predicted = torch.argmax(logits, dim=1)
            loss = criterion(logits, batch_y.long()) + config.alpha * center_loss(features, batch_y.long())
        total_correct += float((predicted == batch_y).sum())
        total_loss += float(loss.item())
        predictions.extend(predicted.detach().cpu().tolist())
        references.extend(batch_y.detach().cpu().tolist())

    f1 = float(f1_score(np.asarray(references), np.asarray(predictions), average="weighted"))
    eer, _ = calculate_equal_error_rate(np.asarray(references), np.asarray(predictions))
    return EvaluationResult(
        loss=total_loss / len(dataloader),
        accuracy=total_correct / total_samples,
        f1=f1,
        eer=eer,
    )

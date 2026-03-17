"""Experiment training orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from ..config import ExperimentConfig
from ..data.loaders import load_experiment_data
from ..models.registry import create_model
from ..utils import ensure_directory, resolve_device, save_frame, save_json, set_seed
from .evaluation import EvaluationResult, evaluate, evaluate_center_loss
from .losses import CenterLoss


@dataclass
class SeedRunResult:
    """Metrics collected for one random seed."""

    seed: int
    accuracy: float
    f1: float
    eer: float


def _checkpoint_path(config: ExperimentConfig, seed: int) -> Path:
    """Build the checkpoint path for one seed."""

    dataset_name = config.dataset or "-".join(config.cross_tasks or ["unknown"])
    return ensure_directory(config.paths.checkpoints_dir() / dataset_name) / f"{config.model}_{seed}.pth"


def _report_path(config: ExperimentConfig) -> Path:
    """Build the report CSV path."""

    dataset_name = config.dataset or "-".join(config.cross_tasks or ["unknown"])
    return ensure_directory(config.paths.reports_dir() / dataset_name) / f"{config.run_name()}.csv"


def _train_standard(
    config: ExperimentConfig,
    seed: int,
    logger: logging.Logger,
) -> tuple[nn.Module, EvaluationResult]:
    """Train a model with standard cross-entropy loss."""

    device = resolve_device(config.device_index)
    loaders = load_experiment_data(config, seed)
    model = create_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)

    best_epoch = 0
    best_accuracy = -1.0
    checkpoint_path = _checkpoint_path(config, seed)

    for epoch in range(config.epochs):
        model.train()
        train_correct = 0.0
        epoch_loss = 0.0
        for batch_x, batch_y in loaders.train:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            _, logits = model(batch_x)
            loss = criterion(logits, batch_y.long())
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            train_correct += float((torch.argmax(logits, dim=1) == batch_y).sum())

        validation = evaluate(model, loaders.validation, device)
        if epoch - best_epoch > config.early_stop:
            break
        if validation.accuracy > best_accuracy:
            best_accuracy = validation.accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        if (epoch + 1) % 10 == 0:
            train_accuracy = train_correct / len(loaders.train.dataset)
            logger.info(
                "epoch=%s train_acc=%.4f val_acc=%.4f train_loss=%.6f val_loss=%.6f",
                epoch + 1,
                train_accuracy,
                validation.accuracy,
                epoch_loss / len(loaders.train),
                validation.loss,
            )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_result = evaluate(model, loaders.test, device)
    logger.info(
        "seed=%s finished best_val_acc=%.4f test_acc=%.4f test_f1=%.4f test_eer=%.4f",
        seed,
        best_accuracy,
        test_result.accuracy,
        test_result.f1,
        test_result.eer,
    )
    return model, test_result


def _train_with_center_loss(
    config: ExperimentConfig,
    seed: int,
    logger: logging.Logger,
) -> tuple[nn.Module, EvaluationResult]:
    """Train a model with additional center loss."""

    device = resolve_device(config.device_index)
    loaders = load_experiment_data(config, seed)
    model = create_model(config).to(device)
    dummy = torch.randn(config.batch_size, 1, config.channels, config.sample_points, device=device)
    feature_dim = model._get_fea_dim(dummy)[-1]
    criterion = nn.CrossEntropyLoss().to(device)
    center_loss = CenterLoss(num_classes=config.n_classes, feat_dim=feature_dim, use_gpu=device.type == "cuda")
    parameters = list(model.parameters()) + list(center_loss.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)

    best_epoch = 0
    best_accuracy = -1.0
    checkpoint_path = _checkpoint_path(config, seed)

    for epoch in range(config.epochs):
        model.train()
        train_correct = 0.0
        epoch_loss = 0.0
        for batch_x, batch_y in loaders.train:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            features, logits = model(batch_x)
            loss = criterion(logits, batch_y.long()) + config.alpha * center_loss(features, batch_y.long())
            loss.backward()
            for parameter in center_loss.parameters():
                if parameter.grad is not None:
                    parameter.grad.data *= 1.0 / config.alpha
            optimizer.step()
            epoch_loss += float(loss.item())
            train_correct += float((torch.argmax(logits, dim=1) == batch_y).sum())

        validation = evaluate_center_loss(model, loaders.validation, device, config)
        if epoch - best_epoch > config.early_stop:
            break
        if validation.accuracy > best_accuracy:
            best_accuracy = validation.accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        if (epoch + 1) % 10 == 0:
            train_accuracy = train_correct / len(loaders.train.dataset)
            logger.info(
                "epoch=%s train_acc=%.4f val_acc=%.4f train_loss=%.6f val_loss=%.6f",
                epoch + 1,
                train_accuracy,
                validation.accuracy,
                epoch_loss / len(loaders.train),
                validation.loss,
            )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_result = evaluate_center_loss(model, loaders.test, device, config)
    logger.info(
        "seed=%s finished best_val_acc=%.4f test_acc=%.4f test_f1=%.4f test_eer=%.4f",
        seed,
        best_accuracy,
        test_result.accuracy,
        test_result.f1,
        test_result.eer,
    )
    return model, test_result


def run_experiment(config: ExperimentConfig, logger: logging.Logger) -> pd.DataFrame:
    """Run one full multi-seed experiment and persist the summary."""

    results: list[SeedRunResult] = []
    start_time = time.time()
    logger.info("starting run=%s", config.run_name())

    for seed in config.seeds:
        logger.info("running seed=%s", seed)
        set_seed(seed)
        if config.use_center_loss:
            _, test_result = _train_with_center_loss(config, seed, logger)
        else:
            _, test_result = _train_standard(config, seed, logger)
        results.append(
            SeedRunResult(
                seed=seed,
                accuracy=test_result.accuracy,
                f1=test_result.f1,
                eer=test_result.eer,
            )
        )

    frame = pd.DataFrame(
        [{"seed": item.seed, "accuracy": item.accuracy, "f1": item.f1, "eer": item.eer} for item in results]
    ).set_index("seed")
    frame.loc["mean"] = frame.mean(axis=0)
    frame.loc["std"] = frame.std(axis=0)
    save_frame(frame, _report_path(config))
    save_json(
        {
            "run_name": config.run_name(),
            "mode": config.mode,
            "dataset": config.dataset,
            "cross_tasks": config.cross_tasks,
            "model": config.model,
            "elapsed_seconds": round(time.time() - start_time, 4),
        },
        config.paths.reports_dir() / f"{config.run_name()}.json",
    )
    logger.info("finished run=%s elapsed=%.4fs", config.run_name(), time.time() - start_time)
    return frame

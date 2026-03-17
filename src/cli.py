"""Command-line interface for BrainprintNet experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from . import __version__
from .catalog import SUPPORTED_DATASETS, SUPPORTED_MODELS
from .logging_config import configure_logging

if TYPE_CHECKING:
    from .config import ExperimentConfig

PACKAGE_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def _parse_name_list(raw_values: list[str] | None) -> list[str] | None:
    """Parse repeated or comma-separated CLI values into a clean string list."""

    if not raw_values:
        return None
    values: list[str] = []
    for raw_value in raw_values:
        values.extend(item.strip() for item in raw_value.split(",") if item.strip())
    return values or None


def build_parser() -> argparse.ArgumentParser:
    """Build the BrainprintNet CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="brainprintnet",
        description="Train and evaluate EEG user-identification models with the BrainprintNet package.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "within-session", "cross-task"],
        default="baseline",
        help="Experiment mode to run.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset name used by baseline or within-session experiments.",
    )
    parser.add_argument(
        "--cross-tasks",
        nargs="+",
        default=None,
        help="Two dataset names used by cross-task experiments. Supports spaces or commas.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ResEEGNet",
        help="Model architecture to train.",
    )
    parser.add_argument("--device-index", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=300, help="Maximum number of epochs.")
    parser.add_argument("--early-stop", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2024, 2025, 2026, 2027, 2028],
        help="One or more random seeds.",
    )
    parser.add_argument("--alpha", type=float, default=1e-3, help="Center-loss weighting factor.")
    parser.add_argument(
        "--center-loss",
        action="store_true",
        help="Enable additional center loss during training.",
    )
    parser.add_argument("--res-blocks", type=int, default=5, help="Residual block count for GWNet-style models.")
    parser.add_argument("--stride-factor", type=int, default=5, help="Stride factor used by multi-scale models.")
    parser.add_argument(
        "--aug-type",
        type=str,
        default=None,
        help="Optional augmentation name such as channel_mixure or channel_noise.",
    )
    parser.add_argument(
        "--session-num",
        type=int,
        default=1,
        help="Session selector used by within-session or cross-task variants.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory that stores EEG datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PACKAGE_OUTPUT_ROOT,
        help="Root directory used for checkpoints, logs, and reports. Defaults to src/outputs.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print all supported dataset names and exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print all supported model names and exit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _build_config(arguments: argparse.Namespace) -> "ExperimentConfig":
    """Create a strongly typed experiment configuration from parsed arguments."""

    from .config import ExperimentConfig, PathsConfig

    cross_tasks = _parse_name_list(arguments.cross_tasks)
    if arguments.mode == "cross-task":
        if not cross_tasks or len(cross_tasks) != 2:
            raise ValueError("Cross-task experiments require exactly two values in `--cross-tasks`.")
        dataset = None
    else:
        if arguments.dataset is None:
            raise ValueError("`--dataset` is required unless `--mode cross-task` is used.")
        dataset = arguments.dataset

    return ExperimentConfig(
        mode=arguments.mode,
        dataset=dataset,
        model=arguments.model,
        device_index=arguments.device_index,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        early_stop=arguments.early_stop,
        learning_rate=arguments.learning_rate,
        seeds=arguments.seeds,
        alpha=arguments.alpha,
        use_center_loss=arguments.center_loss,
        res_blocks=arguments.res_blocks,
        stride_factor=arguments.stride_factor,
        aug_type=arguments.aug_type,
        session_num=arguments.session_num,
        cross_tasks=cross_tasks,
        paths=PathsConfig(
            data_root=arguments.data_root,
            output_root=arguments.output_root,
        ),
    )


def _print_supported_items(items: Sequence[str], title: str) -> None:
    """Print a compact title plus one item per line."""

    print(title)
    for item in items:
        print(f"- {item}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the BrainprintNet command-line entry point."""

    parser = build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    if arguments.list_datasets:
        _print_supported_items(SUPPORTED_DATASETS, "Supported datasets:")
        return 0
    if arguments.list_models:
        _print_supported_items(SUPPORTED_MODELS, "Supported models:")
        return 0

    if arguments.model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {arguments.model}")

    from .training import run_experiment

    config = _build_config(arguments)
    logger = configure_logging(config.paths.logs_dir() / f"{config.run_name()}.log")
    logger.info("package_version=%s", __version__)
    logger.info("resolved_config=%s", config)
    run_experiment(config, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

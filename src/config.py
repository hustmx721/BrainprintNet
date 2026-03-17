"""Configuration objects and dataset metadata for BrainprintNet."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from .utils import ensure_directory

RunMode = Literal["baseline", "within-session", "cross-task"]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs"


@dataclass(frozen=True)
class DatasetMeta:
    """Static metadata required to instantiate a model for one dataset."""

    name: str
    n_classes: int
    channels: int
    sampling_rate: int
    duration_seconds: float


OPENBMI_DATASETS = {"Rest", "MI", "ERP", "SSVEP", "CatERP"}
M3CV_DATASETS = {
    "M3CV_Rest",
    "M3CV_Transient",
    "M3CV_Steady",
    "M3CV_P300",
    "M3CV_Motor",
    "M3CV_SSVEP_SA",
}

DATASET_METADATA: dict[str, DatasetMeta] = {
    "001": DatasetMeta("001", 9, 22, 250, 4.0),
    "004": DatasetMeta("004", 9, 3, 250, 4.0),
    "BCI85": DatasetMeta("BCI85", 85, 27, 250, 5.0),
    "Rest85": DatasetMeta("Rest85", 85, 27, 250, 5.0),
    "LJ30": DatasetMeta("LJ30", 30, 20, 300, 5.0),
    "Rest": DatasetMeta("Rest", 54, 62, 250, 4.0),
    "MI": DatasetMeta("MI", 54, 62, 250, 4.0),
    "ERP": DatasetMeta("ERP", 54, 62, 1000, 0.8),
    "SSVEP": DatasetMeta("SSVEP", 54, 62, 250, 4.0),
    "CatERP": DatasetMeta("CatERP", 54, 62, 250, 4.0),
    "M3CV_Rest": DatasetMeta("M3CV_Rest", 20, 64, 250, 4.0),
    "M3CV_Transient": DatasetMeta("M3CV_Transient", 20, 64, 250, 4.0),
    "M3CV_Steady": DatasetMeta("M3CV_Steady", 20, 64, 250, 4.0),
    "M3CV_P300": DatasetMeta("M3CV_P300", 20, 64, 250, 4.0),
    "M3CV_Motor": DatasetMeta("M3CV_Motor", 20, 64, 250, 4.0),
    "M3CV_SSVEP_SA": DatasetMeta("M3CV_SSVEP_SA", 20, 64, 250, 4.0),
    "SEED": DatasetMeta("SEED", 15, 62, 200, 4.0),
}


@dataclass
class PathsConfig:
    """Filesystem locations used by training runs."""

    data_root: Path = Path("data")
    output_root: Path = DEFAULT_OUTPUT_ROOT

    def dataset_dir(self) -> Path:
        """Return the top-level dataset directory."""

        return ensure_directory(self.data_root)

    def logs_dir(self) -> Path:
        """Return the logging output directory."""

        return ensure_directory(self.output_root / "logs")

    def checkpoints_dir(self) -> Path:
        """Return the checkpoint output directory."""

        return ensure_directory(self.output_root / "checkpoints")

    def reports_dir(self) -> Path:
        """Return the metrics/report output directory."""

        return ensure_directory(self.output_root / "reports")


@dataclass
class ExperimentConfig:
    """Runtime configuration for one BrainprintNet experiment."""

    mode: RunMode = "baseline"
    dataset: Optional[str] = None
    model: str = "ResEEGNet"
    device_index: int = 0
    batch_size: int = 64
    epochs: int = 300
    early_stop: int = 30
    learning_rate: float = 1e-3
    seeds: list[int] = field(default_factory=lambda: [2024, 2025, 2026, 2027, 2028])
    alpha: float = 1e-3
    use_center_loss: bool = False
    res_blocks: int = 5
    stride_factor: int = 5
    aug_type: Optional[str] = None
    session_num: Optional[int] = None
    cross_tasks: Optional[list[str]] = None
    paths: PathsConfig = field(default_factory=PathsConfig)

    @property
    def dataset_meta(self) -> DatasetMeta:
        """Return resolved dataset metadata for the current experiment."""

        dataset_name: Optional[str]
        if self.dataset is not None:
            dataset_name = self.dataset
        elif self.cross_tasks:
            dataset_name = self.cross_tasks[0]
        else:
            dataset_name = None

        if dataset_name is None or dataset_name not in DATASET_METADATA:
            raise ValueError("A valid dataset name is required to resolve metadata.")
        return DATASET_METADATA[dataset_name]

    @property
    def n_classes(self) -> int:
        """Return the number of classes for the current experiment."""

        return self.dataset_meta.n_classes

    @property
    def channels(self) -> int:
        """Return the EEG channel count for the current experiment."""

        return self.dataset_meta.channels

    @property
    def sampling_rate(self) -> int:
        """Return the sampling rate for the current experiment."""

        return self.dataset_meta.sampling_rate

    @property
    def duration_seconds(self) -> float:
        """Return the trial duration in seconds for the current experiment."""

        return self.dataset_meta.duration_seconds

    @property
    def sample_points(self) -> int:
        """Return the number of time points per trial."""

        return int(self.sampling_rate * self.duration_seconds)

    def run_name(self) -> str:
        """Build a stable run name for logs and output artefacts."""

        dataset_name = self.dataset or "-".join(self.cross_tasks or ["unknown"])
        return f"{self.mode}_{dataset_name}_{self.model}"

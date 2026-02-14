from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any
import re

@dataclass
class DatasetConfig:
    """Configuration object for the dataset processing pipeline.

    Attributes:
        name (str): Unique identifier for this configuration.
        VAR_NAME (str): Name of the variable in scan files to read.
        DEBUG (bool): If True, runs in debug mode (verbose output). Defaults to True.
        RAW_DATA_PATH (Path): Directory containing the raw NetCDF (.nc) files. Defaults to "./data/raw/".
        PROCESSED_PATH (Path): Output directory for processed images and patches. Defaults to "./data/processed/".
        RATIO (tuple[float, float, float]): Data split ratios for (Train, Validation, Test). Defaults to (0.75, 0.15, 0.1).
        SEED (int): Random seed for reproducibility in splitting. Defaults to 42.
        MIN_VAL (int): Minimum intensity value for clipping (lower bound). Defaults to 10750.
        MAX_VAL (int): Maximum intensity value for clipping (upper bound). Defaults to 21800.
        EXTRACT_PATCHES (bool): Whether to extract patches from full slices. Defaults to True.
        PATCH_SIZE (int): Spatial dimension (Height/Width) of square patches. Defaults to 256.
        OVERLAP (float): Overlap ratio between adjacent patches (0.0 to 1.0). Defaults to 0.2.
        VAR_THRESHOLD (float): Minimum variance required to save a patch (filters empty background). Defaults to 0.0.
    """
    name: str
    VAR_NAME: str = 'tomo'
    DEBUG: bool = True
    RAW_DATA_PATH: Path = Path("./data/raw/")
    PROCESSED_PATH: Path = Path("./data/processed/")
    RATIO: tuple[float, float, float] = (0.75, 0.15, 0.1)
    SEED: int = 42
    MIN_VAL: int = 10750
    MAX_VAL: int = 21800
    EXTRACT_PATCHES: bool = True
    PATCH_SIZE: int = 256
    OVERLAP: float = 0.2
    VAR_THRESHOLD: float = 0.0

default_dataset_cfg = DatasetConfig(name="default")


@dataclass
class Experiment:
    """Metadata for the experiment run."""
    name: str = "New_Experiment"
    description: str = ""
    tags: list[str] = field(default_factory=list)

@dataclass
class Logs:
    """Configuration for the UnifiedLogger. [src.utils.logger.UnifiedLoggger]"""
    name: str = "Logger"
    use_tensorboard: bool = False
    use_wandb: bool = False
    log_interval: int = 10

@dataclass
class Data:
    """Data configuration parameters for training."""
    train_dir: Path = Path("./data/processed/train/images")
    val_dir: Path = Path("./data/processed/val/images")

    batch_size: int = 4
    num_workers: int = 4

    # Contains mode, transform_params, noise_params (specific to model)
    preprocess_params: dict = field(default_factory=dict)

@dataclass
class Model:
    """Configuration for the model architecture."""
    name: str = ""
    model_params: dict = field(default_factory=dict)

@dataclass
class Optimizer:
    """Wrapper for different optimizers."""
    label: str = "main"
    name: str = "Adam"
    params: dict[str, Any] = field(default_factory=dict)

@dataclass
class Scheduler:
    """Wrapper for different LR schedulers."""
    label: str = "main"
    name: str = "ReduceLROnPlate"
    params: dict[str, Any] = field(default_factory=dict)

@dataclass
class Train:
    """Hyperparameters for the training loop."""
    max_epochs: int = 100
    use_amp: bool = False
    save_interval: int = -1     # No save.
    resume_checkpoint: str = "" # Example: "./experiment/New_Experiment/checkpoints/latest.pth"
    kwargs: dict[str, Any] = field(default_factory=dict)
    monitor_metrics: dict[str, str] = field(default_factory=lambda : {"val_loss": "min"})
    optimizers: list[Optimizer] = field(default_factory=list[Optimizer])
    schedulers: list[Scheduler] = field(default_factory=list[Scheduler])

@dataclass
class Config:
    """Master configuration object for the project."""
    project_name: str = "New_Project"
    root_dir: Path = Path("./experiments")
    seed: int = 42
    use_gpu: bool = False
    run_dir: Path = field(init=False)

    experiment: Experiment = field(default_factory=Experiment)
    logs: Logs = field(default_factory=Logs)
    data: Data = field(default_factory=Data)
    models: list[Model] = field(default_factory=list)
    train: Train = field(default_factory=Train)

    def __post_init__(self):
        self._setup_directories()

    def _setup_directories(self):
        """Automatically resolves the base directory structure for the run."""

        if self.train.resume_checkpoint:
            ckpt_path = Path(self.train.resume_checkpoint)
            
            # Navigate up two levels: latest.pth -> checkpoints -> exp-00-name
            self.run_dir = ckpt_path.parent.parent 
            
            if not self.run_dir.exists():
                print(f"⚠️ Warning: Resumed run_dir {self.run_dir} does not exist. Creating it.")
                self.run_dir.mkdir(parents=True, exist_ok=True)
                
            # Exit early so we don't trigger the auto-increment logic!
            return

        base_path = Path(self.root_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        # Auto-Increment Logic: Scan for existing folders with prefix as the experiment's name.
        next_id = 0
        existing_exps = [p.name for p in base_path.iterdir() if p.is_dir() and p.name.startswith(f"{self.experiment.name}-")]
        
        if existing_exps:
            ids = []
            for name in existing_exps:
                # Regex: match "name-ID" to extract the ID
                match = re.search(rf"{re.escape(self.experiment.name)}-(\d+)", name)
                if match:
                    ids.append(int(match.group(1)))
            if ids:
                next_id = max(ids) + 1

        # Format Run Name: exp-{N:02d}-{name}-{tags}_{date}
        timestamp = datetime.now().strftime("%Y-%m-%d")
        tags_str = "-".join(self.experiment.tags) if self.experiment.tags else ""
        
        components = [self.experiment.name, f"{next_id:02d}"]
        if tags_str: components.append(tags_str)
        components.append(timestamp)
        
        run_name = "-".join(c for c in components if c)
        
        # Define Paths
        self.run_dir = base_path / run_name

        # Create directories
        self.run_dir.mkdir(exist_ok=True)

    
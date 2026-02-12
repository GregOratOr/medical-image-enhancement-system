from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
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
class ExperimentConfig:
    """
    Metadata for the experiment run.
    """
    project_name: str = "noise2noise-medical"
    base_dir: str = "./experiments"
    name: str = "exp"
    description: str = ""
    tags: list[str] = field(default_factory=list)

@dataclass
class LogConfig:
    """
    Configuration for the UnifiedLogger.
    Matches arguments in src.utils.logger.UnifiedLogger.
    """
    # Backend Toggles
    use_tensorboard: bool = True           # Default: Enabled
    use_wandb: bool = False                # Default: Disabled
    
    # Frequency
    log_interval: int = 10                 # Log metrics every N steps

@dataclass
class TrainDataConfig:
    """Configuration for the CTScans Dataset during training."""
    # Base Paths
    train_dir: str = "./data/processed/train"
    val_dir: str = "./data/processed/val"

    # Subfolder Selection ('patches' or 'images')
    data_source: str = "patches"

    # Loader Settings
    batch_size: int = 4
    num_workers: int = 4
    mode: str = "n2n"             # 'n2n' or 'n2c'
    
    # Clean Image Transformations
    transform_params: dict[str, Any] = field(default_factory=dict)

    # Noise Configuration
    noise_ops: list[dict[str, Any]] = field(default_factory=list)

@dataclass
class ModelConfig:
    """Configuration for the Noise2Noise U-Net."""
    architecture: str = "unet"     # 'unet' (dynamic) or 'original' (fixed)
    in_channels: int = 1          
    out_channels: int = 1         
    base_channels: int = 48       
    depth: int = 4                
    activation: str = "leaky_relu"

@dataclass
class TrainConfig:
    """Hyperparameters for the Training Loop."""
    # Optimization
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.0
    accum_steps: int = 1          
    
    # Loss Logic
    loss_alpha: float = 0.5       
    
    # Performance
    use_amp: bool = False         
    
    # Checkpointing
    save_interval: int = 0           
    
    # Metrics monitored during validation step.
    monitor_metrics: dict[str, str] = field(default_factory=lambda: {
        "val_loss": "min",
        # "psnr": "max",   # Peak Signal-to-Noise Ratio (Higher is better)
        # "ssim": "max"    # Structural Similarity (Higher is better)
    })

    # Optimizers. Default: Adam
    optimizers: dict[str, dict[str, Any]] = field(default_factory=lambda: {'default': {
        "type": "adam", 
        "params": {}
    }})
    
    # LR Schedulers. Default: ReduceLROnPlateau
    schedulers: dict[str, dict[str, Any]] = field(default_factory=lambda: { 'default': {
        "type": "plateau", 
        "params": {"patience": 10, "factor": 0.5, "mode": "min"}
    }})

@dataclass
class Config:
    """The Master Configuration Object."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    log: LogConfig = field(default_factory=LogConfig)
    data: TrainDataConfig = field(default_factory=TrainDataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    seed: int = 42
    device: str = "cpu"

    run_dir: Path = field(init=False)

    def __post_init__(self):
        """Automatically resolves the directory structure on initialization."""
        self._setup_directories()

    def _setup_directories(self):
        """Automatically resolves the directory structure on initialization."""
        base_path = Path(self.experiment.base_dir)
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

        # 4. Create directories
        self.run_dir.mkdir(exist_ok=True)
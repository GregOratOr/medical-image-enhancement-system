from pathlib import Path
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    """Configuration object for the dataset processing pipeline.

    Attributes:
        name (str): Unique identifier for this configuration.
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

default_cfg = DatasetConfig(name="default")

config1 = DatasetConfig(
    name="Test Config 1",
    DEBUG = True,
    RAW_DATA_PATH = Path("./data/raw/"),
    PROCESSED_PATH = Path("./data/processed/"),
    RATIO = (0.8, 0.1, 0.1),
    SEED = 42,
    MIN_VAL = 10750,
    MAX_VAL = 21800,
    EXTRACT_PATCHES = True,
    PATCH_SIZE = 512,
    OVERLAP = 0.2,
    VAR_THRESHOLD = 0.0
)


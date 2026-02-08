from pathlib import Path
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    '''
    Dataset Config dataclass.
    '''
    name:str
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


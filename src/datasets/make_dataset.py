import netCDF4 as nc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

from configs.dataset_config import DatasetConfig, default_cfg
from src.utils.image_operations import process_slice
from src.datasets.preparation import split_dataset_blocks, image_to_patches

@dataclass
class PipelineConfig:
    processed_path: Path
    split_name: str = "train"
    extract_patches: bool = False
    patch_size: int = 256
    overlap: float = 0.2
    threshold: float = 0.0
    clip_range: tuple[int, int] | None = None
    
    @property
    def images_dir(self) -> Path:
        return self.processed_path / self.split_name / "images"
    
    @property
    def patches_dir(self) -> Path:
        return self.processed_path / self.split_name / "patches"

    def create_dirs(self):
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if self.extract_patches:
            self.patches_dir.mkdir(parents=True, exist_ok=True)


def process_block(block_path: Path, p_config: PipelineConfig, global_slice_idx: int) -> int:
    '''
    Reads a block, saves full images, and optionally extracts patches.
    
    :param block_path: Path of the current block being processed.
    :type block_path: Path
    :param config: Contains parameters for the pipeline dataflow.
    :type config: PipelineConfig
    :param global_slice_idx: Global index of the current image being processed.
    :type global_slice_idx: int
    :return: Returns the updated global index of the next image in the block.
    :rtype: int

    '''
    try:
        with nc.Dataset(block_path, 'r') as block:
            scan_data = block.variables['tomo'][:] # Load volume
            
            # Dimensionality Check
            if scan_data.ndim != 3:
                raise ValueError(f"Expected 3D volume, got {scan_data.shape}")
                
            Z, _, _ = scan_data.shape

            for i in tqdm(range(Z), desc=f"  Processing Slice [{block_path.name}]", leave=False, position=1):
                # 1. Pure Processing (Utils)
                raw_slice = scan_data[i, :, :]
                img = process_slice(raw_slice, clip_range=p_config.clip_range)
                
                # 2. Save Full Image (Orchestration)
                img.save(p_config.images_dir / f"{global_slice_idx:06d}.png")
                
                # 3. Patching (Dataset Prep)
                if p_config.extract_patches:
                    image_to_patches(
                        save_dir=p_config.patches_dir,
                        image=img,
                        image_idx=global_slice_idx,
                        patch_size=p_config.patch_size,
                        overlap=p_config.overlap,
                        threshold=p_config.threshold
                    )
                
                global_slice_idx += 1

    except OSError as ose:
        print(f"❌ Error reading file: {block_path.name}: {ose}")
    except Exception as e:
        print(f"❌ Exception occurred while processing {block_path.name}")
        print(f"More Info: \n{e}")
        
    return global_slice_idx


def prepare_dataset(cfg: DatasetConfig) -> None:
    print("🚀 Starting Data Pipeline ...")
    
    # 1. Split Data
    splits = split_dataset_blocks(cfg.RAW_DATA_PATH, ratio=cfg.RATIO, seed=cfg.SEED)
    
    # 2. Iterate Splits
    pbar = tqdm(splits.items(), leave=False, position=0)
    for split_name, block_paths in pbar:
        pbar.set_description(f"📂 Processing '{split_name}' set({len(block_paths)} blocks)")

        # Setup Config for this split
        p_config = PipelineConfig(
            processed_path=cfg.PROCESSED_PATH,
            split_name=split_name,
            extract_patches=cfg.EXTRACT_PATCHES,
            patch_size=cfg.PATCH_SIZE,
            overlap=cfg.OVERLAP,
            threshold=cfg.VAR_THRESHOLD,
            clip_range=(cfg.MIN_VAL, cfg.MAX_VAL)
        )
        p_config.create_dirs()
        
        global_idx = 0
        for block_path in block_paths:
            global_idx = process_block(block_path, p_config, global_idx)
            
    print("\n✅ Done!")


def main():
    prepare_dataset(default_cfg)


if __name__ == "__main__":
    main()

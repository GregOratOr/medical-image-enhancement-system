import random
import numpy as np
from pathlib import Path
from PIL import Image

def split_dataset_blocks(data_path: Path, ratio: tuple[float, float, float]=(0.8, 0.1, 0.1), seed: int=42) -> dict[str, list]:
    """Splits the dataset into training, validation and test sets using the given ratio and seed.
    
    Ensures deterministic splitting via a fixed seed and verifies no set leakage.

    Args:
        data_path (Path): Path to the dataset directory.
        ratio (tuple[float, float, float], optional): Ratio of the training, validation and test sets. 
            Defaults to (0.8, 0.1, 0.1).
        seed (int, optional): Seed value for deterministic randomness. Defaults to 42.

    Returns:
        dict[str, list]: A dictionary with the set category ('train', 'val', 'test') as keys 
        and the list of file paths as values.

    Raises:
        AssertionError: If the sum of ratios is not 1.0 or if leakage is detected.
        ValueError: If no .nc files are found or if the dataset is too small.
    """
    assert abs(sum(ratio) - 1.0) < 1e-6, "Sum of ratios must be 1."

    file_paths = sorted(list(data_path.glob("block*.nc")))
    n_total = len(file_paths)
    
    if n_total == 0:
        raise ValueError(f"No .nc files found at {data_path}")

    # Calculate counts
    n_val = int(n_total * ratio[1])
    n_test = int(n_total * ratio[2])

    # Safety for small datasets
    if n_val == 0 and ratio[1] > 0.0: n_val = 1
    if n_test == 0 and ratio[2] > 0.0: n_test = 1

    n_train = n_total - n_val - n_test

    if ratio[0] > 0 and n_train < 1:
        raise ValueError(f"Dataset too small [{ n_total }] for this split ratio [{ ratio }].")

    # Shuffle
    random.seed(seed)
    random.shuffle(file_paths)

    # Slice
    train_paths = file_paths[:n_train]
    val_paths = file_paths[n_train:n_train + n_val]
    test_paths = file_paths[n_train + n_val:]

    # Verify Leakage
    assert len(set(train_paths) & set(val_paths)) == 0, "Leakage detected! Training and Validation sets have data leakage."

    return {'train': train_paths, 'val': val_paths, 'test': test_paths}


def image_to_patches(save_dir: Path, image: Image.Image, image_idx: int, patch_size: int=256, overlap: float=0.2, threshold: float=0.0) -> None:
    """Slices a PIL image into patches and saves them to disk.

    Args:
        save_dir (Path): Path of the location to save the patches.
        image (Image.Image): Image to be converted to patches.
        image_idx (int): Global index of the image.
        patch_size (int, optional): Size of the square patches. Defaults to 256.
        overlap (float, optional): Fraction of overlap between the two patches. Defaults to 0.2.
        threshold (float, optional): The variance threshold for the image values to be accepted 
            as a useful patch. Defaults to 0.0.
    """
    img_arr = np.array(image)
    h, w = img_arr.shape[:2]
    step = int(patch_size * (1 - overlap))

    # Calculate coordinates
    y_coords = list(range(0, h - patch_size + 1, step))
    if (h - patch_size) % step != 0: y_coords.append(h - patch_size)

    x_coords = list(range(0, w - patch_size + 1, step))
    if (w - patch_size) % step != 0: x_coords.append(w - patch_size)

    patch_idx = 0
    for y in y_coords:
        for x in x_coords:
            patch = img_arr[y : y + patch_size, x : x + patch_size]

            # Filter boring patches (background)
            if threshold > 0 and np.var(patch) < threshold:
                continue

            # Save
            filename = f"{image_idx:06d}_{patch_idx:04d}.png"
            Image.fromarray(patch).save(save_dir / filename)
            patch_idx += 1

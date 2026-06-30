import numpy as np
from PIL import Image


def normalize_image(image: np.ndarray, clip_range: tuple[int, int] | None) -> np.ndarray:
    """Normalizes raw scan data to [0, 255] uint8 with input clipping.

    Args:
        image (np.ndarray): Image to be normalized.
        clip_range (tuple[int, int] | None): Range of raw scan values (min, max) to consider. 
            If None, clipping uses the image's min/max.

    Returns:
        np.ndarray: The normalized image as a NumPy uint8 ndarray.
    """
    
    if clip_range is None:
        clip_range = (np.min(image), np.max(image))

    # 1. Cast directly to float to preserve negative values (e.g. -1000 HU)
    image = image.astype(np.float64)

    # 2. Clip values to the Domain Specific Range (Windowing)
    image = np.clip(image, clip_range[0], clip_range[1])
    
    # 3. Min-Max Normalization
    range_val = float(clip_range[1]) - float(clip_range[0])
    if range_val == 0:
        return np.zeros_like(image, dtype=np.uint8)
        
    image = ((image - clip_range[0]) / range_val) * 255.0
    
    # 4. Final Cast to uint8
    return image.astype(np.uint8)


def process_slice(slice_data: np.ndarray, clip_range: tuple[int, int] | None = None) -> Image.Image:
    """Converts a raw 2D numpy array into a PIL Image object.
    
    Does NOT save to disk (separation of concerns).

    Args:
        slice_data (np.ndarray): Image values for a single slice of a 3D scan.
        clip_range (tuple[int, int] | None, optional): Range of raw scan values to consider. 
            Defaults to None.

    Returns:
        Image.Image: A PIL Image object representing the processed slice (Mode 'L').
    """
    
    norm_img = normalize_image(slice_data, clip_range=clip_range)
    return Image.fromarray(norm_img, mode='L')
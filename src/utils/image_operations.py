import numpy as np
from PIL import Image


def normalize_image(image: np.ndarray, clip_range:tuple[int, int]|None) -> np.ndarray:
    '''
    Normalizes raw scan data to [0, 255] uint8 with input clipping.
    
    :param image: Image to be normalized.
    :type image: np.ndarray
    :param clip_range: Range of raw scan values to consider. If None, no clipping is performed.
    :type clip_range: tuple[int, int] | None
    :return: Returns the normalized image as a NumPy uint8 ndarray.
    :rtype: ndarray[_AnyShape, dtype[Any]]
    
    '''
    
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

def process_slice(slice_data: np.ndarray, clip_range: tuple[int, int]|None = None) -> Image.Image:
    '''
    Converts a raw 2D numpy array into a PIL Image object.
    Does NOT save to disk (separation of concerns).

    :param slice_data: Image values for a single slice of a 3D scan.
    :type slice_data: np.ndarray
    :param clip_range: Range of raw scan values to consider.
    :type clip_range: tuple[int, int] | None
    :return: Returns an Image object representing the processed slice.
    :rtype: Image
    
    '''
    
    norm_img = normalize_image(slice_data, clip_range=clip_range)
    return Image.fromarray(norm_img, mode='L')
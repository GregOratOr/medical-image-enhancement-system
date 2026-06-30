import base64
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def image_to_base64(image: Image.Image) -> str:
    """ Converts a PIL Image to a Base64 string for embedding in HTML/JS.
    
    Args:
        image: A PIL Image object (Original or Denoised)
        
    Returns:
        str: A full data URI string (e.g., "data:image/png;base64,iVBOR...")
    """

    img_str = base64.b64encode(image_to_bytes(image)).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def image_to_bytes(image: Image.Image) -> bytes:
    """ Converts a PIL Image to Bytes.
    
    Args:
        image: A PIL Image object (Original or Denoised)
        
    Returns:
        bytes: Full image in Byte format.
    """

    buffer = BytesIO()
    # Save as PNG to preserve quality (lossless)
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def figure_to_bytes(figure) -> bytes:
    """ Converts a Matplotlib figure to Bytes.
    
    Args:
        image: A PIL Image object (Original or Denoised)
        
    Returns:
        bytes: Full image in Byte format.
    """

    buffer = BytesIO()
    # Save as PNG to preserve quality (lossless)
    figure.savefig(buffer, format="PNG")
    return buffer.getvalue()

def get_difference_heatmap(orig_img: Image.Image, denoised_img: Image.Image) -> Image.Image:
    """ Calculates the absolute difference between two images and applies 
        an INFERNO colormap using Matplotlib to highlight the removed noise.
    """
    # Convert PIL images to NumPy arrays (Grayscale, Float32 for math)
    arr_orig = np.array(orig_img.convert("L"), dtype=np.float32)
    arr_denoised = np.array(denoised_img.convert("L"), dtype=np.float32)

    # Calculate the absolute difference
    diff = np.abs(arr_orig - arr_denoised)

    # Cleanup
    del arr_orig, arr_denoised

    # Normalize the difference to exactly [0.0, 1.0] for Matplotlib
    max_val = np.max(diff)
    if max_val > 0:
        diff_normalized = diff / max_val
    else:
        diff_normalized = diff
        
    # Fetch the 'inferno' colormap from Matplotlib
    # Inferno maps 0.0 -> Black, 0.5 -> Orange/Red, 1.0 -> Yellow/White
    cmap = plt.get_cmap('inferno')

    # Apply the colormap (this returns an RGBA array with floats from 0.0 to 1.0)
    heatmap_rgba = cmap(diff_normalized)

    # Drop the Alpha channel ([:, :, :3]), scale to 255, and convert to uint8
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    # Convert back to PIL Image
    return Image.fromarray(heatmap_rgb)

def get_pixel_intensity_histogram(orig_img: Image.Image, denoised_img: Image.Image):
    """ Generates a Matplotlib figure comparing the pixel histograms of both images."""

    # Flatten the 2D image arrays into 1D lists of pixels
    arr_orig = np.array(orig_img.convert("L")).flatten()
    arr_denoised = np.array(denoised_img.convert("L")).flatten()

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot both histograms with transparency (alpha)
    ax.hist(arr_orig, bins=256, range=(0, 256), color='cornflowerblue', alpha=0.7, label='Original', histtype='stepfilled')
    ax.hist(arr_denoised, bins=256, range=(0, 256), color='salmon', alpha=0.6, label='Denoised', histtype='stepfilled')
    
    # Styling
    ax.set_title("Pixel Intensity Distribution")
    ax.set_xlabel("Pixel Value (0-Black, 255-White)")
    ax.set_ylabel("Pixel Count")
    ax.legend()
    
    # Clean up layout
    plt.tight_layout()

    return fig
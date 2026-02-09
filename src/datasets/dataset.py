import warnings
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from typing import Callable, Optional

class CTScans(Dataset):
    """A PyTorch Dataset for loading CT scan images and applying noise on-the-fly.

    This dataset reads PNG images from a directory, converts them to grayscale,
    and returns pairs of images for training denoising models depending on its mode.

    Modes:
        - 'n2n': Returns (Noisy_1, Noisy_2, Clean) for self-supervised Noise2Noise.
        - 'n2c': Returns (Noisy, Clean) for standard supervised Noise2Clean.
    """

    def __init__(self, image_dir: Path | str, transform: Optional[Callable]=None, noise_transform: Optional[Callable]=None, mode: str="n2n") -> None:
        """Initializes the CTScans dataset.

        Args:
            image_dir (Path | str): Path to directory containing processed PNG images.
            transform (Callable, optional): Transforms for the underlying clean image 
                (e.g., ToTensor, Normalize, RandomCrop). Defaults to None.
            noise_transform (Callable, optional): Transform that adds synthetic noise. Defaults to None
            mode (str, optional): Training mode ('n2n' or 'n2c'). Defaults to 'n2n'.

        Raises:
            FileNotFoundError: If no .png files are found in the specified directory.
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.noise_transform = noise_transform
        self.mode = mode
        
        # List all PNGs
        self.files = sorted(list(self.image_dir.glob("*.png")))
        
        # Sanity check
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found in {self.image_dir}. ⚠️ Did you run make_dataset.py?")

    def __len__(self) -> int:
        """Returns the total number of images in the dataset.

        Returns:
            int: Number of image files found.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves image pairs based on the selected mode for the given index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            (noisy_img1, noisy_img2, clean_img) if **mode='n2n'**.
            (noisy_img, clean_img) if **mode='n2c'**.
        """
        img_path = self.files[idx]

        # 1. Load Image (Grayscale)
        clean_img = Image.open(img_path).convert("L")
        
        # 2. Apply Transforms (if any)
        if self.transform:
            clean_img = self.transform(clean_img)
        else:
            # Fallback if no transform is provided
            clean_img = T.ToTensor()(clean_img)
        
        # Sanity check.
        assert isinstance(clean_img, torch.Tensor), f"Expected torch.Tensor after transform, got {type(clean_img)}"

        # 3. Generate Noisy Versions
        if self.noise_transform:
            # Generate first noisy version (Input)
            noisy_1 = self.noise_transform(clean_img.clone())
            
            if self.mode == 'n2n':
                # Generate second INDEPENDENT noisy version (Target)
                noisy_2 = self.noise_transform(clean_img.clone())
                
                # Return tuple: (Input, Target, GroundTruth)
                # GroundTruth is strictly for validation metrics (PSNR), NOT loss.
                return noisy_1, noisy_2, clean_img
            
            elif self.mode == 'n2c':
                # Standard training: Input=Noisy, Target=Clean
                return noisy_1, clean_img
        
        # Fallback if no noise transform. Just return (clean, clean) or (clean, clean, clean) depending on mode.
        warnings.warn("No Noise Transforms provides, falling back to clean image tuples")
        return (clean_img, clean_img) if self.mode == 'n2c' else (clean_img, clean_img, clean_img)
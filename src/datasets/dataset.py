import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

class CTScans(Dataset):
    """A PyTorch Dataset for loading CT scan images and applying noise on-the-fly.

    This dataset reads PNG images from a directory, converts them to grayscale,
    and returns pairs of (noisy_image, clean_image) for training denoising models.
    """

    def __init__(self, image_dir: Path | str, transform=None, noise_transform=None) -> None:
        """Initializes the CTScans dataset.

        Args:
            image_dir (Path | str): Path to the directory containing processed PNG images.
            transform (callable, optional): Optional transform to be applied on the clean image 
                (e.g., ToTensor, Normalize, RandomCrop). Defaults to None.
            noise_transform (callable, optional): Optional transform to generate the noisy input 
                from the clean image. Defaults to None.

        Raises:
            FileNotFoundError: If no .png files are found in the specified directory.
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.noise_transform = noise_transform
        
        # 2. List all PNGs
        self.files = sorted(list(self.image_dir.glob("*.png")))
        
        # 3. Sanity check
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found in {self.image_dir}. ⚠️ Did you run make_dataset.py?")

    def __len__(self) -> int:
        """Returns the total number of images in the dataset.

        Returns:
            int: Number of image files found.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves an image pair (noisy, clean) by index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - noisy_img (Tensor): The input image with synthetic noise applied.
                - clean_img (Tensor): The ground truth clean image.
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
        
        noisy_img = clean_img.clone()

        if self.noise_transform:
            noisy_img = self.noise_transform(noisy_img)

        return noisy_img, clean_img
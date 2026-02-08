import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from src.utils.logging_utils import get_noise_name

from configs.dataset_config import default_cfg
from src.datasets.dataset import CTScans
from src.transforms.noise import (
    AddGaussianNoise, 
    AddPoissonNoise, 
    SpectralGaussianBlur, 
    SpectralBernoulliNoise
)

def visualize_compound_noise():
    print("⚗️ Experimenting with Sequential Noise Compounding...\n")
    
    # Define The "Cocktail"
    # We apply Spectral FIRST (Physics: Acquisition happens first)
    # Then Real (Physics: Detector noise happens last)
    compound_transform = T.Compose([
        # Step A: Spectral Degradation
        SpectralBernoulliNoise(prob_drop_range=(0.2, 0.2)), # Drop 20% freq
        SpectralGaussianBlur(sigma_range=(0.2, 0.2)),       # Blur high freq
        
        # Step B: Real Domain Degradation
        AddPoissonNoise(lam_range=(50, 50)),                # Photon Starvation
        AddGaussianNoise(std_range=(0.05, 0.05))            # Electronic Hiss
    ])
    
    # Load One Clean Image
    val_path = default_cfg.PROCESSED_PATH / "val" / "images"
    
    # We use a dummy transform just to get the clean Tensor
    ds = CTScans(image_dir=val_path, transform=T.Compose([T.CenterCrop(512), T.ToTensor()]))
    
    # Get clean image (Dataset returns input=clean, target=clean currently)
    result = ds[0]
    if len(result) == 2:
        noisy, clean_tensor = result
    else:
        noisy, noisy2, clean_tensor = result
    
    # Apply Compounding
    noisy_tensor = compound_transform(clean_tensor.clone())
    
    # Get string log of the compounding.
    noise_string = get_noise_name(compound_transform)

    print(f"  Noise Log: [ {noise_string} ]")

    # Visualization
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot Clean
    ax[0].imshow(clean_tensor.squeeze(), cmap='gray')
    ax[0].set_title("Original (Target)")
    ax[0].axis('off')

    # Plot Noisy
    ax[1].imshow(noisy_tensor.squeeze(), cmap='gray')
    ax[1].set_title(f"Compound Noise (Input)\n({noise_string})")
    ax[1].axis('off')
    
    # Plot Difference (The Noise Itself)
    diff = torch.abs(clean_tensor - noisy_tensor)
    ax[2].imshow(diff.squeeze(), cmap='inferno')
    ax[2].set_title("Noise Map (Difference)")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

    print("\n✅ Noise Compounding Test Concluded!!!")


if __name__ == "__main__":
    visualize_compound_noise()
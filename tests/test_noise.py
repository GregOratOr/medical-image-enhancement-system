import torchvision.transforms as T
import matplotlib.pyplot as plt

from src.datasets.dataset import CTScans
from src.transforms.noise import AddGaussianNoise, RandomSpectralDrop
from configs.dataset_config import default_cfg
from src.utils.logging_utils import get_noise_name

def test_noise_vis() -> None:
    # 1. Define Pipeline
    # Common: Crop & Convert
    common_tx = T.Compose([
        T.CenterCrop(512),
        T.ToTensor(),
    ])
    
    # Noise: Mix of Spectral and Real
    noise_tx = T.Compose([
        RandomSpectralDrop(drop_ratio=0.1), # Heavy spectral damage
        AddGaussianNoise(std_range=(0.1, 0.2)) # Additive real noise
    ])

    # 2. Load Data
    val_path = default_cfg.PROCESSED_PATH / "val" / "images"
    ds = CTScans(image_dir=val_path, transform=common_tx, noise_transform=noise_tx)
    
    # 3. Get Sample
    result = ds[0]
    if len(result) == 2:
        noisy, clean = result
    else:
        noisy, noisy2, clean = result
    
    # 4. Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(clean.squeeze(), cmap='gray')
    ax[0].set_title("Target (Clean)")
    ax[1].imshow(noisy.squeeze(), cmap='gray')
    ax[1].set_title(f"Input ({get_noise_name(noise_tx)})")
    plt.show()

    print("\n✅ Noise Test Concluded!!!")

if __name__ == "__main__":
    test_noise_vis()
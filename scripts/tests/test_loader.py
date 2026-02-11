from configs.config import default_dataset_cfg
from src.datasets.dataset import CTScans
from torch.utils.data import DataLoader
import torchvision.transforms as T

def test_loader():
    print("🚀 Starting Data Loader Test...\n")
    
    train_transform = T.Compose([
        T.ToTensor(),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.CenterCrop(512)
    ])
    

    # Initialize Dataset using Config
    try:
        # Construct path: data/processed/train/patches
        train_path = default_dataset_cfg.PROCESSED_PATH / "train" / "patches"
        train_ds = CTScans(image_dir=train_path, transform=train_transform)
        
        # Construct path: data/processed/val/images
        val_path = default_dataset_cfg.PROCESSED_PATH / "val" / "images"
        val_ds = CTScans(image_dir=val_path, transform=val_transform)
        
        print(f"✅ Train Dataset Loaded: {len(train_ds)} items from {train_path}")
        print(f"✅ Val Dataset Loaded:   {len(val_ds)} items from {val_path}")

    except FileNotFoundError as e:
        print(f"\n❌ CRITICAL: {e}")
        print("   Did you run 'scripts/make_dataset.py' yet?")
        return

    # Initialize DataLoader
    batch_size = getattr(default_dataset_cfg, 'BATCH_SIZE', 4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    tx, ty = next(iter(train_loader))

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    vx, vy = next(iter(val_loader))

    
    # Fetch & Inspect One Batch
    print("\n📦 Inspecting Train Batch...")
    print(f"   Input Shape:  {tx.shape}")   # Expect [B, 1, 256, 256]
    print(f"   Target Shape: {ty.shape}")  # Expect [B, 1, 256, 256]
    print(f"   Value Range:  [{tx.min():.4f}, {tx.max():.4f}]\n")

    print("\n📦 Inspecting Val Batch...")
    print(f"   Input Shape:  {vx.shape}")   # Expect [B, 1, 512, 512]
    print(f"   Target Shape: {vy.shape}")  # Expect [B, 1, 512, 512]
    print(f"   Value Range:  [{vx.min():.4f}, {vx.max():.4f}]\n")
    
    # Automatic QA Checks
    if tx.shape[1] != 1:
        print("   ⚠️  WARNING: Channel dim is not 1 (Grayscale expected).")

    if vx.shape[-1] == 512:
        print("   ✅ Validation Crop Successful (512x512).")
    else:
        print(f"   ❌ Validation Crop Failed! Got {vx.shape}")
        
    if tx.max() <= 1.0 and tx.min() >= 0.0:
        print("   ✅ Normalization Verified (0.0 - 1.0).")
    else:
        print(f"   ❌ Normalization Failed! Range: [{tx.min()}, {tx.max()}]")
    
    print("\n✅ Loader Test Concluded!!!")
    
if __name__ == "__main__":
    test_loader()
import gc
import draccus

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from src.models.wrappers import DynamicPadWrapper
from src.datasets.dataset import SimulationDataset
from configs.config import InferenceConfig
from src.utils.factories import resolve_models, resolve_device, resolve_noise_transforms

def main():

    cfg = draccus.parse(config_class=InferenceConfig)
    print(f"✅ Configuration Loaded.")
    print("🚀 Initializing Inference Pipeline...")
    print(f"📂 Run Directory: {cfg.run_dir}")
    # Set the device
    device = resolve_device(cfg.use_gpu)
    
    # Quick sanity check
    print(f"✅ Config Loaded Successfully!")
    print(f"📂 Output Directory: {cfg.run_dir}")
    print(f"💻 Device: {device}")
    print(f"📦 Checkpoint: {Path(cfg.inference['checkpoint_path']) / cfg.inference['checkpoint_name']}")
    print(f"🖼️  Data Path: {cfg.inference['data_path']}")
    print(f"⚙️  Model Params: {cfg.models[0].model_params}")

    # Load Dataset
    data_path = cfg.inference['data_path']
    print(f"🔍 Loading dataset from: {data_path}")
    
    common_transforms = T.Compose([
        T.CenterCrop(256),
        T.ToTensor()
    ])
    noise_transforms = resolve_noise_transforms(cfg.inference["noise_params"])

    
    dataset = SimulationDataset(
        data_path=data_path,
        transforms=common_transforms,
        noise_transforms=noise_transforms,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.inference['batch_size'],
        num_workers=cfg.inference['num_workers'],
        shuffle=False, # inference!
        pin_memory=True if device.type != "cpu" else False,
        persistent_workers=True
    )

    print(f"📦 Found {len(dataset)} images. Batches to process: {len(dataloader)}")

    # Instantiate Base model.
    base_models = resolve_models(model_configs=cfg.models, device=device)
    base_model = base_models[cfg.models[0].name]
    # Load Checkpoint
    ckpt_path = Path(cfg.inference['checkpoint_path']) / cfg.inference['checkpoint_name']
    print(f"🧠 Loading model checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint) # Fallback if it's a raw state_dict
    base_model.load_state_dict(state_dict)

    # Add DynamicPadWrapper to the base model.
    depth = cfg.models[0].model_params.get('depth', 4)
    model = DynamicPadWrapper(base_model, depth=depth)

    model.to(device)
    model.eval()
    print("✅ Model loaded, wrapped, and ready for inference.")
    
    print("✨ Starting Denoising...")
    
    with torch.inference_mode():
        for images, filenames, clean_imgs in tqdm(dataloader, desc="Inference Progress"):
            # Move images to GPU
            images = images.to(device)
            
            # Forward pass (Dynamic padding handles everything under the hood)
            outputs = model(images)
            
            # Post-process: Ensure range [0.0, 1.0]
            outputs = torch.clamp(outputs, 0.0, 1.0)
            
            # outputs: GPU -> CPU for saving.
            outputs = outputs.cpu()
            clean_imgs = clean_imgs.cpu()
            images = images.cpu()
            
            # Iterate through the batch to save each image individually
            for output_tensor, filename, clean_tensor, input_tensor in zip(outputs, filenames, clean_imgs, images):
                # # 3D tensor [1, H, W] -> 2D PIL Image
                output_img = F.to_pil_image(output_tensor)
                clean_img = F.to_pil_image(clean_tensor)
                input_img = F.to_pil_image(input_tensor)
                # final_img = [ [input]+[output]+[ground_truth] ]
                final_img = F.to_pil_image(torch.cat([input_tensor, output_tensor, clean_tensor], dim=2))

                # Save the image
                clean_img.save(cfg.run_dir / f"{filename}")
                input_img.save(cfg.run_dir / f"inp_{filename}")
                output_img.save(cfg.run_dir / f"denoised_{filename}")
                final_img.save(cfg.run_dir / f"inp-op-gt_{filename}")
                
                
    print(f"✅ Pipeline Complete! All results saved to: {cfg.run_dir.absolute()}")

    del model
    del dataset
    del dataloader
    gc.collect() # GC collects the deleted model and clears the memory.
    torch.cuda.empty_cache() # Removes any cache data related to the removed model from the GPU.
    torch.cuda.ipc_collect() # Clears memory from multi-processing workers

if __name__ == "__main__":
    main()


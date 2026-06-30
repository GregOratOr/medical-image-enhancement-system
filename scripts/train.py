import gc
import sys
import draccus
from pathlib import Path

import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from configs.config import Config, Model
from src.datasets.dataset import CTScans
from src.utils.logger import UnifiedLogger
from src.utils import factories as F
from src.trainers.n2n import Noise2NoiseTrainer
from src.evaluation.metrics import Metrics
# print(f"Project root is set to: [{str(root_dir)}]")



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (slightly slower but reproducible)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def main() -> None:
    """Main training entry point.

    Args:
        cfg (Config): The Config object, auto-populated using Draccus(defausts + yaml + cli args).
    """
    # Load the config file.
    cfg = draccus.parse(config_class=Config)
    print(f"✅ Configuration Loaded.")
    print(f"📂 Run Directory: {cfg.run_dir}")
    
    # Simple check to ensure it worked
    if not cfg.run_dir.exists():
        print("⚠️ Warning: Run directory was not created. Check Config.__post_init__")
    
    # print(cfg)

    # Set the seed
    print(f"🌱 Setting Global Seed: {cfg.seed}")
    set_seed(cfg.seed)

    # Set the device to [cuda, xpu, mps] respectively if gpu is available, else cpu
    device = F.resolve_device(cfg.use_gpu)
    print(f"⚡ Using device: {device}")

    # Initialize Logger.
    logger = UnifiedLogger(
        log_dir=cfg.run_dir,
        project_name=cfg.project_name,
        experiment_name=cfg.experiment.name,
        name=cfg.logs.name,
        config=draccus.encode(cfg),
        use_tensorboard=cfg.logs.use_tensorboard,
        use_wandb=cfg.logs.use_wandb,
        log_interval=cfg.logs.log_interval
    )
    logger.info(f"🚀 Experiment '{cfg.experiment.name}' Initialized.")
    logger.debug(f"📂 Run Directory: {str(cfg.run_dir)}")
    logger.info("🛠️ Setting up Data Pipeline...")

    # Get Noise Transforms
    noise_transforms = F.resolve_noise_transforms(cfg.data.preprocess_params.get("noise_params", []))
    common_transforms = T.Compose([
        T.CenterCrop(256),
        T.ToTensor(),
    ])
    print(noise_transforms)

    # Create Datasets
    train_dataset = CTScans(
        image_dir=Path(cfg.data.train_dir),
        transform=common_transforms,
        noise_transform=noise_transforms,
        mode=cfg.data.preprocess_params.get("mode", 'n2n'),
        # subset_size=cfg.data.subset # Optional: good for debugging!
    )
    logger.info(f"📊 Training Samples: {len(train_dataset)}")

    val_dataset = CTScans(
        image_dir=Path(cfg.data.val_dir),
        transform=common_transforms,
        noise_transform=noise_transforms,
        mode=cfg.data.preprocess_params.get("mode", 'n2n'),
        # subset_size=cfg.data.subset
    )
    logger.info(f"📊 Validation Samples: {len(val_dataset)}")

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True, # Faster transfer to GPU
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False, # No need to shuffle validation
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    try:
        sample_batch = next(iter(train_loader))
        noisy1, noisy2, clean = sample_batch
        logger.info(f"✅ Data Flow Verified. Batch Shape: {clean.shape}")
    except Exception as e:
        logger.error(f"❌ Data Loading Failed: {e}")
        raise e

    # Initialize Model
    logger.info("🧠 Initializing Model...")

    models_dict = F.resolve_models(model_configs=cfg.models, device=device)
    if not cfg.models:
        raise ValueError("❌ No models were defined in the config (.yaml) file!")
    
    # Only one model for this project.
    model = models_dict[cfg.models[0].name]
    
    optimizers_dict = F.resolve_optimizers(models_dict=models_dict, optimizer_configs=cfg.train.optimizers)
    if not cfg.train.optimizers:
        raise ValueError("❌ No optimizers were defined in the configuration!")
    
    # Only one optimizer for this project.
    optimizer = optimizers_dict[cfg.train.optimizers[0].label]

    schedulers_dict = F.resolve_schedulers(optimizers_dict=optimizers_dict, scheduler_configs=cfg.train.schedulers)
    if not cfg.train.schedulers:
        raise ValueError("❌ No Schedulers were defined in the configuration!")

    scheduler = schedulers_dict['main']
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Model: {model.__class__.__name__} | Parameters: {param_count:,}")
    logger.info(f"🔧 Optimizer: {optimizer.__class__.__name__} | LR: {optimizer.param_groups[0]['lr']} | Scheduler: {scheduler.__class__.__name__}")

    logger.info("🏋️ Initializing Trainer...")
    
    # Instantiate Trainer
    trainer = Noise2NoiseTrainer(
        model=model,
        optimizers=optimizer,
        schedulers=scheduler,
        loaders={'train':train_loader,
                 'val': val_loader},
        criterion={'mse': nn.MSELoss(),
                   'l1': nn.L1Loss()},
        use_amp=cfg.train.use_amp,
        save_interval=cfg.train.save_interval,
        monitor_metrics=cfg.train.monitor_metrics,
        logger=logger,
        device=device,
        max_epochs=cfg.train.max_epochs,
        **cfg.train.kwargs
    )
    
    # Start Training
    logger.info("🚀 Starting Training Loop...")
    
    try:
        if cfg.train.resume_checkpoint != "":
            ckpt_path = Path(cfg.train.resume_checkpoint)
            if ckpt_path.exists():
                trainer.resume_from_checkpoint(str(ckpt_path))
            else:
                logger.warning(f"⚠️ Requested checkpoint '{ckpt_path}' not found! Starting from scratch.")

        trainer.fit(epochs=cfg.train.max_epochs)
        logger.info("🎉 Training Completed Successfully.")

    except KeyboardInterrupt:
        logger.warning("🛑 Training interrupted by user.")
    except Exception as e:
        logger.error(f"❌ Training Failed: {e}")
        raise e
    
    logger.close()

    del trainer
    del model
    del train_loader
    del val_loader
    del train_dataset
    del val_dataset
    gc.collect() # GC collects the deleted model and clears the memory.
    torch.cuda.empty_cache() # Removes any cache data related to the removed model from the GPU.
    torch.cuda.ipc_collect() # Clears memory from multi-processing workers

if __name__ == "__main__":
    main()
    


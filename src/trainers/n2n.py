import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.optim.optimizer import Optimizer as Optimizer
from typing import Dict, Any

from src.trainers.engine import Engine
from src.evaluation.metrics import Metrics

class Noise2NoiseTrainer(Engine):
    """
    Trainer for Self-Supervised Denoising (Noise2Noise).
    
    Inherits from Engine to handle:
    - Device management
    - Logging & Checkpointing
    - Loop iteration
    """
    
    # We don't need an __init__ here unless we need to add 
    # specific variables that Engine doesn't have. 
    # For now, Engine's __init__ is enough!
    def __init__(self, model: nn.Module, criterion: Dict[str, nn.Module], optimizers: torch.optim.Optimizer, **kwargs):
        super().__init__(model=model, criterion=criterion, optimizers=optimizers, **kwargs)

        self.model: nn.Module = model
        self.criterion: Dict[str, nn.Module] = criterion
        self.optimizers: torch.optim.Optimizer = optimizers

    def train_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        # 1. Unpack Data
        # source = Input Noisy Image
        # target = Another Noisy Version (The "Target")
        # clean = Ground Truth (We IGNORE this during training - that's the magic of N2N!)
        source, target, _ = batch
        source, target = source.to(self.device), target.to(self.device)

        # 2. Forward Pass (Auto-Mixed Precision)
        # The 'self.use_amp' flag comes from Engine
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            output = self.model(source)
            
            # 3. Calculate Loss
            loss_mse = self.criterion['mse'](output, target)
            loss_l1 = self.criterion['l1'](output, target)
            
            # Weighted Sum (Alpha default 0.5)
            # You can tune 'loss_alpha' in your config.py
            alpha = self.cfg.get("loss_alpha", 0.5)
            train_loss = (alpha * loss_mse) + ((1 - alpha) * loss_l1)

        # 4. Backward Pass & Optimization
        self.optimizers.zero_grad()
        
        # Scale loss for AMP stability (Engine handles the scaler)
        self.scaler.scale(train_loss).backward()
        
        # Step the optimizer
        self.scaler.step(self.optimizers)
        self.scaler.update()

        # 5. Return Metrics for Logging
        # These keys ("loss", "mse") will appear in TensorBoard/W&B
        return {
            "loss": train_loss.item(),
            "mse": loss_mse.item(),
            "l1": loss_l1.item()
        }
    
    def validate_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        # 1. Unpack Data
        # CRITICAL DIFFERENCE: We NEED 'clean_gt' here for true validation!
        source, target, clean_gt = batch
        source, target, clean_gt = source.to(self.device), target.to(self.device), clean_gt.to(self.device)

        # 2. Forward Pass
        # (Engine automatically wraps this in 'with torch.no_grad():')
        denoised_output = self.model(source)
        
        # 3. Calculate Proxy Loss (Same as training)
        # We check this to ensure the model is converging mathematically
        loss_mse = self.criterion['mse'](denoised_output, target)
        loss_l1 = self.criterion['l1'](denoised_output, target)
        alpha = self.cfg.get("loss_alpha", 0.5)
        val_loss = (alpha * loss_mse) + ((1 - alpha) * loss_l1)

        # 4. Calculate Metrics (The New Way)
        # Compute against 'clean_gt' because we want to know 
        # "How close we are to the REAL image, not the noisy target."
        metric_results = Metrics.compute(
            prediction=denoised_output, 
            target=clean_gt, 
            metrics=['psnr', 'ssim'], 
            device=self.device
        )
        
        # 5. Visualization (Only for the first batch of the epoch)
        if batch_idx == 0 and self.logger:
            self._log_comparison_grid(source, denoised_output, clean_gt, step=self.current_epoch)

        # 5. Return everything
        # The Engine will average these over the epoch automatically.
        return {
            "val_loss": val_loss.item(),
            **metric_results  # Unpacks {'psnr': 30.5, 'ssim': 0.9} into this dict
        }
    
    def _log_comparison_grid(self, source, output, target, step):
        """
        Creates a side-by-side comparison grid: [Noisy Input | Denoised Output | Clean GT]
        """
        # Take just the first 4 images from the batch to save storage
        n_images = min(source.size(0), 4)
        
        # 1. Denormalize if necessary (assuming data is 0-1)
        # If your loader normalizes to -1,1, you'd un-normalize here.
        # For now, we assume [0,1].
        
        # 2. Stack images horizontally: [B, C, H, W] -> [B, C, H, W*3]
        # We usually want:
        # Row 1: Source, Output, Target
        # Row 2: Source, Output, Target
        
        # Helper to concatenate along width (dim=3)
        comparison = torch.cat([source[:n_images], output[:n_images], target[:n_images]], dim=3)
        
        # 3. Send to Logger
        # This will create a grid in TensorBoard/W&B
        self.logger.log_images("val_comparison", list(comparison), step=step)
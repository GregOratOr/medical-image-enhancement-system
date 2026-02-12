import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.optim.optimizer import Optimizer as Optimizer
from typing import Any

from src.trainers.engine import Engine
from src.evaluation.metrics import Metrics
from src.utils.visualization import Visualizer


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
    def __init__(self, model: nn.Module, criterion: dict[str, nn.Module], optimizers: torch.optim.Optimizer, **kwargs):
        super().__init__(model=model, criterion=criterion, optimizers=optimizers, **kwargs)

        self.model: nn.Module = model
        self.criterion: dict[str, nn.Module] = criterion
        self.optimizers: torch.optim.Optimizer = optimizers
        self.viz: Visualizer = Visualizer()

    def train_step(self, batch: Any, batch_idx: int) -> dict[str, float]:
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
    
    def validate_step(self, batch: Any, batch_idx: int) -> dict[str, float]:
        # 1. Unpack Data
        # CRITICAL: NEED 'clean_gt' here for true validation!
        source, target, clean_gt = batch
        source, target, clean_gt = source.to(self.device), target.to(self.device), clean_gt.to(self.device)

        # 2. Forward Pass
        denoised_output = self.model(source)
        
        # 3. Calculate Proxy Loss (Same as training)
        loss_mse = self.criterion['mse'](denoised_output, target)
        loss_l1 = self.criterion['l1'](denoised_output, target)
        alpha = self.cfg.get("loss_alpha", 0.5)
        val_loss = (alpha * loss_mse) + ((1 - alpha) * loss_l1)

        # 4. Calculate Metrics
        # Computes —"How close we are to the REAL image, not the noisy target."
        metric_results = Metrics.compute(
            prediction=denoised_output, 
            target=clean_gt, 
            metrics=['psnr', 'ssim'], 
            device=self.device
        )
        
        # 5. Visualization (Only for the first batch of the epoch)
        if batch_idx == 0 and self.logger:
            self._log_comparison_grid(source=source, output=denoised_output, target=target, clean_gt=clean_gt, step=self.current_epoch)

        # 6. Return everything
        # The Engine will average these over the epoch automatically.
        return {
            "val_loss": val_loss.item(),
            **metric_results  # Unpacks {'psnr': 30.5, 'ssim': 0.9} into this dict
        }
    
    def _log_comparison_grid(self, source: torch.Tensor, output: torch.Tensor, target: torch.Tensor, clean_gt: torch.Tensor, step):
        """Creates a diagnostic grid and logs it to Disk, TensorBoard, and W&B.

        Args:
            source (torch.Tensor): The input(noisy) image.
            output (torch.Tensor): The denoised output.
            target (torch.Tensor): The target image (clean/noisy).
            clean_gt (torch.Tensor): The clean ground truth image.
            step (int): The current epoch.
        """

        # Create the rich 3-row grid using Visualizer
        # (Spatial Domain | Frequency Domain | Error Maps)
        grid = self.viz.create_grid(
            clean_gt=clean_gt,
            noisy_input=source,
            denoised_output=output,
            target=target,
            max_images=4  # Number of samples to show
        )
        
        # Define the save path
        save_path = self.logger.log_dir / "training"
        
        # Log and Save
        # Passing the path enables saving to disk.
        if self.logger:
            self.logger.log_image(
                tag="val/comparison", 
                images=grid, 
                step=step, 
                path=save_path
            )
import math
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.optim.optimizer import Optimizer
from typing import Any, cast

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
    def __init__(self, model: nn.Module, criterion: dict[str, nn.Module], optimizers: Optimizer, **kwargs):
        super().__init__(model=model, optimizers=optimizers, criterion=criterion, **kwargs)
        self.model = cast(nn.Module, self.model)
        self.criterion = cast(dict[str, nn.Module], self.criterion)
        self.optimizers = cast(torch.optim.Optimizer, self.optimizers)
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
            
            # Post-processing
            output = self._post_output_operations(source=source, output=output)

            # 3. Calculate Loss
            loss_mse = self.criterion['mse'](output, target)
            loss_l1 = self.criterion['l1'](output, target)
            
            # Weighted Sum (Alpha default 0.5)
            # You can tune 'loss_alpha' in your config.py
            alpha = self.kwargs.get("loss_alpha", 0.5)
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
        alpha = self.kwargs.get("loss_alpha", 0.5)
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
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Log and Save
        # Passing the path enables saving to disk.
        self.logger.log_image(
            tag="val/comparison", 
            images=grid, 
            step=step, 
            path=save_path
        )
    
    def _post_output_operations(self, source: torch.Tensor, output: torch.Tensor):
        if self.kwargs.get("force_freq_op", False):
            # We need the mask! (See Step 3 below)
            mask = self._get_mask(source.shape, self.device) 
            output = self._force_frequencies(output, source, mask)

        return output

    def _force_frequencies(self, denoised: torch.Tensor, source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Performs the post-operation procedure of forcing known frequencies before loss calculation.

        Args:
            denoised (Tensor): The spatial prediction from the model.
            source (Tensor): The original noisy input (contains the ground-truth acquired frequencies).
            mask (Tensor): A [0, 1] mask where 1 indicates a 'known/trusted' frequency.
            
        Returns:
            Tensor: The blended spatial image.
        """
        
        orig_dtype = denoised.dtype
        
        # Upcast to float32 for mathematically stable FFTs
        denoised_f32 = denoised.to(torch.float32)
        source_f32 = source.to(torch.float32)
        
        # FFT and Shift
        denoised_spec = torch.fft.fftshift(torch.fft.fft2(denoised_f32))
        source_spec = torch.fft.fftshift(torch.fft.fft2(source_f32))
        
        # Match dtypes and devices
        source_spec = source_spec.to(denoised_spec.dtype).to(denoised_spec.device)
        mask_f32 = mask.to(denoised_spec.dtype).to(denoised_spec.device)
        
        # Force known frequencies using the mask
        forced_spec = (source_spec * mask_f32) + (denoised_spec * (1.0 - mask_f32))
        
        # Shift back and IFFT to spatial domain
        forced_spatial = torch.fft.ifft2(torch.fft.ifftshift(forced_spec)).real
        
        return forced_spatial
    
    def _get_mask(self, shape, device):
        # Create a 32x32 square mask in the center of the frequencies
        B, C, H, W = shape
        mask = torch.zeros((B, C, H, W), device=device)

        start_size = 32
        end_size = 0  
        
        # ==========================================
        # COSINE DECAY MATH
        # ==========================================
        # Calculates progress from 0.0 to 1.0
        max_epochs = self.kwargs.get("max_epochs", 100)
        progress = self.current_epoch / max_epochs
        
        # Cosine multiplier goes from 1.0 down to 0.0 smoothly
        cosine_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Calculate current size based on the multiplier
        current_size = int(end_size + (start_size - end_size) * cosine_mult)
        
        # Ensure it's an even number so the center box stays perfectly symmetric
        if current_size % 2 != 0:
            current_size += 1

        if current_size <= 0:
            return mask

        cy, cx = H // 2, W // 2
        mask[:, :, cy-16:cy+16, cx-16:cx+16] = 1.0
        return mask
    

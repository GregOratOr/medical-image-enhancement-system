import torch
from torch.amp.grad_scaler import GradScaler
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional
from pathlib import Path
from tqdm import tqdm

from datetime import datetime
from src.utils.logger import UnifiedLogger

class FallbackLogger:
    """A dummy logger that prints to console but skips file/W&B logging.
    Used when no UnifiedLogger is provided.
    """
    def __init__(self):
        self.log_dir = Path(f"./tests/debug/Experiment-{datetime.now()}") # Default for debug runs
    
    def info(self, msg: str):
        print(f"[Engine] {msg}")
    
    def log_metrics(self, *args, **kwargs): pass # Do nothing
    def log_images(self, *args, **kwargs): pass  # Do nothing
    def close(self): pass # Do nothing

class Engine(ABC):
    """
    Abstract Base Class for training loops.
    
    Responsibilities:
    - Device management (CPU/CUDA).
    - Epoch iteration (Train -> Val).
    - Logging (TensorBoard/W&B/Console).
    - Checkpointing (Save/Resume).
    - Mixed Precision (AMP) setup.
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict[str, nn.Module]],
        loaders: Dict[str, torch.utils.data.DataLoader],
        optimizers: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
        criterion: Union[nn.Module, Dict[str, nn.Module]],
        schedulers: Union[Any, Dict[str, Any]] = None,
        config: Optional[Dict] = None,
        logger: Optional[UnifiedLogger] = None,
        device: str = "cuda"
    ):
        
        self.cfg = config or {}
        self.logger = logger if logger is not None else FallbackLogger()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 1. Setup Model(s)
        if isinstance(model, dict):
            self.model = {k: v.to(self.device) for k, v in model.items()}
        else:
            self.model = model.to(self.device)

        # 2. Setup Data & Optimization
        self.loaders = loaders
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.criterion = criterion

        # 3. Mixed Precision
        self.use_amp: bool = self.cfg.get("use_amp", False)
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)
        
        # 4. State Tracking
        self.start_epoch = 0
        self.current_epoch = 0

        self.monitor_metrics = self.cfg.get("monitor_metrics", {'val_loss': 'min'})
        self.best_metrics = {}
        for metric, mode in self.monitor_metrics.items():
            self.best_metrics[metric] = float('-inf') if mode == 'max' else float('inf')
        
        # Log setup
        if self.logger:
            self.logger.info(f"Engine initialized on {self.device}. AMP: {self.use_amp}")
        else:
            raise ValueError("Logger not initialized.")

    @abstractmethod
    def train_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        """
        Execute a single training step.
        Must return a dictionary of scalar metrics (e.g., {'loss': 0.5}).
        """
        pass

    @abstractmethod
    def validate_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        """
        Execute a single validation step.
        Must return a dictionary of scalar metrics (e.g., {'psnr': 30.0}).
        """
        pass

    def _run_epoch(self, epoch: int, mode: str = "train") -> Dict[str, float]:
        """
        Iterates over the DataLoader for one epoch.
        """
        is_train = mode == "train"
        loader = self.loaders["train"] if is_train else self.loaders["val"]
        
        # 1. Set Model Mode (Behavioral)
        # train(): enables Dropout, BatchNorm updates
        # eval(): disables Dropout, freezes BatchNorm stats
        if isinstance(self.model, dict):
            for m in self.model.values(): m.train() if is_train else m.eval()
        else:    
            self.model.train() if is_train else self.model.eval()

        # 2. Set Context Manager (Computational)
        # train: enable_grad() -> Tracks operations for backprop
        # val: inference_mode() -> Disables tracking & version counters (Faster & Less RAM)
        context = torch.enable_grad() if is_train else torch.inference_mode()

        # Trackers
        epoch_metrics = {}
        metric_counts = {}
        
        # Progress Bar
        pbar = tqdm(loader, desc=f"{mode.upper()} Ep {epoch}", leave=False, dynamic_ncols=True)

        # Context Manager (Grad / No Grad)
        with context:
            for batch_idx, batch in enumerate(pbar):
                
                # Execute Step
                if is_train:
                    step_metrics = self.train_step(batch, batch_idx)
                else:
                    step_metrics = self.validate_step(batch, batch_idx)

                # Accumulate Metrics
                for k, v in step_metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                    metric_counts[k] = metric_counts.get(k, 0) + 1
                # Update Progress Bar (Show current loss)
                if batch_idx % 10 == 0:
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in step_metrics.items()})

        # Average Metrics
        avg_metrics = {k: v / metric_counts[k] for k, v in epoch_metrics.items()}
        return avg_metrics

    def _step_schedulers(self, val_metrics: Dict[str, float]):
        """Steps all schedulers. Handles ReduceLROnPlateau specially."""
        if not self.schedulers: 
            return
        
        # FIX: Explicitly handle Dict vs List vs Single Object
        if isinstance(self.schedulers, dict):
            # Extract the actual scheduler objects from the dict values
            schedulers_list = list(self.schedulers.values())
        elif isinstance(self.schedulers, list):
            schedulers_list = self.schedulers
        else:
            # Assume it's a single scheduler object
            schedulers_list = [self.schedulers]

        for sched in schedulers_list:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(val_metrics.get("val_loss", 0.0)) 
            else:
                sched.step()

    def _get_scheduler_state(self):
        if not self.schedulers: return None
        if isinstance(self.schedulers, dict):
            return {k: v.state_dict() for k, v in self.schedulers.items()}
        elif isinstance(self.schedulers, list):
            return [s.state_dict() for s in self.schedulers]
        else:
            return self.schedulers.state_dict()

    def _load_scheduler_state(self, state):
        if isinstance(self.schedulers, dict):
            for k, v in self.schedulers.items(): v.load_state_dict(state[k])
        elif isinstance(self.schedulers, list):
            for i, s in enumerate(self.schedulers): s.load_state_dict(state[i])
        else:
            self.schedulers.load_state_dict(state)
    
    def fit(self, epochs: int):
        """
        The main training loop.
        """
        self.logger.info(f"🚀 Starting training from {self.start_epoch} to {epochs} epochs.")
        
        for epoch in range(self.start_epoch, epochs):
            self.current_epoch = epoch

            # --- Training ---
            train_metrics = self._run_epoch(epoch, mode="train")
            self.logger.log_metrics(train_metrics, step=epoch, prefix="train")

            # --- Validation ---
            val_metrics = self._run_epoch(epoch, mode="val")
            self.logger.log_metrics(val_metrics, step=epoch, prefix="val")

            # --- Scheduling ---
            if self.schedulers:
                self._step_schedulers(val_metrics)
            
            # --- Multi-Metric Monitoring
            for metric, mode in self.monitor_metrics.items():
                if metric not in val_metrics:
                    continue
                
                current_val = val_metrics[metric]
                best_val = self.best_metrics[metric]

                hasImproved = (current_val > best_val) if mode == "max" else (current_val < best_val)

                if hasImproved:
                    # Update state
                    self.best_metrics[metric] = current_val
                    # Save checkpoint
                    self.save_checkpoint(epoch=epoch, filename=f"best_{metric}.pth")
                    # Log the update.
                    self.logger.info(f"⭐ New Best {metric.upper()}: {current_val:.4f}")
                
            # --- Checkpointing ---
            # Save 'latest' every epoch
            self.save_checkpoint(epoch)
        self.logger.info("✅ Training Complete.")
        self.logger.close()

    def save_checkpoint(self, epoch: int, filename: str='latest.pth'):
        """Saves the training state."""
        save_dir = Path(self.logger.log_dir) / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            "epoch": epoch,
            "best_metrics": self.best_metrics,
            "config": self.cfg,
            # Handle Single vs Dict Models
            "model_state": self.model.state_dict() if not isinstance(self.model, dict) 
                           else {k: v.state_dict() for k, v in self.model.items()},
            # Handle Single vs Dict Optimizers
            "optimizer_state": self.optimizers.state_dict() if not isinstance(self.optimizers, dict) 
                               else {k: v.state_dict() for k, v in self.optimizers.items()},
            "scheduler_state": self._get_scheduler_state()
        }
        # 1. ALWAYS save 'latest' or custom-name (for resuming/metric-based checkpointing)
        torch.save(state, save_dir / filename)
        
        if filename == 'latest.pth':
            # PERIODICALLY save history (for analysis)
            # Get interval from config, default to 0 (disabled) 
            save_interval = self.cfg.get("save_interval", 0) 
        
            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                history_filename = f"checkpoint-{epoch + 1:04d}.pth"
                torch.save(state, save_dir / history_filename)
                self.logger.info(f"Saved historical checkpoint: {history_filename}")

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Loads the training state."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_metrics = checkpoint["best_metrics"]

        # Load Model
        if isinstance(self.model, dict):
            for k, v in self.model.items():
                v.load_state_dict(checkpoint["model_state"][k])
        else:
            self.model.load_state_dict(checkpoint["model_state"])
            
        # Load Optimizer
        if isinstance(self.optimizers, dict):
            for k, v in self.optimizers.items():
                v.load_state_dict(checkpoint["optimizer_state"][k])
        else:
            self.optimizers.load_state_dict(checkpoint["optimizer_state"])

        # Load Scheduler states
        if "scheduler_state" in checkpoint and self.schedulers:
            self._load_scheduler_state(checkpoint["scheduler_state"])
            
        self.logger.info(f"Resumed from Epoch {self.start_epoch}")
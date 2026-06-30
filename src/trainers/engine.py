import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
from tqdm import tqdm

from datetime import datetime
from src.utils.logger import UnifiedLogger

class FallbackLogger:
    """
    A dummy logger that prints to console but skips file/W&B logging.
    Used when no UnifiedLogger is provided.
    """
    def __init__(self):
        self.log_dir = Path(f"./experiments/debug/Experiment-{datetime.now()}") # Default for debug runs
        self.log_interval = 10
        self.info("[Engine] FallbackLogger assigned and initialized.")
        self.info("[Engine] NOTE: This logger only prints to console.")
    
    def info(self, msg: str):
        print(f"[INFO] [Engine] {msg}")
    
    def warning(self, msg: str) -> None:
        print(f"[WARNING] [Engine] {msg}")

    def error(self, msg: str) -> None:
        print(f"[ERROR] [Engine] {msg}")

    def critical(self, msg: str) -> None:
        print(f"[CRITICAL] [Engine] {msg}")
    
    def debug(self, msg: str) -> None:
        print(f"[DEBUG] [Engine] {msg}")

    def log_metrics(self, *args, **kwargs): pass # Do nothing
    def log_image(self, *args, **kwargs): pass  # Do nothing
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
        model: nn.Module | dict[str, nn.Module],
        loaders: dict[str, torch.utils.data.DataLoader],
        optimizers: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
        criterion: nn.Module | dict[str, nn.Module],
        schedulers: torch.optim.lr_scheduler.LRScheduler | dict[str, torch.optim.lr_scheduler.LRScheduler] | None = None,
        logger: UnifiedLogger | None = None,
        device: str | torch.device = "cpu",
        use_amp: bool = False,
        monitor_metrics: dict = {'val_loss': 'min'},
        save_interval: int = 10,
        **kwargs
    ):
        """Initializes the Trainer class.

        Args:
            model (nn.Module | dict[str, nn.Module]): The model(s) to train.
            loaders (dict[str, torch.utils.data.DataLoader]): Data loaders for training and validation datatasets.
            optimizers (torch.optim.Optimizer | dict[str, torch.optim.Optimizer]): Optimizer(s) for training.
            criterion (nn.Module | dict[str, nn.Module]): Loss function(s) for training.
            schedulers (torch.optim.lr_scheduler.LRScheduler | dict[str, torch.optim.lr_scheduler.LRScheduler] | None, optional): LR scheduler(s) for training. Defaults to None.
            config (dict | None, optional): Config object for various parameters of the training process . Defaults to None.
            logger (UnifiedLogger | None, optional): Logger object. Defaults to None.
            device (str, optional): Device to train on ('cuda' or 'cpu'). Defaults to "cpu".

        Raises:
            ValueError: If the logger is not initialized correctly.
        """

        self.logger = logger if logger is not None else FallbackLogger()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Setup Model(s)
        if isinstance(model, dict):
            self.model = {k: v.to(self.device) for k, v in model.items()}
        else:
            self.model = model.to(self.device)

        # Setup Data & Optimization
        self.loaders = loaders
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.criterion = criterion
        if isinstance(criterion, dict):
            self.criterion = {k: v.to(self.device) for k, v in criterion.items()}
        else:
            self.criterion = criterion.to(self.device)

        # Mixed Precision
        self.use_amp: bool = use_amp
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)
        
        # State Tracking
        self.start_epoch = 0
        self.current_epoch = 0
        self.save_interval = save_interval # -1 to disable saving. 
        self.kwargs = kwargs

        self.monitor_metrics = monitor_metrics
        self.best_metrics = {}
        for metric, mode in self.monitor_metrics.items():
            self.best_metrics[metric] = float('-inf') if mode == 'max' else float('inf')
        
        # Log setup
        if self.logger:
            self.logger.info(f"⚙️ Engine initialized on {self.device}.⚡ AMP: {self.use_amp}")
        else:
            raise ValueError("❌ Logger not initialized.")
        

    @abstractmethod
    def train_step(self, batch: Any, batch_idx: int) -> dict[str, float]:
        """Execute a single training step.
        Must return a dictionary of scalar metrics (e.g., {'loss': 0.5}).
        """
        pass

    @abstractmethod
    def validate_step(self, batch: Any, batch_idx: int) -> dict[str, float]:
        """Execute a single validation step.
        Must return a dictionary of scalar metrics (e.g., {'psnr': 30.0}).
        """
        pass

    def _run_epoch(self, epoch: int, mode: str = "train") -> dict[str, float]:
        """Iterates over the DataLoader for one epoch.

        Args:
            epoch (int): Current epoch.
            mode (str, optional): Decides between a training or a validation epoch. Defaults to "train".

        Returns:
            dict[str, float]: Returns average performance metrics after the epoch.
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

    def _step_schedulers(self, val_metrics: dict[str, float]):
        """Steps all schedulers. Handles ReduceLROnPlateau specially."""
        
        if not self.schedulers: 
            return
        
        # Handle Dict vs List vs Single Object
        if isinstance(self.schedulers, dict):
            # Extract the actual scheduler objects from the Dict values
            schedulers_list = list(self.schedulers.values())
        elif isinstance(self.schedulers, list):
            schedulers_list = self.schedulers
        else:
            # It's a single scheduler object
            schedulers_list = [self.schedulers]

        for sched in schedulers_list:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                primary_metric = list(self.monitor_metrics.keys())[0]
                if primary_metric not in val_metrics:
                    self.logger.warning(
                        f"⚠️ Scheduler expected metric '{primary_metric}' but it wasn't returned "
                        f"by validate_step! Available metrics: {list(val_metrics.keys())}"
                    )
                sched.step(val_metrics[primary_metric]) 
            else:
                sched.step()

    def _get_scheduler_state(self) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Gets the state dict(s) for scheduler(s).

        Returns:
            _type_: Returns a state dictionary or list of state disctionaries.
        """

        if not self.schedulers: return None
        if isinstance(self.schedulers, dict):
            return {k: v.state_dict() for k, v in self.schedulers.items()}
        elif isinstance(self.schedulers, list):
            return [s.state_dict() for s in self.schedulers]
        else:
            return self.schedulers.state_dict()

    def _load_scheduler_state(self, state) -> None:
        """Loads the state dict(s) of scheduler(s).

        Args:
            state (_type_): State dictionary(s).
        """
        
        assert self.schedulers is not None, "No schedulers found."

        if isinstance(self.schedulers, dict):
            for k, v in self.schedulers.items(): v.load_state_dict(state[k])
        elif isinstance(self.schedulers, list):
            for i, s in enumerate(self.schedulers): s.load_state_dict(state[i])
        else:
            self.schedulers.load_state_dict(state)
    
    def fit(self, epochs: int) -> None:
        """The main training loop. Trains the model for the given number of epochs.

        Args:
            epochs (int): Max epochs to run.
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

    def save_checkpoint(self, epoch: int, filename: str='latest.pth') -> None:
        """Save a checkpoint of the model, optimizer states and scheduler states.

        Args:
            epoch (int): Current epoch.
            filename (str, optional): Filename to save the checkpoint as. Defaults to 'latest.pth'.
        """
        
        save_dir = Path(self.logger.log_dir) / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            "epoch": epoch,
            "best_metrics": self.best_metrics,
            # Handle Single vs dict Models
            "model_state": self.model.state_dict() if not isinstance(self.model, dict) 
                           else {k: v.state_dict() for k, v in self.model.items()},
            # Handle Single vs dict Optimizers
            "optimizer_state": self.optimizers.state_dict() if not isinstance(self.optimizers, dict) 
                               else {k: v.state_dict() for k, v in self.optimizers.items()},
            "scheduler_state": self._get_scheduler_state()
        }
        # 1. ALWAYS save 'latest' or custom-name (for resuming/metric-based checkpointing)
        torch.save(state, save_dir / filename)
        
        if filename == 'latest.pth':
            # PERIODICALLY save history (for analysis)        
            if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
                history_filename = f"checkpoint-{epoch + 1:04d}.pth"
                torch.save(state, save_dir / history_filename)
                self.logger.info(f"💾 Saved historical checkpoint: {history_filename}")

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from a saved checkpoint. Loads the model, optimizer and scheduler state dictionaries.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """

        self.logger.info(f"📍 Loading checkpoint from {checkpoint_path}")
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
            
        self.logger.info(f"✅ Resumed from Epoch {self.start_epoch}")
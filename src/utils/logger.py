import logging
import torch
import torchvision.utils as tvutils
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import wandb
import inspect

class UnifiedLogger:
    """
    A hybrid logger supporting TensorBoard, W&B, and standard File logging.
    
    This logger abstracts the experiment tracking backend, allowing the training 
    loop to remain clean and agnostic to the visualization tool used.
    """

    def __init__(
        self, 
        log_dir: str | Path,
        project_name: str | None,
        experiment_name: str,
        name: str = "LOGGER",
        config: dict | None = None,
        use_tensorboard: bool = False,
        use_wandb: bool = False
    ) -> None:
        """Initializes the UnifiedLogger with specified backends.

        Args:
            log_dir (str | Path): Base directory where logs will be stored.
            project_name (str | None): Name of the project for W&B logging.
            experiment_name (str):Unique name for the experiment to organize logs.
            name (str, optional): Name for the logger instance. Defaults to "LOGGER".
            config (dict | None, optional): Configuration dictionary to log to W&B. Defaults to None.
            use_tensorboard (bool, optional): Whether to enable TensorBoard logging. Defaults to True.
            use_wandb (bool, optional): Whether to enable Weights & Biases logging. Defaults to False.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_tb = use_tensorboard
        self.use_wandb = use_wandb
        
        self.name = name
        self.project_name = "New_Project" if not project_name else project_name


        # Standard Console/File Logger
        self._setup_standard_logging()

        # TensorBoard Setup
        self.tb_writer = None
        if self.use_tb:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tb"))
            self.info("✅ TensorBoard backend initialized.")

        # W&B Setup
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                dir=str(self.log_dir)
            )
            self.info("✅ Weights & Biases backend initialized.")

        self.info(f"📝 Logger Initialized.")

    def _setup_standard_logging(self) -> None:
        """Configures the standard Python logging module to write to file and console.

        Sets up a file handler for 'session.log' inside the log directory and a 
        stream handler for console output.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            encoding="utf-8",
            handlers=[
                logging.FileHandler(self.log_dir / "session.log", encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        self.console = logging.getLogger(self.name)

    def _get_caller_name(self) -> str:
        """
        Internal helper to find the name of the class or function calling the logger.
        It looks back 2 stack frames: 
        Frame 0: _get_caller_name (this function)
        Frame 1: info/debug/error (the wrapper function)
        Frame 2: The actual caller (e.g. Trainer.train_step)
        """
        try:
            # We want the caller of the wrapper function, so we look at index 2
            stack = inspect.stack()
            if len(stack) < 3:
                return self.name
            
            frame = stack[2][0]
            caller_self = frame.f_locals.get('self', None)
            
            if caller_self:
                return caller_self.__class__.__name__
            else:
                # If no 'self', it might be a standalone function or script
                return "Global"
        except Exception:
            # Fallback for safety
            return self.name
        
    def debug(self, msg: str) -> None:
        """Logs an debug message to the console and log file.

        Args:
            msg (str): The message to log.
        """
        caller = self._get_caller_name()
        self.console.debug(f"[{caller}] {msg}")
    
    def info(self, msg: str) -> None:
        """Logs an informational message to the console and log file.

        Args:
            msg (str): The message to log.
        """
        caller = self._get_caller_name()
        self.console.info(f"[{caller}] {msg}")

    def warning(self, msg: str) -> None:
        """Logs a warning message to the console and log file.

        Args:
            msg (str): The message to log.
        """
        caller = self._get_caller_name()
        self.console.warning(f"[{caller}] {msg}")

    def error(self, msg: str) -> None:
        """Logs an error message to the console and log file.

        Args:
            msg (str): The message to log.
        """
        caller = self._get_caller_name()
        self.console.error(f"[{caller}] {msg}")

    def critical(self, msg: str) -> None:
        """Logs a critical error message to the console and log file.

        Args:
            msg (str): The message to log.
        """
        caller = self._get_caller_name()
        self.console.critical(f"[{caller}] {msg}")

    def log_metrics(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        """Sends scalar metrics to all active backends.

        Args:
            metrics (dict[str, float]): A dictionary of metric names and their values (e.g., {'loss': 0.5}).
            step (int): The current global training step or epoch.
            prefix (str, optional): A string prefix to group metrics (e.g., 'train', 'val'). Defaults to "".
        """
        # Log to Console (Optional: filtering for specific intervals)
        if step % 10 == 0:
            self.info(f"Step {step} | {prefix} {metrics}")

        # Log to TensorBoard
        if self.tb_writer:
            for name, val in metrics.items():
                self.tb_writer.add_scalar(f"{prefix}/{name}", val, step)

        # Log to W&B
        if self.use_wandb:
            # W&B likes a flat dict; we add the prefix to the key
            wandb_metrics = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=step)

    def log_image(self, tag: str, images: list[torch.Tensor] | torch.Tensor, step: int, path:str | Path | None=None) -> None:
        """Logs an image or list of images to Disk, TensorBoard, and W&B.

        Args:
            tag (str): Identifier (e.g., 'val/prediction').
            images (list | Tensor): 
                - If List: Concatenates them horizontally and Normalizes [0-1].
                - If Tensor: Logs as-is (Assumes pre-processed/grid).
            step (int): Global step.
            path (str | Path, optional): Directory to save image to disk. Defaults to None.
        """
        # Prepare Image
        if isinstance(images, list):
            # Simple horizontal stack + normalize
            combined = torch.cat(images, dim=3)
            final_img = tvutils.make_grid(combined, nrow=1, normalize=True)
        else:
            # Tensor is already a grid or single image
            final_img = images
        
        # Log to Disk.
        if path:
            safe_tag = tag.replace("/", "_").replace(" ", "_")
            file_path = Path(path) / f"{step:06d}_{safe_tag}.png"
            tvutils.save_image(final_img, file_path)

        # Log to Backends
        if self.tb_writer:
            self.tb_writer.add_image(tag, final_img, step)

        if self.use_wandb:
            wandb.log({tag: [wandb.Image(final_img, caption=f"Step {step}")]}, step=step)

    def close(self) -> None:
        """Cleanly shutdown loggers.

        Closes the TensorBoard writer and finishes the W&B run if they are active.
        """
        if self.tb_writer:
            self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()
        self.info("📝 Logger sessions closed.")
import logging
import torch
import torchvision.utils as tvutils
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import wandb


class UnifiedLogger:
    """A hybrid logger supporting TensorBoard, W&B, and standard File logging.
    
    This logger abstracts the experiment tracking backend, allowing the training 
    loop to remain clean and agnostic to the visualization tool used.
    """

    def __init__(
        self, 
        log_dir: str | Path, 
        experiment_name: str, 
        name: str = "LOGGER",
        config: dict | None = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False
    ) -> None:
        """Initializes the UnifiedLogger with specified backends.

        Args:
            log_dir (str | Path): Base directory where logs will be stored.
            experiment_name (str): Unique name for the experiment to organize logs.
            config (Optional[Dict], optional): Configuration dictionary to log to W&B. Defaults to None.
            use_tensorboard (bool, optional): Whether to enable TensorBoard logging. Defaults to True.
            use_wandb (bool, optional): Whether to enable Weights & Biases logging. Defaults to False.
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_tb = use_tensorboard
        self.use_wandb = use_wandb
        self.name = name

        # 1. Standard Console/File Logger
        self._setup_standard_logging()

        # 2. TensorBoard Setup
        self.tb_writer = None
        if self.use_tb:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tb"))
            self.info("TensorBoard backend initialized.")

        # 3. W&B Setup
        if self.use_wandb:
            wandb.init(
                project="noise2noise-medical",
                name=experiment_name,
                config=config,
                dir=str(self.log_dir)
            )
            self.info("Weights & Biases backend initialized.")

    def _setup_standard_logging(self) -> None:
        """Configures the standard Python logging module to write to file and console.

        Sets up a file handler for 'session.log' inside the log directory and a 
        stream handler for console output.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_dir / "session.log"),
                logging.StreamHandler()
            ]
        )
        self.console = logging.getLogger(self.name)

    def info(self, msg: str) -> None:
        """Logs an informational message to the console and log file.

        Args:
            msg (str): The message to log.
        """
        self.console.info(msg)

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

    def log_images(self, tag: str, images: list[torch.Tensor], step: int) -> None:
        """Logs a horizontal grid of comparison images.

        Concatenates a list of tensors horizontally and logs them as a single image grid.
        Useful for visualizing [Input, Output, Target] side-by-side.

        Args:
            tag (str): Identifier for the image group (e.g., 'val/predictions').
            images (list[torch.Tensor]): List of image tensors to concatenate. Each should be [B, C, H, W].
            step (int): The current global training step.
        """
        # Concatenate images horizontally [B, 1, H, W] -> [1, H, W * N]
        combined = torch.cat(images, dim=3)
        grid = tvutils.make_grid(combined, nrow=1, normalize=True)

        if self.tb_writer:
            self.tb_writer.add_image(tag, grid, step)

        if self.use_wandb:
            wandb.log({tag: [wandb.Image(grid, caption=f"Step {step}")]}, step=step)

    def close(self) -> None:
        """Cleanly shutdown loggers.

        Closes the TensorBoard writer and finishes the W&B run if they are active.
        """
        if self.tb_writer:
            self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()
        self.info("Logger sessions closed.")
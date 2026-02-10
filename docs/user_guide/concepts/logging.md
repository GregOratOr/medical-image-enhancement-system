# Experiment Tracking

In deep learning research, effectively tracking metrics (Loss, PSNR) and visualizing results (Noisy vs. Denoised images) is crucial. Instead of hard-coding a specific tool like TensorBoard or Weights & Biases (W&B) into the training loop, this project uses a **Unified Logger**.

## The Abstraction Layer

The `UnifiedLogger` acts as a facade, decoupling the experiment logic from the visualization backend. This allows you to:
1.  **Switch Backends Easily:** Enable or disable TensorBoard/W&B with a simple boolean flag in the config, without changing a single line of training code.
2.  **Centralized Formatting:** Ensure that metrics are logged consistently (e.g., same step count, same naming convention) across all platforms.
3.  **Local Fallback:** If internet access is lost (breaking W&B), the logger automatically falls back to local TensorBoard and text logs.



## Comparison Grids
A key feature for image enhancement projects is the visual comparison. The logger automatically handles the creation of "Image Grids" during validation steps:

$$\text{Grid} = [ \text{Input Noisy} \quad | \quad \text{Model Output} \quad | \quad \text{Ground Truth} ]$$

This side-by-side comparison allows for instant qualitative assessment of the denoising performance.

## Usage
Initialize the logger once at the start of your script:

```python
logger = UnifiedLogger(
    log_dir="runs",
    experiment_name="n2n-experiment-01",
    use_tensorboard=True,
    use_wandb=True
)

# In your training loop:
logger.log_metrics({'loss': 0.05, 'psnr': 32.5}, step=global_step)

``` 
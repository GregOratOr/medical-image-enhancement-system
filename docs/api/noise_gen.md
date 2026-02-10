# Noise Generation API

Utilities for simulating various noise profiles (Gaussian, Poisson, Speckle) to create noisy-noisy pairs for training.

## Mathematical Implementations
We implement noise injection following the zero-mean requirement:

$$\hat{z} = z + \mathcal{N}(0, \sigma^2)$$

Where $\mathcal{N}$ represents the stochastic noise component that the model learns to ignore.

## Noise Modules

::: src.transforms.noise
    options:
        show_root_heading: true
        show_source: false


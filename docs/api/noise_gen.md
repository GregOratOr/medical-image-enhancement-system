# Noise Generation API

Utilities for simulating various noise profiles to create noisy–noisy pairs for Noise2Noise training.

---

## Mathematical note

Noise injection typically follows the zero-mean requirement:

$$\hat{z} = z + \mathcal{N}(0, \sigma^2)$$

Where \(\mathcal{N}\) represents the stochastic noise component that the model learns to ignore.

---

## Noise modules

::: src.transforms.noise
    options:
      show_root_heading: true
      show_source: false


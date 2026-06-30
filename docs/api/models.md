# Models

Technical reference for the deep learning models used in this project. The architecture is primarily based on the **U-Net** design, optimized for the **Noise2Noise** self-supervised training objective.

---

## Primary Model
The modern, flexible implementation used for the majority of experiments. It allows for dynamic depth configuration and automatic channel scaling.


### `Noise2Noise`
::: src.models.noise2noise.Noise2Noise
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward

---

## Reference Model
This class strictly follows the architecture described by **Lehtinen et al. (2018)** in the original Noise2Noise paper. It is useful for benchmarking and reproducing baseline results.

### `Noise2NoiseOriginal`
::: src.models.noise2noise.Noise2NoiseOriginal
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward

---

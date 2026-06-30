# Medical Image Quality Enhancement Documentation

Welcome to the documentation for the **Real-time Medical Image Quality Enhancement System**. This project leverages deep learning, specifically **Noise2Noise** architectures, to denoise CT scans and improve diagnostic clarity without requiring clean ground-truth data.

---

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Get Started**
    ---
    Learn how to set up your environment using Docker and WSL, and run your first denoising inference.
    
    [:octicons-arrow-right-24: Installation](user_guide/installation.md)  
    [:octicons-arrow-right-24: Quickstart](tutorials/training.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**
    ---
    Understand the core concepts behind Noise2Noise training and our CT-specific denoising pipeline.
    
    [:octicons-arrow-right-24: Noise2Noise Theory](user_guide/concepts/noise2noise.md)  
    [:octicons-arrow-right-24: Deployment Guide](user_guide/deployment/real_time.md)

-   :material-code-braces:{ .lg .middle } **API Reference**
    ---
    Technical documentation for models, datasets, and image processing utilities.
    
    [:octicons-arrow-right-24: Models](api/models.md)  
    [:octicons-arrow-right-24: Noise Generation](api/noise_gen.md)

-   :material-school:{ .lg .middle } **Tutorials**
    ---
    Step-by-step guides for training models on custom datasets and evaluating image quality.
    
    [:octicons-arrow-right-24: Training Guide](tutorials/training.md)  
    [:octicons-arrow-right-24: Quality Evaluation](tutorials/evaluation.md)

</div>

---

## Key Features

* **Noise2Noise Implementation:** Efficient training using only noisy image pairs.
* **Real-time Performance:** Optimized for low-latency CT scan enhancement.
* **PyTorch Native:** Built entirely on the PyTorch ecosystem for modularity and speed.
* **Ready for Deployment:** Dockerized environment for seamless integration into medical imaging workflows.

## Core Objective

The mathematical foundation of this project is based on the observation that we can recover a signal $s$ from noisy observations $z_i = s + n_i$ by minimizing:

$$\arg\min_{\theta} \sum_{i} L(f_{\theta}(\hat{z}_i), z_i)$$

Where $\hat{z}_i$ and $z_i$ are two different noisy realizations of the same underlying signal.
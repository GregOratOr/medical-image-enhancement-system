# Datasets

Technical documentation for the data loading pipeline, optimized for high-throughput CT slice processing.

## Data Preparations

Contains Low-level utilities for dataset splitting and image-to-patch decomposition.

::: src.datasets.preparation
    options: 
        show_root_heading: true
        members:
            - split_dataset_blocks
            - image_to_patches

## Data Orchestration
The high-level pipeline that coordinates volume reading, slice processing, and directory management.

::: src.datasets.make_dataset
    options: 
        show_root_heading: true
        members:
            - prepare_dataset

## PyTorch Dataset & Augmentation
The final interface used by the PyTorch DataLoader.

### CTScans Dataset

The primary class for loading processed PNG images, optimized for high-throughput CT slice processing.

::: src.datasets.dataset.CTScans 
    options: 
        show_root_heading: true 
        members: 
            - __init__ 
            - __len__
            - __getitem__

### Medical Data Augmentation

Coming Soon...

<!-- 
Utilities for on-the-fly augmentation specific to medical imaging. 

This includes geometric transforms (rotations, flipping) and intensity transforms (windowing) to simulate different CT viewing levels.

::: src.transforms
    options:
        show_source: true 
        show_root_heading: false 
-->
# Image Processing API

Technical reference for the preprocessing and postprocessing utilities used throughout the dataset pipeline and inference workflows.

---

## Intensity normalization

Medical images (including CT) can have pixel intensities far outside 0–255. This project uses normalization/windowing utilities to map raw values into a stable grayscale range before saving PNG slices.

::: src.utils.image_operations.normalize_image
    options:
      show_root_heading: true
      show_source: true

---

## Slice processing pipeline

The dataset preparation pipeline uses `process_slice` to apply clipping/windowing and normalization before saving PNG slices.

::: src.utils.image_operations.process_slice
    options:
      show_root_heading: true
      show_source: true


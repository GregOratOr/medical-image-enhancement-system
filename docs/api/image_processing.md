# Image Processing API
**File:** `docs/api/image_processing.md`

Technical reference for the mathematical operations applied to raw medical scans.

## Intensity Normalization

Medical images, particularly CT scans, often contain pixel values (Hounsfield Units) that far exceed the standard 0-255 range of digital images. "Windowing" or normalization is the process of focusing on a specific range of these values—such as the range for soft tissue or bone—and mapping them to a visible spectrum.



### `normalize_image`
The core function that handles the conversion from high-dynamic-range float data to standard 8-bit integers.

::: src.utils.image_operations.normalize_image
    options:
      show_root_heading: true
      show_source: true
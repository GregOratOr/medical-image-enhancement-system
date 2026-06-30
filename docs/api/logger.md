# Experiment Logging API
**File:** `docs/api/logger.md`

This page provides the technical details of the `UnifiedLogger` class.

## Unified Logger
The core class that manages multiple logging backends simultaneously.

::: src.utils.logger.UnifiedLogger
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - info
        - log_metrics
        - log_images
        - close
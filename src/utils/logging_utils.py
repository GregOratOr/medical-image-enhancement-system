import torchvision.transforms as T

def get_noise_name(transform, delimiter=" + "):
    """Recursively extracts human-readable names from a Transform or Compose pipeline.

    Args:
        transform (callable or None): The transformation pipeline (e.g., torchvision.transforms.Compose) 
            or a single transform object. If None, returns "Clean (No Noise)".
        delimiter (str, optional): String used to join multiple transform names. Defaults to " + ".

    Returns:
        str: A formatted string representing the noise pipeline (e.g., "Spectral Blur + Gaussian Noise").
    """
    # Case 1: If it's None
    if transform is None:
        return "Clean (No Noise)"

    # Case 2: If it's a Compose pipeline
    if isinstance(transform, T.Compose):
        names = []
        for t in transform.transforms:
            # Check if it has our custom 'name' attribute
            if hasattr(t, 'name'):
                names.append(t.name)
            else:
                # Fallback to Class Name (e.g., "ToTensor")
                names.append(t.__class__.__name__)
        return delimiter.join(names)

    # Case 3: Single Transform
    if hasattr(transform, 'name'):
        return transform.name
    
    return transform.__class__.__name__
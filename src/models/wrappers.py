import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicPadWrapper(nn.Module):
    """
    Wraps a Fully Convolutional Network to ensure input spatial dimensions 
    are divisible by a specific multiple (e.g., 16 for a 4-pool U-Net).
    Dynamically pads the input before the forward pass and crops it back afterward.
    """
    def __init__(self, base_model: nn.Module, depth: int=4) -> None:
        """Initializes the Wrapper.

        Args:
            base_model (nn.Module): The model to wrap around.
            depth (int, optional): Number of pooling layers in the base model. Defaults to 4.
        """

        super().__init__()
        self.base_model = base_model
        self.multiple_of = 2 ** depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        pad_h = (self.multiple_of - (h % self.multiple_of)) % self.multiple_of
        pad_w = (self.multiple_of - (w % self.multiple_of)) % self.multiple_of
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        out = self.base_model(x)

        out = out[:, :, :h, :w]

        return out


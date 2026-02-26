# src/deploy/onnx_wrappers.py

import numpy as np
import onnxruntime as onnxrt

class ONNXPadWrapper:
    """
    NumPy-based inference wrapper for ONNX models.
    Dynamically pads odd-sized clinical images to satisfy the U-Net's 
    spatial divisibility requirements (e.g., multiples of 16), runs 
    the ONNX InferenceSession, and cleanly crops the output.
    """
    def __init__(self, onnx_session: onnxrt.InferenceSession , depth: int = 4):        
        self.session = onnx_session
        # Get the exact string name of the input node ONNX expects.
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Calculate divisibility requirement (2^4 = 16).
        self.multiple_of = 2 ** depth

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: A NumPy array of shape (Batch, Channels, Height, Width)
        Returns:
            A denoised NumPy array of the exact same shape.
        """
        # Intercept original dimensions.
        _, _, h, w = x.shape

        # Calculate required padding (dynamic math).
        pad_h = (self.multiple_of - (h % self.multiple_of)) % self.multiple_of
        pad_w = (self.multiple_of - (w % self.multiple_of)) % self.multiple_of

        # Apply NumPy Pad (only if needed).
        # (before, after) for each of the 4 dimensions.
        # Add padding only to the bottom and right edges. [(0, pad_h) and (0, pad_w)]
        if pad_h > 0 or pad_w > 0:
            x = np.pad(
                x, 
                pad_width=((0, 0), (0, 0), (0, pad_h), (0, pad_w)), 
                mode='reflect'
            )

        # Critical: Ensure data type is strictly Float32.
        x = x.astype(np.float32)

        # Forward pass through the pure ONNX model.
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        
        out = np.asarray(out)

        # Apply UnPad (crop back to exact original dimensions).
        out = out[:, :, :h, :w]

        return out
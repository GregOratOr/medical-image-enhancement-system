# onnx/inference_cpu.py
import numpy as np
import onnxruntime as ort
import sys
from pathlib import Path

# Add project root to path for imports if needed
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from onnx.onnx_wrappers import ONNXPadWrapper

class CPUInference:
    def __init__(self, onnx_path: str, is_wrapped_externally: bool = False):
        """
        Simple CPU Pipeline: Wraps the model with standard padding logic.
        Best for single images or when VRAM is not available.

        Args:
            onnx_path: Path to model
            is_wrapped_externally: If True, assumes the ONNX file itself handles padding.
                                   If False, applies Python-side padding wrapper.
        """
        print(f"🐢 Initializing Standard CPU Inference Pipeline (Wrapped in ONNX File: {is_wrapped_externally})...")
        
        # Force CPU Provider
        self.session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        
        self.is_wrapped_externally = is_wrapped_externally

        # Use your existing Wrapper class
        if not self.is_wrapped_externally:
            self.wrapper = ONNXPadWrapper(onnx_session=self.session, depth=4)
        else:
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        

    def predict(self, full_image: np.ndarray) -> np.ndarray:
        """
        Args: full_image (1, 1, H, W)
        """
        if not self.is_wrapped_externally:
            # The wrapper handles padding -> inference -> cropping internally
            return self.wrapper(full_image)
        else:
            # Model has built-in padding wrapper.
            x = full_image.astype(np.float32)
            
            out = self.session.run([self.output_name], {self.input_name:x})[0]

            return np.asarray(out)
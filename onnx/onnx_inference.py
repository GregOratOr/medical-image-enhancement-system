# Execute Inference pipeline with base_model.onnx
# onnx/onnx_inference.py

import sys
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import onnxruntime as onnxrt

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from onnx_wrappers import ONNXPadWrapper

class Inference:
    def __init__(self, model_path: Path | str, input_dir: Path | str, output_dir: Path | str, model_name: str='model.onnx', wrapped_model=True, enable_cuda=False, **kwargs) -> None:
        self.model_path = Path(model_path)
        self.ip_dir = Path(input_dir)
        self.op_dir = Path(output_dir)
        self.wrapped_model = wrapped_model
        self.kwargs = kwargs
        self.model_name = model_name
        self.model_path /= model_name
        self.op_dir.mkdir(parents=True, exist_ok=True)

        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if enable_cuda else ['CPUExecutionProvider']
        
        print("🚀 Initializing Pure-ONNX Inference Pipeline...")
        print(f"🚀 Initializing ONNX Runtime Session from {str(self.model_path)}...")
        self.session = onnxrt.InferenceSession(
            self.model_path,
            providers = self.providers,
            # sess_options=options
        )
    
    def run(self):
        if self.wrapped_model:
            self.kwargs["input_name"] = self.session.get_inputs()[0].name
            self.kwargs["output_name"] = self.session.get_outputs()[0].name
        else:
            self.kwargs["base_model"] = ONNXPadWrapper(onnx_session=self.session, depth=4)
        
        # Grab images
        valid_extensions = {'.png', '.jpg', '.jpeg', '.tif'}
        image_paths = [p for p in self.ip_dir.rglob("*") if p.suffix.lower() in valid_extensions]
        
        if not image_paths:
            raise FileNotFoundError(f"No valid images found in {self.ip_dir.absolute()}")
            
        print(f"📦 Found {len(image_paths)} image(s). Starting NumPy denoising...")
        
        # The Inference Loop
        start_time = time.time()
        
        for img_path in tqdm(image_paths, desc="ONNX Processing"):
            # Pre-process (NumPy)
            x = self.process_image(img_path)
            
            # Forward Pass (ONNX Runtime + NumPy Pad/Crop)
            y = self.session.run([self.kwargs["output_name"]], {self.kwargs["input_name"]: x})[0] if self.wrapped_model else self.kwargs["base_model"](x)
            
            # Post-process & Save (NumPy to PIL)
            save_path = self.op_dir / f"onnx_denoised_{img_path.name}"
            self.save_image(y, save_path) # type: ignore

            del y
            
        total_time = time.time() - start_time
        fps = len(image_paths) / total_time
        
        print(f"✅ Pipeline Complete! Saved to {self.op_dir.absolute()}")
        print(f"⚡ Speed: {fps:.2f} frames per second.")
        print(f"🕰️ Avg time per image = {total_time/len(image_paths):.2f}.")
        print(f"🕰️ Inference Time: {total_time:.2f}.")


    def process_image(self, img_path: Path) -> np.ndarray:
        """Loads an image using PIL and formats it exactly like PyTorch's ToTensor()"""
        # Load and convert to grayscale
        img = Image.open(img_path).convert("L")

        # img = img.resize((2048,2048), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and scale to [0.0, 1.0] (float32)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add Batch and Channel dimensions: (H, W) -> (1, 1, H, W)
        img_array = np.expand_dims(img_array, axis=0) # Add Channel
        img_array = np.expand_dims(img_array, axis=0) # Add Batch
        
        return img_array

    def save_image(self, tensor_array: np.ndarray, output_path: Path):
        """Converts a (1, 1, H, W) float32 array back to a standard PNG image"""
        # Squeeze out the Batch and Channel dimensions: (1, 1, H, W) -> (H, W)
        img_array = np.squeeze(tensor_array)
        
        # Clamp values strictly between 0.0 and 1.0
        img_array = np.clip(img_array, 0.0, 1.0)
        
        # Scale back to [0, 255] and convert to 8-bit integer
        img_array = (img_array * 255.0).astype(np.uint8)
        
        # Save using PIL
        Image.fromarray(img_array, mode="L").save(output_path)

def main():
    obj = Inference(
        model_path="onnx/models",
        input_dir="data/processed/test/sample_images",
        output_dir="inferences/onnx/outputs",
        model_name="medical_denoiser_base.onnx",
        wrapped_model=False,
        enable_cuda=False
    )

    obj.run()

if __name__ == "__main__":
    main()
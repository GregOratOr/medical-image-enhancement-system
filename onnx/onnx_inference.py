# onnx/onnx_inference.py

import sys
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from inference_gpu import GPUInference
from inference_cpu import CPUInference

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

class InferenceManager:
    def __init__(self, model_path: str, model_name: str, input_dir: str, output_dir: str, enable_cuda: bool=False, wrapped_model: bool=False) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = Path(model_path) / f"{model_name}.onnx"

        if enable_cuda:
            self.pipeline = GPUInference(
                onnx_path=self.model_path.__str__(),
                tile_size=512,
                overlap_ratio=0.5,
                batch_size=4,
                is_wrapped_externally=wrapped_model
            )
        else:
            self.pipeline = CPUInference(onnx_path=self.model_path.__str__(), is_wrapped_externally=wrapped_model)

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
    
    def run(self):
        valid_extensions = {'.png', '.jpg', '.jpeg', '.tif'}
        image_paths = [p for p in self.input_dir.rglob("*") if p.suffix.lower() in valid_extensions]
        
        print(f"📦 Found {len(image_paths)} images.")

        # The Inference Loop
        start_time = time.time()

        for img_path in tqdm(image_paths, desc="Processing"):
            # Load and match dims.
            x = self.process_image(img_path)
            
            # Predict (Polymorphic call)
            y = self.pipeline.predict(x)
            
            # Save output to disk
            save_path = self.output_dir / f"onnx_denoised_{img_path.name}"
            self.save_image(y, save_path)
            
            # Cleanup
            del x, y
        
        total_time = time.time() - start_time
        fps = len(image_paths) / total_time
        
        print(f"✅ Pipeline Complete! Saved to {self.output_dir.absolute()}")
        print(f"⚡ Speed: {fps:.2f} frames per second.")
        print(f"🕰️  Avg time per image = {total_time/len(image_paths):.2f}.")
        print(f"🕰️  Inference Time: {total_time:.2f}.")



if __name__ == "__main__":
    # Toggle this to switch pipelines!
    ENABLE_CUDA = True 

    # CONFIGURATION
    MODEL_PATH = "onnx/models"
    MODEL_NAME = "medical_denoiser_dyno"
    INPUT_DIR = "data/processed/test/sample_images"
    OUTPUT_DIR = "inferences/onnx/outputs" + ("_GPU" if ENABLE_CUDA else "_CPU")
    
    manager = InferenceManager(MODEL_PATH, MODEL_NAME, INPUT_DIR, OUTPUT_DIR, ENABLE_CUDA, wrapped_model=False)
    manager.run()
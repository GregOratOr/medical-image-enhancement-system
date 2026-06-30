import sys
import io
import gc
import uvicorn
import numpy as np
import onnxruntime as onnxrt
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Response, HTTPException, Query
from contextlib import asynccontextmanager

# Setup Path to import your custom modules
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from onnx.onnx_inference import InferenceManager
SYSTEM_HAS_GPU = 'CUDAExecutionProvider' in onnxrt.get_available_providers()

# CONFIGURATION
DEFAULT_WRAPPED = False
DEFAULT_CUDA = SYSTEM_HAS_GPU
MODEL_DIR = root_dir / "onnx" / "models"
DEFAULT_MODEL = "medical_denoiser_dyno"
INPUT_DIR = "data/processed/test/sample_images"
OUTPUT_DIR = "inferences/onnx/outputs" + ("_GPU" if DEFAULT_CUDA else "_CPU")

manager: InferenceManager | None = None
current_config = {
    "model_name": DEFAULT_MODEL,
    "enable_cuda": DEFAULT_CUDA,
    "wrapped_model": DEFAULT_WRAPPED,
    "has_gpu": SYSTEM_HAS_GPU
}

class ConfigRequest(BaseModel):
    model_name: str
    enable_cuda: bool
    wrapped_model: bool

@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    print(f"🚀 API Startup: Loading {DEFAULT_MODEL}...")
    
    load_engine(DEFAULT_MODEL, DEFAULT_CUDA, DEFAULT_WRAPPED)
    yield 
    
    # SHUTDOWN
    print("🛑 API Shutdown: Cleaning up resources...")

    # If InferenceManager had a .close() method, you would call it here.
    if manager: del manager
    gc.collect()

def load_engine(model_name: str, enable_cuda: bool, wrapped_model: bool):
    """Helper to safely swap the backend engine."""
    global manager, current_config
    
    try:
        if manager is None:
            # Initialize the manager
            manager = InferenceManager(
                model_path=str(MODEL_DIR),
                model_name=model_name,
                input_dir="tmp_in", 
                output_dir="tmp_out", 
                enable_cuda=enable_cuda,
                wrapped_model=wrapped_model
            )
        else:
            # HOT SWAP using the new method
            manager.update_engine(model_name, enable_cuda, wrapped_model)
        
        # Update global config tracker
        current_config["model_name"] = model_name
        current_config["enable_cuda"] = enable_cuda
        current_config["wrapped_model"] = wrapped_model
        
        print(f"✅ Engine Swapped: {model_name} | CUDA: {enable_cuda} | Wrapped: {wrapped_model}")
        return True
    except Exception as e:
        print(f"❌ Load Failed: {e}")
        raise e

def preprocess_bytes(image_bytes: bytes) -> np.ndarray:
    """Bytes -> (1, 1, H, W) float32 array"""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img_array = np.array(img, dtype=np.float32) / 255.0
    # Add Batch and Channel dims: (H, W) -> (1, 1, H, W)
    return np.expand_dims(np.expand_dims(img_array, 0), 0)

def postprocess_array(output_tensor: np.ndarray) -> bytes:
    """(1, 1, H, W) float32 -> PNG Bytes"""
    # Squeeze to (H, W)
    img_array = np.squeeze(output_tensor)
    img_array = np.clip(img_array, 0.0, 1.0)
    img_array = (img_array * 255.0).astype(np.uint8)
    
    # Save to buffer
    buffer = io.BytesIO()
    Image.fromarray(img_array, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


# Initialize the App
app = FastAPI(
    title="Medical Image Denoiser API",
    lifespan=lifespan,
    description="An inference server for ONNX runtime.",
    version="1.0.0"
)


@app.get("/")
def health_check():
    return {
        "status": "online", 
        "active_config": current_config
    }

@app.post("/set_config")
def set_config(config: ConfigRequest):
    """Explicitly updates the running model."""
    # Don't reload if nothing changed
    if (config.model_name == current_config["model_name"] and 
        config.enable_cuda == current_config["enable_cuda"] and
        config.wrapped_model == current_config["wrapped_model"]):
        return {"message": "Config already active", "config": current_config}

    try:
        load_engine(config.model_name, config.enable_cuda, config.wrapped_model)
        return {"message": "Engine updated successfully", "config": current_config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if manager is None:
        print("⚠️ Model was unloaded. Auto-reloading before inference...")
        
        # Figure out what model to load (fallback to default if it says "Unloaded")
        target_model = current_config.get("model_name", "None (Unloaded)")
        if target_model in [None, "None (Unloaded)"]: target_model = DEFAULT_MODEL
            
        target_cuda = current_config.get("enable_cuda", DEFAULT_CUDA)
        target_wrapped = current_config.get("wrapped_model", DEFAULT_WRAPPED)
        
        # Reload the engine!
        try:
            load_engine(target_model, target_cuda, target_wrapped)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to auto-reload model: {e}")
    
    assert manager is not None, "Inference Manager is Not-initialized."
    assert manager.pipeline is not None, "Inference Pipeline is Not-initialized."
    
    input_tensor, output_tensor = None, None

    try:
        # READ: Get the raw bytes from the uploaded file
        contents = await file.read()
        
        # PREPROCESS: Bytes -> (1, 1, H, W) float32 array in memory.
        input_tensor = preprocess_bytes(contents)
        
        # # INFERENCE: Delegated to Inference Manager.
        # Automatically handles Tiling (GPU) or Padding (CPU)
        output_tensor = manager.pipeline.predict(input_tensor)
        
        # POSTPROCESS: Convert (1, 1, H, W) float32 back to PNG bytes
        result_bytes = postprocess_array(output_tensor)
        
        return Response(content=result_bytes, media_type="image/png")
    
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # CLEANUP PROTOCOL (Runs even if code crashes)
        if input_tensor is not None: del input_tensor
        if output_tensor is not None:del output_tensor

        gc.collect()

@app.post("/unload")
def unload_engine():
    """Destroys the active inference manager and frees GPU VRAM."""
    global manager, current_config
    
    if manager is not None:
        del manager
        manager = None
        
    # Update config so the frontend knows it's empty
    current_config["model_name"] = "None (Unloaded)"
    
    # Force Python to clean up the deleted objects
    gc.collect()
    
    print("🧹 Model unloaded and VRAM freed!")
    return {"message": "VRAM successfully cleared."}


@app.post("/error_map")
async def error_map(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    img1_array = None
    img2_array = None
    diff = None
    heatmap = None

    try:
        image1_bytes = await image1.read()
        image2_bytes = await image2.read()

        img1 = Image.open(io.BytesIO(image1_bytes)).convert("L")
        img2 = Image.open(io.BytesIO(image2_bytes)).convert("L")
        img1_array = np.array(img1, dtype=np.float32)
        img2_array = np.array(img2, dtype=np.float32)

        diff = np.abs(img1_array - img2_array)
        heatmap = np.clip(diff * 1000, 0, 255)

        error_pil = Image.fromarray(heatmap.astype(np.uint8), mode="L")
        buffer = io.BytesIO()
        error_pil.save(buffer, format="PNG")
        
        return Response(content=buffer.getvalue(), media_type="image/png")
    
    finally:
        del img1_array, img2_array, diff, heatmap

        gc.collect()
        print("🧹 Memory cleaned up!")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
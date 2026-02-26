import sys
import io
import gc
import numpy as np
import onnxruntime as onnxrt
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Response

# Setup Path to import your custom modules
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Import your actual model wrapper
from onnx.onnx_wrappers import ONNXPadWrapper

# Initialize the App
app = FastAPI(
    title="Medical Image Denoiser",
    description="A CPU-based ONNX inference server.",
    version="1.0.0"
)

# Load the Model Globally (The "Brain")
# This runs at server startup ONLY!
print("🚀 Loading ONNX Model into memory...")
MODEL_PATH = root_dir / "onnx" / "models" / "medical_denoiser_dyno.onnx"

# Create Inference Session. CPU pipeline since it's stable
session = onnxrt.InferenceSession(
            str(MODEL_PATH),
            providers = ['CPUExecutionProvider']
        )

denoiser = ONNXPadWrapper(onnx_session=session, depth=4)
print("✅ Model loaded and ready!")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Medical Denoiser API is ready."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    1. Reads uploaded image bytes.
    2. Preprocesses (Bytes -> PIL -> NumPy).
    3. Runs ONNX Inference.
    4. Postprocesses (NumPy -> PIL -> Bytes).
    5. Returns the denoised image.
    """
    img_array = None
    input_tensor = None
    output_tensor = None
    out_array = None

    try:
        # READ: Get the raw bytes from the uploaded file
        image_bytes = await file.read()
        
        # PREPROCESS: Convert bytes to (1, 1, H, W) float32 array
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add Batch and Channel dimensions
        input_tensor = np.expand_dims(img_array, axis=0) # (H, W) -> (1, H, W)
        input_tensor = np.expand_dims(input_tensor, axis=0) # (1, H, W) -> (1, 1, H, W)
        
        # INFERENCE: Run the model
        # Since this is CPU, it might take ~10-30 seconds for large images
        output_tensor = denoiser(input_tensor)
        
        # POSTPROCESS: Convert (1, 1, H, W) float32 back to PNG bytes
        out_array = np.squeeze(output_tensor)             # Remove batch/channel dims
        out_array = np.clip(out_array, 0.0, 1.0)          # Force values to [0, 1]
        out_array = (out_array * 255.0).astype(np.uint8)  # Scale to [0, 255]
        
        # Convert back to PIL Image
        out_pil = Image.fromarray(out_array, mode="L")
        
        # Save PIL image to a byte buffer (like writing to a fake file in RAM)
        buffer = io.BytesIO()
        out_pil.save(buffer, format="PNG")
        
        # Get the raw bytes
        processed_bytes = buffer.getvalue()
        
        # RETURN: Send the bytes back with the correct media type
        return Response(content=processed_bytes, media_type="image/png")


    finally:
        # CLEANUP PROTOCOL (Runs even if code crashes)
        del img_array
        del input_tensor
        del output_tensor
        del out_array

        gc.collect()
        print("🧹 Memory cleaned up!")


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

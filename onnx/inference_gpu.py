# onnx/inference_gpu.py
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from onnx.onnx_wrappers import ONNXPadWrapper

class GPUInference:
    def __init__(self, onnx_path: str, tile_size: int = 512, overlap_ratio: float = 0.25, batch_size: int = 4,is_wrapped_externally: bool = False):
        """
        Specialized pipeline for GPU: Handles Tiling and Batching.
        """
        print(f"⚡ Initializing GPU Pipeline (Tiled Inference)...")
        
        # Force CUDA Provider
        self.session = ort.InferenceSession(str(onnx_path), providers=['CUDAExecutionProvider'])
        
        # Verify we actually got CUDA (or crash early)
        if 'CUDAExecutionProvider' not in self.session.get_providers():
            raise RuntimeError("❌ CUDA requested but not available! Check your drivers/onnxruntime-gpu.")

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        self.is_wrapped_externally = is_wrapped_externally
        if not self.is_wrapped_externally:
            # If raw model, then wrap it to handle the odd tile sizes (e.g. 500x500)
            self.model_wrapper = ONNXPadWrapper(onnx_session=self.session, depth=4)


        self.tile_size = tile_size
        self.batch_size = batch_size
        
        # Calculate Stride & Margins
        # Overlap = Total pixels shared between two tiles
        # Margin = The strip we discard from ONE side of a tile (Overlap / 2)
        self.overlap_px = int(tile_size * overlap_ratio)
        if self.overlap_px % 2 != 0: self.overlap_px += 1 # Ensure even number for symmetric cropping
        
        self.margin = self.overlap_px // 2
        self.stride = tile_size - self.overlap_px
        
        print(f"   🧩 Configuration: Tile={tile_size}, Stride={self.stride}, Margin={self.margin}")

    def predict(self, full_image: np.ndarray) -> np.ndarray:
        """
        Args: full_image (1, 1, H, W) or (H, W) normalized float32
        """
        # Squeeze to 2D (H, W) for slicing logic
        img = np.squeeze(full_image)
        h, w = img.shape
        
        # Pad image to be divisible by stride
        # Pad enough so that the "Valid Region" of the last tile reaches the edge
        pad_h = (self.stride - (h % self.stride)) % self.stride + self.overlap_px
        pad_w = (self.stride - (w % self.stride)) % self.stride + self.overlap_px
        
        img_padded = np.pad(img, ((self.margin, pad_h), (self.margin, pad_w)), mode='reflect')
        pad_h_total, pad_w_total = img_padded.shape
        
        # Output Canvas (un-padded size)
        output_canvas = np.zeros((pad_h_total, pad_w_total), dtype=np.float32)
        
        # Generate Tile Coordinates
        tiles = []
        write_coords = [] # coordinates to VALID regions on the canvas
        
        # Loop over the padded image with the defined stride
        for y in range(0, pad_h_total - self.tile_size + 1, self.stride):
            for x in range(0, pad_w_total - self.tile_size + 1, self.stride):
                # Extract Input Tile
                tile = img_padded[y:y+self.tile_size, x:x+self.tile_size]
                
                # Add Batch/Channel dims: (1, 1, Tile, Tile)
                tile = np.expand_dims(tile, 0)
                tile = np.expand_dims(tile, 0)
                tiles.append(tile)
                
                # Calculate where the VALID Center goes in the output canvas
                # The valid center starts at (y + margin, x + margin)
                write_coords.append((y + self.margin, x + self.margin))

        # Batch Inference
        for i in tqdm(range(0, len(tiles), self.batch_size), desc="GPU Tiling"):
            batch_input = np.concatenate(tiles[i:i+self.batch_size], axis=0)
            
            # Run CUDA Inference
            if self.is_wrapped_externally:
                batch_output = self.session.run([self.output_name], {self.input_name: batch_input})[0]
                batch_output = np.asarray(batch_output)
            else:
                batch_output = self.model_wrapper(batch_input)

            # Stitching
            current_write_coords = write_coords[i:i+self.batch_size]
            for j, (wy, wx) in enumerate(current_write_coords):
                # Extract the Valid Center from the prediction
                # We crop 'margin' from all 4 sides
                pred_tile = batch_output[j, 0, :, :]
                valid_center = pred_tile[self.margin : -self.margin, self.margin : -self.margin]
                
                # Write to canvas
                output_canvas[wy : wy+self.stride, wx : wx+self.stride] = valid_center

        # Final Crop (Remove the padding we added at the top)
        final_output = output_canvas[self.margin : self.margin+h, self.margin : self.margin+w]
        
        return np.expand_dims(np.expand_dims(final_output, 0), 0)
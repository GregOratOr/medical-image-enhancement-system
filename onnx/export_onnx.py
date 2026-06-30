import torch
from pathlib import Path
from torch.export import Dim
from src.models.noise2noise import Noise2Noise
from src.models.wrappers import DynamicPadWrapper

def export_to_onnx(checkpoint_path: str, output_path: str, model_name: str="model.onnx", use_wrapper: bool=False, use_dynamo: bool=True):
    device = torch.device('cpu') # Using gpu may lead to compatibility issues with different hardware.
    chkpt_path = Path(checkpoint_path)
    opt_path = Path(output_path)
    opt_path.mkdir(parents=True, exist_ok=True)

    print("🚀 Initializing ONNX Export (Wrapper={use_wrapper}, Dynamo={use_dynamo})...")

    # Initialize the Base model.
    model = Noise2Noise(
                    in_channels=1, 
                    out_channels=1, 
                    base_channels=48, 
                    depth=4, 
                    activation='leaky_relu'
                    )
    # Load weights.
    print(f"📂 Loading weights from {chkpt_path.name}...")
    checkpoint = torch.load(chkpt_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)

    model.eval()

    if use_wrapper:
        print("🛡️  Wrapping model with DynamicPadWrapper...")
        model = DynamicPadWrapper(base_model=model, depth=4)
        model.eval()

    # Dummy input Tensor.
    dummy_input = torch.randn(1, 1, 256,256, device=device)

    export_args = {
        "model": model,
        "args": (dummy_input,),
        "f": opt_path / f"{model_name}.onnx",
        "input_names": ["input"],
        "output_names": ["output"],
        "opset_version": 18,
        "dynamo": use_dynamo, 
        "external_data": False
    }

    if use_dynamo:
        print(f"📦 Tracing (Dynamo) and exporting to {output_path}...")
        # Define explicit dynamic dimensions with safe minimums and maximums
        batch_dim = Dim("batch", min=1, max=16)
        height_dim = Dim("height", min=16, max=4000) # Covers your 2570px images easily
        width_dim = Dim("width", min=16, max=4000)
        
        # Map dimensions -> input tensor.
        export_args['dynamic_shapes'] = {"x":{0: batch_dim, 2: height_dim, 3: width_dim}}
    else:
        print(f"📦 Tracing (Standard) and exporting to {output_path}...")
        export_args['dynamic_axes'] = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"}
        }
    
    # Export!
    try:
        onnx_program = torch.onnx.export(**export_args)
        print("✅ ONNX Export Complete!")
    except Exception as e:
        print(f"❌ Export Failed: {e}")

    

if __name__ == "__main__":
    ckpt = "./experiments/exp-01-testing-pipeline-2026-02-14/checkpoints/checkpoint-0100.pth"
    out = "./onnx/models"
    export_to_onnx(ckpt, out, model_name="medical_denoiser_dyno", use_dynamo=True, use_wrapper=False)
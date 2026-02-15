import torch
import torchvision.transforms as T
from configs.config import Optimizer, Scheduler, Model
from src.transforms.noise import NOISE_REGISTRY
from src.models.noise2noise import Noise2Noise, Noise2NoiseOriginal
from typing import Any

def resolve_device(use_gpu: bool) -> torch.device:
    """Safely resolves the best available hardware accelerator."""
    if not use_gpu:
        return torch.device("cpu")

    # 1. NVIDIA / AMD (ROCm)
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # 2. Apple Silicon (Mac M-series)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    
    # 3. Intel GPUs (XPU)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    
    # 4. Fallback
    else:
        print("⚠️ GPU requested but no compatible hardware found. Falling back to CPU.")
        return torch.device("cpu")


def resolve_noise_transforms(noise_list: list[dict]) -> T.Compose:
    """Factory: dynamically builds the noise pipeline from config."""

    transforms = []
    if noise_list:
        for noise_item in noise_list:
            noise_type = noise_item.get("type")
            params = noise_item.get("params", {})

            if not noise_type:
                continue
            
            noise_type = noise_type.lower()

            if noise_type not in NOISE_REGISTRY:
                raise ValueError(f"Unknown noise type: '{noise_type}'. Available: {list(NOISE_REGISTRY.keys())}")

            NoiseClass = NOISE_REGISTRY[noise_type]

            try:
                transforms.append(NoiseClass(**params))
                print(f"➕ Added Noise: {noise_type}")
            except TypeError as e:
                raise TypeError(f"Error initializing {noise_type}: {e}. Check your YAML params!")
    
    return T.Compose(transforms)


def resolve_models(model_configs: list[Model], device: torch.device) -> dict[str, torch.nn.Module]:
    """Instantiates models based on the config and moves them to the target device.
    
    Args:
        model_configs (list[Model]): List of model config data objects.
        device (torch.device): Target device for the models.

    Returns:
       dict[str, nn.Module]: A dictionary of models mapped by their name/label.
    """
    
    MODEL_REGISTRY = {
        "n2n-new": Noise2Noise,
        "n2n-orig": Noise2NoiseOriginal,
        # "discriminator": PatchGAN,  # Easy to add more later
    }

    models_dict = {}
    if model_configs:
        for model in model_configs:
            if model.name not in MODEL_REGISTRY:
                raise ValueError(
                    f"❌ Model '{model.name}' not found in registry. "
                    f"⚠️ Available models: {list(MODEL_REGISTRY.keys())}"
                )
                
            ModelClass = MODEL_REGISTRY[model.name]
            
            print(f"🧠 Instantiating Model: '{model.name}'")
            
            # 2. Instantiate using **model_params and move to device
            try:
                model_instance = ModelClass(**model.model_params).to(device)
                models_dict[model.name] = model_instance
            except TypeError as e:
                raise TypeError(f"❌ Error initializing model '{model.name}': {e}. Check your YAML model_params!")
                
    return models_dict


def resolve_optimizers(models_dict: dict[str, torch.nn.Module], optimizer_configs: list[Optimizer]) -> dict[str, torch.optim.Optimizer]:
    """Factory: Returns a dictionary of Optimizers."""
    
    optimizers = {}

    if optimizer_configs:
        for optimizer in optimizer_configs:
            # Copy of params to avoid mutating the original config
            kwargs = optimizer.params.copy()

            # Extract the model linkage
            model_label = kwargs.pop("model_label", None)
            
            if not model_label:
                raise ValueError(
                    f"❌ Optimizer '{optimizer.label}' is missing 'model_label' in its params. "
                    "⚠️ It needs to know which model to optimize!"
                )
                
            if model_label not in models_dict:
                raise KeyError(
                    f"❌ Model '{model_label}' requested by optimizer '{optimizer.label}' "
                    f"was not found. ⚠️ Available models: {list(models_dict.keys())}"
                )

            target_model = models_dict[model_label]

            # Fetch the Optimizer class dynamically from PyTorch
            try:
                opt_class = getattr(torch.optim, optimizer.name)
            except AttributeError:
                raise ValueError(f"Optimizer '{optimizer.name}' is not a valid torch.optim class.")

            # 4. Instantiate
            try:
                opt_instance = opt_class(target_model.parameters(), **kwargs)
                optimizers[optimizer.label] = opt_instance
                print(f"🔧 Resolved Optimizer: '{optimizer.label}' ({optimizer.name}) -> Linked to Model: '{model_label}'")
            except TypeError as e:
                raise TypeError(f"❌ Error initializing optimizer '{optimizer.name}': {e}. Check your YAML params!")

    return optimizers


def resolve_schedulers(optimizers_dict: dict[str, torch.optim.Optimizer], scheduler_configs: list[Scheduler]) -> dict[str, Any]:
    """Factory: Dynamically builds schedulers and resolves nested dependencies (like ChainedScheduler or SequentialLR).
    
    Args:
        optimizers: Dictionary of instantiated optimizers, e.g., {"main": opt_instance}
        sched_configs: List of Scheduler config objects from the YAML.
    """
    
    schedulers_map = {cfg.label: cfg for cfg in scheduler_configs}
    scheduler_instances = {}
    currently_building = set()

    # DFS to resolve all schedulers and their dependencies.
    def build_scheduler(label: str):
        """Recursive helper to build a scheduler and its dependencies."""
        if label in scheduler_instances:
            return scheduler_instances[label]
        
        if label not in schedulers_map:
            raise ValueError(f"❌ Scheduler label '{label}' not found in configs.")

        # Detect circular dependencies in YAML
        if label in currently_building:
            raise RecursionError(f"❌ Circular dependency detected while building scheduler '{label}'")
        
        currently_building.add(label)
        
        cfg = schedulers_map[label]
        kwargs = cfg.params.copy()
        
        # Resolve Optimizer (if specified)
        opt_label = kwargs.pop("optimizer_label", None)
        target_opt = None
        if opt_label:
            if opt_label not in optimizers_dict:
                raise KeyError(
                    f"❌ Optimizer '{opt_label}' required by scheduler '{label}' "
                    f"was not found. ⚠️ Available optimizers: {list(optimizers_dict.keys())}"
                )
            target_opt = optimizers_dict[opt_label]

        # Resolve Nested Schedulers (if specified)
        if "schedulers" in kwargs:
            nested_configs = kwargs.pop("schedulers")
            resolved_nested = []
            
            for item in nested_configs:
                # Handle both formats: `- label: "sch1"` (dict) OR `- "sch1"` (string)
                sub_label = item["label"] if isinstance(item, dict) else item
                resolved_nested.append(build_scheduler(sub_label))
            
            # Put the instantiated list back into kwargs so it gets passed to the constructor
            kwargs["schedulers"] = resolved_nested

        # Fetch the Scheduler class dynamically
        try:
            sched_cls = getattr(torch.optim.lr_scheduler, cfg.name)
        except AttributeError:
            raise ValueError(f"❌ Scheduler '{cfg.name}' is not a valid torch.optim.lr_scheduler class.")

        # Instantiate Class
        try:
            # Schedulers like 'ChainedScheduler' only take 'schedulers' as args (no optimizer).
            # Schedulers like 'SequentialLR' take BOTH 'optimizer' and 'schedulers'.
            # Standard schedulers take 'optimizer'.
            if target_opt is not None:
                instance = sched_cls(target_opt, **kwargs)
            else:
                instance = sched_cls(**kwargs)
                
            scheduler_instances[label] = instance
            print(f"⏰ Resolved Scheduler: '{label}' ({cfg.name})")
            
        except TypeError as e:
            raise TypeError(f"❌ Error initializing scheduler '{cfg.name}' (label '{label}'): {e}. Check YAML params!")
        finally:
            currently_building.remove(label)

        return instance

    # Kick off the recursive build process for all defined schedulers
    if scheduler_configs:
        for cfg in scheduler_configs:
            build_scheduler(cfg.label)

    return scheduler_instances

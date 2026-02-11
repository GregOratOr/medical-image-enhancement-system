import torch
import torch.optim as optim
import torchvision.transforms as T
from configs.config import Config
from src.transforms import noise as N


def get_noise_transform(cfg: Config) -> T.Compose:
    """
    Factory: dynamically builds the noise pipeline from config.
    """
    # 1. The Registry: Map config strings to Class Objects
    NOISE_REGISTRY = {
        "gaussian": N.AddGaussianNoise,
        "poisson": N.AddPoissonNoise,
        "spec_poisson": N.SpectralPoissonNoise,
        "spec_gaussian_blur": N.SpectralGaussianBlur,
        "spec_bernoulli": N.SpectralBernoulliNoise,
        "spec_drop": N.RandomSpectralDrop,
        "spec_gaussian_noise": N.AddSpectralGaussianNoise
    }
    
    transforms_list = []
    
    for op in cfg.data.noise_ops:
        noise_type = op['type'].lower()
        
        if noise_type not in NOISE_REGISTRY:
            raise ValueError(f"Unknown noise type: '{noise_type}'. Available: {list(NOISE_REGISTRY.keys())}")
            
        # 2. Get the class
        NoiseClass = NOISE_REGISTRY[noise_type]
        
        # 3. Instantiate with dictionary unpacking
        # Python automatically maps yaml keys (std_range) to __init__ args
        try:
            transforms_list.append(NoiseClass(**op['params']))
        except TypeError as e:
            raise TypeError(f"Error initializing {noise_type}: {e}. Check your YAML params!")
            
    return T.Compose(transforms_list)


def get_optimizer(model: torch.nn.Module, cfg: Config) -> dict[str, torch.optim.Optimizer]:
    """
    Factory: Returns a Dictionary of Optimizers.
    """
    opts_cfg = cfg.train.optimizers
    optimizers = {}
    
    # We define available classes
    OPTIMIZERS = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop
    }

    # Strategy: 
    # If model is a single nn.Module, all optimizers apply to it (unless specific logic added).
    # If model is a Dict[nn.Module], we could match keys. 
    # For now, we assume simple case: 1 Model -> N Optimizers (rare) or 1 Model -> 1 Optimizer.
    
    for name, conf in opts_cfg.items():
        opt_type = conf.get("type", "adam").lower()
        opt_params = conf.get("params", {})
        
        # Global Fallbacks
        lr = opt_params.pop("lr", cfg.train.lr)
        wd = opt_params.pop("weight_decay", cfg.train.weight_decay)
        
        if opt_type not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {opt_type}")
            
        # Initialize
        # Note: In complex multi-model setups (GANs), you'd filter model.parameters() here.
        # For N2N, we pass all parameters.
        opt_instance = OPTIMIZERS[opt_type](
            model.parameters(), 
            lr=lr, 
            weight_decay=wd, 
            **opt_params
        )
        optimizers[name] = opt_instance
        
    return optimizers


def get_schedulers(optimizers: dict[str, torch.optim.Optimizer], cfg: Config) -> dict[str, torch.optim.lr_scheduler.LRScheduler]:
    """
    Factory: Returns a Dictionary of Schedulers, matching optimizers by key.
    """
    scheds_cfg = cfg.train.schedulers
    schedulers = {}
    
    for name, conf in scheds_cfg.items():
        # Safety: Ensure we have an optimizer with this name
        if name not in optimizers:
            # If explicit mapping missing, maybe apply to 'default'? 
            # For strictness, we skip or raise error.
            continue
            
        sched_type = conf.get("type", "plateau").lower()
        sched_params = conf.get("params", {})
        optimizer = optimizers[name]
        
        if sched_type == "plateau":
            s = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_params)
        elif sched_type == "step":
            s = optim.lr_scheduler.StepLR(optimizer, **sched_params)
        elif sched_type == "cosine":
            if "T_max" not in sched_params: sched_params["T_max"] = cfg.train.epochs
            s = optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_params)
        else:
            continue
            
        schedulers[name] = s
        
    return schedulers
import inspect
import torch
from typing import Any, Callable
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

class Metrics:
    """
    A central registry for evaluation metrics with device-aware dispatch.
    
    Features:
    - distinct implementations for 'cpu' and 'gpu'
    - automatic resolution of device strings (e.g., 'cuda:0' -> 'gpu')
    - fallback mechanisms if a specific device implementation is missing
    """
    # Registry Structure: { 'metric_name': { 'cpu': cpu_fn, 'gpu': gpu_fn } }
    _registry: dict[str, dict[str, Callable]] = {}

    @staticmethod
    def _resolve_device_type(device: str | torch.device) -> str:
        """Resolves various device representations to either 'cpu' or 'gpu'.
        
        Examples:
            'cuda' -> 'gpu'
            'cuda:0' -> 'gpu'
            torch.device('cuda') -> 'gpu'
            'cpu' -> 'cpu'
        """
        
        if isinstance(device, torch.device):
            device = device.type
        
        device = str(device).lower()
        
        if 'cuda' in device or 'gpu' in device:
            return 'gpu'
        return 'cpu'

    @classmethod
    def register(cls, name: str, device: str = 'cpu'):
        """Decorator to register a metric function for a specific device.
        
        Args:
            name (str): The name of the metric (e.g., 'psnr').
            device (str): 'cpu', 'gpu', or 'cuda' (will be resolved to 'gpu').
        """
        
        norm_device = cls._resolve_device_type(device)
        
        def decorator(func: Callable):
            if name not in cls._registry:
                cls._registry[name] = {}
            
            cls._registry[name][norm_device] = func
            return func
        return decorator

    @classmethod
    def compute(
        cls, 
        prediction: Any, 
        target: Any, 
        metrics: list[str], 
        device: str | torch.device = 'cpu'
    ) -> dict[str, float]:
        """Computes requested metrics, dispatching to the appropriate implementation based on device.
        
        Args:
            pred: Predictions
            target: Ground Truth
            metrics: List of metric names to compute.
            device: The target device (e.g., 'cuda', 'cpu', torch.device('cuda:0')).
            
        Returns:
            dict[str, float]: Dictionary of results.
        """
        
        results = {}
        target_device = cls._resolve_device_type(device)
        
        for name in metrics:
            if name not in cls._registry:
                # print(f"Warning: Metric '{name}' not found in registry. Skipping.")
                continue
            
            implementations = cls._registry[name]
            
            # 1. Try to get the requested device implementation
            func = implementations.get(target_device)
            
            # 2. Fallback Mechanism
            if func is None:
                # If we wanted GPU but only have CPU (or vice versa), try the other
                fallback_device = 'cpu' if target_device == 'gpu' else 'gpu'
                if fallback_device in implementations:
                    func = implementations[fallback_device]
                else:
                    print(f"Error: No implementation found for '{name}'.")
                    continue

            # 3. Execute
            try:
                val = func(prediction, target)
                
                # Standardize output to python float for clean logging
                if hasattr(val, 'item'):
                    val = val.item()
                results[name] = val
                
            except Exception as e:
                print(f"Error computing '{name}' on {target_device}: {e}")
                results[name] = 0.0
            
        return results
    
    @classmethod
    def help(cls, metric_name: str, device: str = 'cpu'):
        """Retrieves and prints the signature and docstring for a specific metric implementation.

        This is useful for introspecting registered metrics to understand their 
        expected arguments, return types, and behavior, especially since the 
        registry pattern hides the actual function definitions from the user.

        Args:
            name (str): The name of the registered metric (e.g., 'psnr', 'ssim').
            device (Union[str, torch.device], optional): The device implementation to inspect 
                (e.g., 'cpu', 'cuda', 'gpu'). The method will resolve this to the 
                correct backend. Defaults to 'cpu'.

        Returns:
            None: This method prints directly to standard output.
        """
        
        device_type = cls._resolve_device_type(device)
        
        # 1. Get the function
        try:
            func = cls._registry[metric_name][device_type]
        except KeyError:
            print(f"Metric '{metric_name}' not found for {device_type}.")
            return

        # 2. Introspect
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or "No documentation."
        
        print(f"--- Metric: {metric_name} ({device_type}) ---")
        print(f"Signature: {func.__name__}{sig}")
        print(f"Docs: {doc}")


# --- Registered functions ---

@Metrics.register("psnr", device="cuda") # 'cuda' automatically becomes 'gpu'
def _psnr_gpu(pred: torch.Tensor, target: torch.Tensor):
    # Ensure inputs are tensors
    if not isinstance(pred, torch.Tensor): pred = torch.tensor(pred)
    if not isinstance(target, torch.Tensor): target = torch.tensor(target)
    
    # torchmetrics expects (B, C, H, W)
    if pred.ndim == 3: pred = pred.unsqueeze(0)
    if target.ndim == 3: target = target.unsqueeze(0)

    # Use functional API for stateless execution
    # data_range=1.0 assumes images are normalized [0, 1]
    return peak_signal_noise_ratio(pred, target, data_range=1.0)

@Metrics.register("ssim", device="cuda")
def _ssim_gpu(pred: torch.Tensor, target: torch.Tensor):
    # Ensure inputs are tensors
    if not isinstance(pred, torch.Tensor): pred = torch.tensor(pred)
    if not isinstance(target, torch.Tensor): target = torch.tensor(target)

    # torchmetrics expects (B, C, H, W)
    if pred.ndim == 3: pred = pred.unsqueeze(0)
    if target.ndim == 3: target = target.unsqueeze(0)
    
    return structural_similarity_index_measure(pred, target, data_range=1.0)
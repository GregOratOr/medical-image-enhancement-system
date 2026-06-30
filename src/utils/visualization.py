import torch
import torchvision.utils as vutils
import torch.fft
from typing import Callable

class Visualizer:
    """
    Generates rich visualization grids for medical image denoising.
    
    Layout:
        Row 1: [ Original | Input (Noisy) | Output (Denoised) | Target ] (Spatial)
        Row 2: [ Original | Input (Noisy) | Output (Denoised) | Target ] (Frequency/FFT)
        Row 3: [  Blank   | Noise Map     | Error Map         |  Blank ] (Analysis)
    """

    def __init__(self, denormalize_fn: Callable | None = None):
        # Optional: function to map [-1, 1] back to [0, 1]
        self.denorm = denormalize_fn

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Min-Max normalizes a tensor to [0, 1] for display."""
        
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 1e-6:
            return (tensor - min_val) / (max_val - min_val)
        return torch.zeros_like(tensor)

    def _compute_spectral(self, img: torch.Tensor) -> torch.Tensor:
        """Computes the log-magnitude spectrum (Decibel scale) of an image batch."""
        
        # 1. FFT2 (Real -> Complex)
        fft_img = torch.fft.fft2(img)
        
        # 2. Shift zero frequency to center (Explicit dims are safer)
        fft_shift = torch.fft.fftshift(fft_img, dim=(-2, -1))
        
        # 3. Compute Magnitude
        mag = torch.abs(fft_shift)
        
        # 4. Decibel Scale (20 * log10)
        # We use natural log here for speed, the visual effect is similar to log10
        # The '20' factor helps expand faint high-frequency details (noise/edges)
        mag_db = 20 * torch.log(mag + 1e-8)
        
        # 5. Normalize per image (Robust Visualization)
        # We iterate to avoid one bright artifact squashing the contrast of the whole batch
        normalized_ffts = []
        for i in range(mag_db.shape[0]):
            normalized_ffts.append(self._normalize(mag_db[i]))
            
        return torch.stack(normalized_ffts)

    def create_grid(
        self, 
        clean_gt: torch.Tensor, 
        noisy_input: torch.Tensor, 
        denoised_output: torch.Tensor, 
        target: torch.Tensor,
        max_images: int = 4
    ) -> torch.Tensor:
        """Constructs the 3-row diagnostic grid.

        Args:
            clean_gt: The ground truth clean image (Original).
            noisy_input: The input fed to the model (Input).
            denoised_output: The model's prediction (Output).
            target: The regression target (Clean for N2C, Noisy2 for N2N).
            max_images: Max number of samples to visualize from the batch.
        """

        # 1. Slice Batch
        N = min(clean_gt.shape[0], max_images)
        orig = clean_gt[:N].detach().cpu()
        inp = noisy_input[:N].detach().cpu()
        out = denoised_output[:N].detach().cpu()
        tgt = target[:N].detach().cpu()

        # 2. Row 1: Spatial Domain
        # [ Original | Input | Output | Target ]
        row_spatial = torch.cat([orig, inp, out, tgt], dim=3)

        # 3. Row 2: Frequency Domain (FFT)
        # [ Orig_FFT | Inp_FFT | Out_FFT | Tgt_FFT ]
        fft_orig = self._compute_spectral(orig)
        fft_inp = self._compute_spectral(inp)
        fft_out = self._compute_spectral(out)
        fft_tgt = self._compute_spectral(tgt)
        
        row_spectral = torch.cat([fft_orig, fft_inp, fft_out, fft_tgt], dim=3)

        # 4. Row 3: Difference Maps
        # [ Blank | Noise Map (I-O) | Error Map (O-T) | Blank ]
        
        # Blank placeholder
        blank = torch.zeros_like(orig)
        
        # Noise Map: |Input - Output|
        # Shows what the model *removed*. If you see bones here, the model is over-smoothing.
        noise_map = torch.abs(inp - out)
        noise_map = self._normalize(noise_map) # Normalize to utilize full dynamic range
        
        # Error Map: |Output - Target|
        # Shows where the model failed to match the target.
        error_map = torch.abs(out - tgt)
        error_map = self._normalize(error_map)

        row_diff = torch.cat([blank, noise_map, error_map, blank], dim=3)

        # 5. Stack Rows Vertically
        # (B, C, H, W_total) -> (B, C, H_total, W_total)
        final_grid_batch = torch.cat([row_spatial, row_spectral, row_diff], dim=2)

        # 6. Make Final Grid Image
        # Padding=2 puts a small border between samples
        return vutils.make_grid(final_grid_batch, nrow=1, padding=5, normalize=False)
import torch
import torch.nn as nn
import torch.fft as fft

# =========================================================
# 1. REAL DOMAIN NOISE (Spatial)
# =========================================================

class AddGaussianNoise(nn.Module):
    """Adds Additive White Gaussian Noise (AWGN) in the Real Domain.

    Simulates thermal or electronic sensor noise by adding random values sampled 
    from a Normal distribution.
    """

    def __init__(self, mean: float = 0.0, std_range: tuple = (0.01, 0.05), name: str = "Gaussian"):
        """Initializes the Gaussian noise module.

        Args:
            mean (float, optional): Mean of the Gaussian distribution. Defaults to 0.0.
            std_range (tuple, optional): Range (min, max) to sample the standard deviation from. 
                Defaults to (0.01, 0.05).
            name (str, optional): Label for logging purposes. Defaults to "Gaussian".
        """
        super().__init__()
        self.name = name
        self.mean = mean
        self.std_range = std_range

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Applies Gaussian noise to the input image.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Noisy image tensor, clamped to [0, 1].
        """
        # Sample random standard deviation for this image
        sigma = torch.empty(1, device=img.device).uniform_(*self.std_range).item()
        noise = torch.randn_like(img) * sigma + self.mean
        return torch.clamp(img + noise, 0.0, 1.0)
    
    def __repr__(self):
        return f"{self.name}(mean={self.mean}, std={self.std_range})"


class AddPoissonNoise(nn.Module):
    """Adds Poisson (Shot) Noise in the Real Domain.

    Simulates photon starvation (e.g., Low-dose CT).
    Approximates Poisson(lam) using Normal(lam, sqrt(lam)).
    Lower lambda values correspond to higher noise (fewer photons).
    """

    def __init__(self, lam_range: tuple = (50.0, 100.0), name: str = "Poisson"):
        """Initializes the Poisson noise module.

        Args:
            lam_range (tuple, optional): Range (min, max) for the lambda parameter (simulated photon count).
                Defaults to (50.0, 100.0).
            name (str, optional): Label for logging purposes. Defaults to "Poisson".
        """
        super().__init__()
        self.name = name
        self.lam_range = lam_range

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Applies Poisson noise to the input image.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Noisy image tensor, clamped to [0, 1].
        """
        # Random photon count scaling factor
        lam = torch.empty(1, device=img.device).uniform_(*self.lam_range).item()
        
        # Scale to simulated photon counts
        img_counts = img * lam
        
        # Signal-Dependent Noise: Variance = Mean
        noise = torch.randn_like(img) * torch.sqrt(img_counts + 1e-6)
        
        # Rescale back to [0, 1]
        noisy_img = (img_counts + noise) / lam
        return torch.clamp(noisy_img, 0.0, 1.0)

    def __repr__(self):
        return f"{self.name}(lam={self.lam_range})"
    
# =========================================================
# 2. SPECTRAL DOMAIN NOISE (Frequency/K-Space)
# =========================================================

class SpectralPoissonNoise(nn.Module):
    """Applies Poisson statistics to the MAGNITUDE of the k-space data.
    
    This preserves the phase information (structure) while corrupting the intensity,
    simulating acquisition noise that depends on signal strength.
    """

    def __init__(self, strength_range: tuple = (0.1, 0.2), mask_ratio: float = 0.05, name: str = "Spectral Poisson"):
        """Initializes the Spectral Poisson noise module.

        Args:
            strength_range (tuple, optional): Range (min, max) for the noise strength multiplier. Defaults to (0.1, 0.2).
            mask_ratio (float, optional): Percentage of frequencies to corrupt (0.0 to 1.0). Defaults to 0.05.
            name (str, optional): Label for logging purposes. Defaults to "Spectral Poisson".
        """
        super().__init__()
        self.strength_range = strength_range
        self.mask_ratio = mask_ratio
        self.name = name

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Applies Poisson noise to frequency magnitudes.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Reconstructed image with spectral Poisson noise.
        """
        device = img.device
        strength = torch.empty(1, device=device).uniform_(*self.strength_range).item()
        
        # 1. FFT
        fft_img = fft.fft2(img)
        fft_shift = fft.fftshift(fft_img)
        
        magnitude = torch.abs(fft_shift)
        phase = torch.angle(fft_shift)
        
        # 2. Select Frequencies to Corrupt
        mask = torch.rand_like(magnitude) < self.mask_ratio
        
        # 3. Apply Poisson to Magnitude
        noisy_mag_val = torch.poisson((magnitude + 1e-5) * strength) / strength
        
        # Only apply to masked regions
        new_magnitude = torch.where(mask, noisy_mag_val, magnitude)
        
        # 4. Reconstruct with Original Phase
        new_complex = new_magnitude * torch.exp(1j * phase)
        recon = fft.ifft2(fft.ifftshift(new_complex))
        
        return torch.clamp(recon.real, 0.0, 1.0)

    def __repr__(self):
        return f"{self.name}(str={self.strength_range})"


class SpectralGaussianBlur(nn.Module):
    """Applies Gaussian Blur via Frequency Domain multiplication.

    Simulates acquisition smoothing or partial volume effects by attenuating high frequencies.
    The mask is a Gaussian centered at DC (low frequencies).
    """

    def __init__(self, sigma_range: tuple = (0.1, 0.5), name: str = "Gaussian Blur"):
        """Initializes the Spectral Gaussian Blur module.

        Args:
            sigma_range (tuple, optional): Range (min, max) for the blur sigma. 
                Higher sigma in frequency domain = Less blur (wider passband).
                Lower sigma in frequency domain = More blur (narrower passband).
                Defaults to (0.1, 0.5).
            name (str, optional): Label for logging. Defaults to "Gaussian Blur".
        """
        super().__init__()
        self.name = name
        self.sigma_range = sigma_range

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Applies a Gaussian Low-Pass filter in the frequency domain.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Blurred image.
        """
        _, h, w = img.shape
        device = img.device
        
        # 1. FFT
        fft_img = fft.fft2(img)
        fft_shift = fft.fftshift(fft_img)
        
        # 2. Generate Gaussian Mask
        sigma = torch.empty(1, device=device).uniform_(*self.sigma_range).item()
        
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        radius_sq = X**2 + Y**2
        
        # Gaussian Low-Pass Filter
        mask = torch.exp(-radius_sq / (2 * sigma**2))
        
        # 3. Apply Mask
        fft_masked = fft_shift * mask
        
        # 4. IFFT
        ifft_shift = fft.ifftshift(fft_masked)
        recon = fft.ifft2(ifft_shift)
        
        return torch.clamp(recon.real, 0.0, 1.0)
    
    def __repr__(self):
        return f"{self.name}(sigma={self.sigma_range})"


class SpectralBernoulliNoise(nn.Module):
    """Randomly drops frequencies (Bernoulli Process) while preserving DC.

    Simulates missing k-space data, sparse sampling, or packet loss.
    """

    def __init__(self, prob_drop_range: tuple = (0.1, 0.3), name: str = "Bernoulli"):
        """Initializes the Spectral Bernoulli noise module.

        Args:
            prob_drop_range (tuple, optional): Range (min, max) for the drop probability. Defaults to (0.1, 0.3).
            name (str, optional): Label for logging. Defaults to "Bernoulli".
        """
        super().__init__()
        self.name = name
        self.prob_drop_range = prob_drop_range

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Applies a random dropout mask to the frequency domain.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Image with missing frequencies.
        """
        _, h, w = img.shape
        device = img.device
        
        # 1. FFT
        fft_img = fft.fft2(img)
        fft_shift = fft.fftshift(fft_img)
        
        # 2. Generate Bernoulli Mask (1 = Keep, 0 = Drop)
        p = torch.empty(1, device=device).uniform_(*self.prob_drop_range).item()
        
        # Create mask: prob(1) = 1 - p
        mask = torch.bernoulli(torch.full((h, w), 1 - p, device=device))
        
        # Always keep the DC component (center) to preserve overall brightness
        cy, cx = h // 2, w // 2
        mask[cy-2:cy+3, cx-2:cx+3] = 1.0
        
        # 3. Apply Mask
        fft_masked = fft_shift * mask
        
        # 4. IFFT
        ifft_shift = fft.ifftshift(fft_masked)
        recon = fft.ifft2(ifft_shift)
        
        return torch.clamp(recon.real, 0.0, 1.0)

    def __repr__(self):
        return f"{self.name}(p={self.prob_drop_range})"


class RandomSpectralDrop(nn.Module):
    """Randomly drops frequencies in the spectral domain.
    
    Effect: Simulates MRI undersampling or blurring.
    """

    def __init__(self, drop_ratio: float = 0.1, name: str = "Spectral Drop"):
        """Initializes the random spectral drop module.

        Args:
            drop_ratio (float, optional): Probability of dropping a frequency. Defaults to 0.1.
            name (str, optional): Label for logging. Defaults to "Spectral Drop".
        """
        super().__init__()
        self.name = name
        self.drop_ratio = drop_ratio

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Applies random spectral dropout.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Image with dropped frequencies.
        """
        # 1. FFT (Real -> Complex Frequency)
        fft_img = fft.fft2(img)
        fft_shift = fft.fftshift(fft_img)

        # 2. Create Mask
        _, h, w = img.shape
        mask = torch.empty((h, w), device=img.device).uniform_(0, 1) > self.drop_ratio
        
        # 3. Apply Mask
        fft_masked = fft_shift * mask

        # 4. IFFT (Complex Frequency -> Real)
        ifft_shift = fft.ifftshift(fft_masked)
        recon = fft.ifft2(ifft_shift)
        
        # Return Magnitude (remove tiny imaginary parts from float errors)
        return torch.clamp(recon.real, 0.0, 1.0)
    
    def __repr__(self) -> str:
        return f"{self.name}(drop_ratio={self.drop_ratio})"


class AddSpectralGaussianNoise(nn.Module):
    """Adds Gaussian noise to the MAGNITUDE of the k-space data.

    Effect: Creates global, correlated artifacts (streaks/ripples) in the spatial domain 
    while preserving structural edges (phase).
    """

    def __init__(self, std_range: tuple = (0.1, 0.2), name: str = "Spectral Gaussian"):
        """Initializes the Spectral Gaussian noise module.

        Args:
            std_range (tuple, optional): Range (min, max) for the noise standard deviation. Defaults to (0.1, 0.2).
            name (str, optional): Label for logging. Defaults to "Spectral Gaussian".
        """
        super().__init__()
        self.std_range = std_range
        self.name = name

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Applies Gaussian noise to the frequency magnitude.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Noisy image.
        """
        device = img.device
        std = torch.empty(1, device=device).uniform_(*self.std_range).item()
        
        # 1. FFT
        fft_img = fft.fft2(img)
        fft_shift = fft.fftshift(fft_img)
        
        magnitude = torch.abs(fft_shift)
        phase = torch.angle(fft_shift)
        
        # 2. Add Noise to Magnitude
        noise = torch.randn_like(magnitude) * std
        new_magnitude = magnitude + noise
        
        # 3. Reconstruct
        new_complex = new_magnitude * torch.exp(1j * phase)
        recon = fft.ifft2(fft.ifftshift(new_complex))
        
        return torch.clamp(recon.real, 0.0, 1.0)

    def __repr__(self):
        return f"{self.name}(std={self.std_range})"
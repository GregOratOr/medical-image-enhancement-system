from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """Standard Convolutional Layer with Kaiming Initialization and Optional Activation.

    Attributes:
        conv (nn.Conv2d): The underlying convolution operation.
        act (nn.Module): The activation function (or Identity if None).
    """

    def __init__(self,
                in_channels: int, out_channels: int, kernel_size: int = 3, 
                stride: int = 1, padding: int | None = None, bias: bool = True,
                activation: str | None = None):
        
        """Initializes the convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output filters.
            kernel_size (int, optional): Size of the convolving kernel. Defaults to 3.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Zero-padding. If None, calculated for 'same' padding.
            bias (bool, optional): Whether to add a learnable bias. Defaults to True.
            activation (str, optional): Activation mode ('relu', 'leaky_relu', etc.). 
                                        If None, no activation is applied.
        """
        super().__init__()
        
        # 1. Padding Logic (Preserve Spatial Size if not specified)
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias
        )
        
        # 2. Activation Logic (Flexible)
        if activation is not None:
            self.act = Activation(mode=activation)
        else:
            self.act = nn.Identity()

        # 3. Initialization (Explicit is better than Implicit)
        # Note: Use 'leaky_relu' as the nonlinearity for Kaiming even if actual act is None. A safe default for "general purpose" weights.
        nn.init.kaiming_normal_(self.conv.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        return x

class UpsampleLayer(nn.Module):
    """Upsamples the input using interpolation to avoid checkerboard artifacts."""

    def __init__(self, scale_factor=2, mode='nearest'):
        """Initializes the upsampling layer.

        Args:
            scale_factor (int, optional): Multiplier for spatial dimensions. Defaults to 2.
            mode (str, optional): Interpolation mode ('nearest', 'bilinear'). Defaults to 'nearest'.
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """Upsamples the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Upsampled tensor.
        """
        # Strict alignment for 'bilinear' mode to avoid coordinate shifts. Ignored for 'nearest' mode.
        align_corners = False if self.mode == 'bilinear' else None
        
        return F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode=self.mode, 
            align_corners=align_corners
        )

class ConcatLayer(nn.Module):
    """Concatenates a list of tensors along a specific dimension.
    
    Robustly handles spatial mismatches by center-cropping all tensors 
    to the size of the smallest tensor in the list.
    """

    def __init__(self, dim: int = 1):
        """Initializes the concatenation layer.

        Args:
            dim (int, optional): The dimension along which to concatenate. Defaults to 1 (Channels).
        """
        super().__init__()
        self.dim = dim

    def forward(self, layers: list[torch.Tensor]) -> torch.Tensor:
        """Concatenates input tensors, cropping them to the smallest common shape.

        Args:
            layers (List[torch.Tensor]): A list of tensors to concatenate [x1, x2, ...].

        Returns:
            torch.Tensor: A single concatenated tensor.
        """
        # 1. Find the target spatial size (Smallest H, W in the batch)
        # Assumption: [Batch, Channel, H, W]
        target_h = min(t.shape[2] for t in layers)
        target_w = min(t.shape[3] for t in layers)
        
        cropped_inputs = []
        for t in layers:
            h, w = t.shape[2], t.shape[3]
            
            # Optimization: If size matches, just append
            if h == target_h and w == target_w:
                cropped_inputs.append(t)
                continue
                
            # Otherwise, Center Crop
            diffY = (h - target_h) // 2
            diffX = (w - target_w) // 2
            
            # Python slicing is robust; it handles off-by-one errors in odd sizes well
            t_cropped = t[:, :, diffY:diffY+target_h, diffX:diffX+target_w]
            cropped_inputs.append(t_cropped)
            
        return torch.cat(cropped_inputs, dim=self.dim)

class DownsampleLayer(nn.Module):
    """Downsamples the input tensor using Max Pooling.

    Reduces spatial dimensions (H, W) by the given scale factor to increase 
    the receptive field and reduce computational cost.
    """

    def __init__(self, scale_factor: int = 2):
        """Initializes the downsampling layer.

        Args:
            scale_factor (int, optional): The factor by which to downsample. 
                Kernel size and stride will be set to this value. Defaults to 2.
        """
        super(DownsampleLayer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for downsampling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Downsampled tensor.
        """
        return self.pool(x)

class Activation(nn.Module):
    """A generic activation layer wrapper.
    
    Allows for easy switching between different activation functions (ReLU, LeakyReLU, PReLU, ELU)
    without changing the network architecture code.
    """

    def __init__(self, mode='leaky_relu', negative_slope=0.1, num_parameters=1):
        """Initializes the activation layer.

        Args:
            mode (str, optional): The type of activation to use. 
                Options: ['relu', 'leaky_relu', 'prelu', 'elu', 'sigmoid', 'tanh', 'gelu]. 
                Defaults to 'leaky_relu'.
            negative_slope (float, optional): Initial slope for LeakyReLU/PReLU or alpha for ELU. 
                Defaults to 0.1.
            num_parameters (int, optional): Number of learnable parameters for PReLU (1 or n_channels).
                Defaults to 1 (shared slope).
        """
        super().__init__()
        self.mode = mode
        
        match(mode):
            case 'leaky_relu':
                self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
            case 'prelu':
                # Note: num_parameters=1 shares the same slope for all channels
                self.act = nn.PReLU(num_parameters=num_parameters, init=negative_slope)
            case 'relu':
                self.act = nn.ReLU(inplace=True)
            case 'elu':
                self.act = nn.ELU(alpha=negative_slope, inplace=True)
            case 'sigmoid':
                self.act = nn.Sigmoid()
            case 'tanh':
                self.act = nn.Tanh()
            case 'gelu':
                self.act = nn.GELU()
            case _:
                raise ValueError(f"Activation mode '{mode}' not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the selected activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated tensor.
        """
        return self.act(x)

class ConvBlock(nn.Module):
    """A double convolution block used in U-Net (Conv -> Act -> Conv -> Act).

    According to Noise2Noise, we use LeakyReLU and typically skip Batch Normalization
    to avoid shifting the intensity distribution of the noise.
    """

    def __init__(self, in_channels: int, out_channels: int, activation: nn.Module, batch_norm: bool = False):
        super().__init__()
        
        layers = []
        
        # 1st Convolution
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation)
        
        # 2nd Convolution
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation)
        
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
class Noise2NoiseOriginal(nn.Module):
    """The original Noise2Noise U-Net architecture (Lehtinen et al., 2018).

    A 5-level U-Net optimized for image restoration.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 48):
        """Initializes the Noise2Noise model.

        Args:
            in_channels (int, optional): Number of input channels (e.g., 1 for grayscale). Defaults to 1.
            out_channels (int, optional): Number of output channels. Defaults to 1.
            base_channels (int, optional): Number of filters in the first layer. 
                The paper often uses 48 or 64. Defaults to 48.
        """
        super().__init__()
        
        # Config
        # Paper uses LeakyReLU with slope 0.1
        self.act = Activation(mode='leaky_relu', negative_slope=0.1)
        
        # --- ENCODER ---
        # Level 1
        self.enc1 = ConvBlock(in_channels, base_channels, self.act)
        self.pool1 = DownsampleLayer()
        
        # Level 2
        self.enc2 = ConvBlock(base_channels, base_channels * 2, self.act)
        self.pool2 = DownsampleLayer()
        
        # Level 3
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, self.act)
        self.pool3 = DownsampleLayer()
        
        # Level 4
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, self.act)
        self.pool4 = DownsampleLayer()
        
        # Level 5 (Bottleneck)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16, self.act)
        
        # --- DECODER ---
        # Level 4
        self.up4 = UpsampleLayer(scale_factor=2, mode='nearest')
        self.concat4 = ConcatLayer()
        self.dec4 = ConvBlock(base_channels * 16 + base_channels * 8, base_channels * 8, self.act)
        
        # Level 3
        self.up3 = UpsampleLayer(scale_factor=2, mode='nearest')
        self.concat3 = ConcatLayer()
        self.dec3 = ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4, self.act)
        
        # Level 2
        self.up2 = UpsampleLayer(scale_factor=2, mode='nearest')
        self.concat2 = ConcatLayer()
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2, self.act)
        
        # Level 1
        self.up1 = UpsampleLayer(scale_factor=2, mode='nearest')
        self.concat1 = ConcatLayer()
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels, self.act)
        
        # --- OUTPUT ---
        # 1x1 Conv to project back to output space (Linear activation)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the U-Net.

        Args:
            x (torch.Tensor): Input image tensor (Batch, In_Channels, H, W).

        Returns:
            torch.Tensor: Denoised output tensor (Batch, Out_Channels, H, W).
        """
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        d4 = self.up4(b)
        d4 = self.concat4([d4, e4]) # Skip connection
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = self.concat3([d3, e3]) # Skip connection
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = self.concat2([d2, e2]) # Skip connection
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = self.concat1([d1, e1]) # Skip connection
        d1 = self.dec1(d1)
        
        # Output
        return self.final_conv(d1)

    def _initialize_weights(self):
        """Apply He Initialization to Conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    

class EncoderBlock(nn.Module):
    """A single encoder stage: Conv -> Act -> Conv -> Act.
    
    Returns the feature map to be stored for the skip connection.
    """
    def __init__(self, in_channels: int, out_channels: int, activation_mode: str = 'leaky_relu'):
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=activation_mode),
            ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=activation_mode)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """A single decoder stage: Upsample -> Concat -> Conv -> Conv.
    
    Handles the fusion of the upsampled features with the skip connection.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, activation_mode: str = 'leaky_relu'):
        super().__init__()
        # 1. Upsample
        self.upsample = UpsampleLayer(scale_factor=2, mode='nearest')
        
        # 2. Concat
        self.concat = ConcatLayer()
        
        # 3. Convolutions
        # Input to conv is (upsampled_channels + skip_channels)
        # Note: Assumption that upsampling doesn't change channel count, but Concat does.
        conv_in_channels = in_channels + skip_channels
        
        self.block = nn.Sequential(
            ConvLayer(conv_in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=activation_mode),
            ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=activation_mode)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.concat([x, skip])
        return self.block(x)
    

class Noise2Noise(nn.Module):
    """A modern, generalized U-Net architecture for Image Denoising.
    
    Features:
    - Dynamic depth (configurable).
    - Automatic channel scaling.
    - Generalized for arbitrary inputs/outputs.
    """
    def __init__(
        self, 
        in_channels: int = 1, 
        out_channels: int = 1, 
        base_channels: int = 48, 
        depth: int = 4, 
        activation_mode: str = 'leaky_relu'
    ):
        super().__init__()
        self.depth = depth
        
        # --- ENCODER PATH ---
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        current_channels = in_channels
        next_channels = base_channels
        
        for i in range(depth):
            self.encoders.append(EncoderBlock(current_channels, next_channels, activation_mode))
            self.downsamples.append(DownsampleLayer(scale_factor=2))
            current_channels = next_channels
            next_channels *= 2 

        # --- BOTTLENECK ---
        self.bottleneck = EncoderBlock(current_channels, next_channels, activation_mode)
        current_channels = next_channels 
        
        # --- DECODER PATH ---
        self.decoders = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            skip_channels = base_channels * (2 ** i) 
            decoder_out_channels = skip_channels 
            
            self.decoders.append(
                DecoderBlock(
                    in_channels=current_channels, 
                    skip_channels=skip_channels, 
                    out_channels=decoder_out_channels, 
                    activation_mode=activation_mode
                )
            )
            current_channels = decoder_out_channels

        # --- OUTPUT HEAD ---
        self.final_head = nn.Conv2d(current_channels, out_channels, kernel_size=1)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        
        # Encoder
        for i in range(self.depth):
            x = self.encoders[i](x)
            skips.append(x)
            x = self.downsamples[i](x)
            
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i+1)]
            x = decoder(x, skip)
            
        return self.final_head(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
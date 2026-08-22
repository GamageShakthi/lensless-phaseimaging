# =============================================================================
# deep_recon.py — Unrolled Deep Neural Network for Reconstruction
# =============================================================================
#
# This module implements the deep learning approach to lensless reconstruction.
#
# KEY IDEA: "Algorithm Unrolling"
#   Take the ADMM iterations from classical_recon.py and "unroll" them
#   into layers of a neural network. Each ADMM iteration becomes one
#   network layer, and the parameters (ρ, λ, denoiser weights) become
#   LEARNABLE through backpropagation!
#
# WHY IS THIS BETTER THAN CLASSICAL ADMM?
#   1. Learnable step sizes (ρ, λ) adapt to the specific imaging system
#   2. CNN denoiser replaces the TV proximal operator → much more expressive
#   3. Fewer iterations needed (10 learned stages ≈ 100 classical iterations)
#   4. End-to-end training optimizes the WHOLE pipeline jointly
#
# ARCHITECTURE:
#   Input: measurement y
#   For k = 1, ..., K stages:
#       x_k = FFT_solve(y, z_{k-1}, u_{k-1}; ρ_k)     [data fidelity]
#       z_k = CNN_denoiser_k(x_k + u_{k-1})             [learned prior]
#       u_k = u_{k-1} + (x_k - z_k)                     [dual update]
#   Output: z_K
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class ResBlock(nn.Module):
    """
    Residual Block — The building block of our CNN denoiser.
    
    A residual block learns the "residual" (difference) between input and
    output, which makes training much easier for deep networks.
    
        output = input + F(input)
    
    Where F is a small CNN (Conv → ReLU → Conv).
    
    WHY RESIDUAL CONNECTIONS?
    Without them, gradients vanish in deep networks, making training
    nearly impossible. With residual connections, the network can learn
    the identity function by default (F=0) and only needs to learn
    small corrections.
    """
    
    def __init__(self, num_channels: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)  # Skip connection!


class CNNDenoiser(nn.Module):
    """
    CNN Denoiser — Replaces the TV proximal operator in ADMM.
    
    In classical ADMM, the z-update uses a hand-crafted prior (TV).
    Here, we replace it with a learned CNN that acts as a more powerful
    "denoiser" / proximal operator.
    
    Architecture: Conv → ResBlocks → Conv (like a mini U-Net)
    
    Input: Noisy/artifact-y image (from x-update + dual variable)
    Output: Cleaned/denoised image
    
    The network learns what "natural images look like" from training data,
    which is a much stronger prior than TV!
    """
    
    def __init__(self, in_channels: int = 1, num_features: int = 64, 
                 num_blocks: int = 4):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_features: Number of intermediate feature channels
            num_blocks: Number of residual blocks (depth of denoiser)
        """
        super().__init__()
        
        # Encoder: lift from image space to feature space
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Body: stack of residual blocks for deep feature extraction
        self.body = nn.Sequential(
            *[ResBlock(num_features) for _ in range(num_blocks)]
        )
        
        # Decoder: project back to image space
        self.tail = nn.Conv2d(num_features, in_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denoise the input image.
        
        Args:
            x: Input image, shape (B, C, H, W)
            
        Returns:
            denoised: Cleaned image, shape (B, C, H, W)
        """
        # Learn a residual: output = input + learned_correction
        # This makes it easy for the network to learn the identity if needed
        residual = self.head(x)
        residual = self.body(residual)
        residual = self.tail(residual)
        
        return x + residual  # Global skip connection


class UnrolledADMMStage(nn.Module):
    """
    One Stage (Iteration) of the Unrolled ADMM Network
    
    This corresponds to ONE iteration of classical ADMM, but with:
    - Learnable penalty parameter ρ
    - Learnable regularization weight λ  
    - CNN denoiser instead of TV proximal operator
    
    Each stage has its OWN set of parameters, allowing the network to
    use different strategies at different stages of reconstruction.
    
    Stage k performs:
        1. x-update: Fourier-domain solve (data fidelity)
        2. z-update: CNN denoiser (learned prior)
        3. u-update: Dual variable update
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 num_features: int = 64,
                 num_blocks: int = 3,
                 init_rho: float = 1.0):
        """
        Args:
            in_channels: Number of image channels
            num_features: CNN denoiser feature channels
            num_blocks: Number of residual blocks in denoiser
            init_rho: Initial value for learnable ρ parameter
        """
        super().__init__()
        
        # Learnable ADMM penalty parameter (ρ)
        # We learn log(ρ) and exponentiate to ensure ρ > 0
        self.log_rho = nn.Parameter(torch.tensor(np.log(init_rho)))
        
        # Learnable step size for dual variable update
        self.dual_step = nn.Parameter(torch.tensor(1.0))
        
        # CNN denoiser (replaces TV proximal operator)
        self.denoiser = CNNDenoiser(
            in_channels=in_channels,
            num_features=num_features,
            num_blocks=num_blocks
        )
    
    @property
    def rho(self):
        """Get ρ (always positive via exp)."""
        return torch.exp(self.log_rho)
    
    def x_update(self, 
                 measurement_fft: torch.Tensor,
                 z: torch.Tensor, 
                 u: torch.Tensor,
                 psf_fft: torch.Tensor,
                 psf_fft_conj: torch.Tensor,
                 psf_fft_abs_sq: torch.Tensor,
                 image_size: Tuple[int, int]) -> torch.Tensor:
        """
        x-update: Solve the data fidelity sub-problem in Fourier domain.
        
        Solves: min_x ½||Ax - y||² + ρ/2·||x - (z - u)||²
        
        Closed-form in Fourier domain:
            X(f) = [H*(f)·Y(f) + ρ·FFT{z-u}] / [|H(f)|² + ρ]
        """
        B, C, H, W = z.shape
        rho = self.rho
        
        channels = []
        for c in range(C):
            # FFT of (z - u)
            v_fft = torch.fft.rfft2((z - u)[:, c, :, :])
            
            # Numerator and denominator
            numerator = psf_fft_conj.unsqueeze(0) * measurement_fft[c] + rho * v_fft
            denominator = psf_fft_abs_sq + rho
            
            X_fft = numerator / denominator.unsqueeze(0)
            x_channel = torch.fft.irfft2(X_fft, s=image_size)
            channels.append(x_channel)
        
        return torch.stack(channels, dim=1)
    
    def z_update(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        z-update: Apply CNN denoiser as learned proximal operator.
        
        Classical ADMM: z = prox_{λ·TV}(x + u)
        Learned ADMM:   z = CNN_denoiser(x + u)
        """
        v = x + u
        z = self.denoiser(v)
        # Enforce non-negativity
        z = torch.clamp(z, 0.0, 1.0)
        return z
    
    def u_update(self, u: torch.Tensor, x: torch.Tensor, 
                 z: torch.Tensor) -> torch.Tensor:
        """
        u-update: Dual variable update.
        
        u ← u + step · (x - z)
        
        The step size is learnable — the network can adjust how
        aggressively it enforces the constraint x = z.
        """
        return u + self.dual_step * (x - z)
    
    def forward(self, 
                measurement_fft: torch.Tensor,
                x: torch.Tensor, 
                z: torch.Tensor, 
                u: torch.Tensor,
                psf_fft: torch.Tensor,
                psf_fft_conj: torch.Tensor,
                psf_fft_abs_sq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run one ADMM stage.
        
        Returns:
            x, z, u: Updated variables
        """
        B, C, H, W = z.shape
        
        # x-update
        x = self.x_update(measurement_fft, z, u, 
                          psf_fft, psf_fft_conj, psf_fft_abs_sq, (H, W))
        
        # z-update
        z = self.z_update(x, u)
        
        # u-update
        u = self.u_update(u, x, z)
        
        return x, z, u


class UnrolledADMMNetwork(nn.Module):
    """
    Full Unrolled ADMM Reconstruction Network
    
    This is the main deep learning model. It stacks K UnrolledADMMStages
    to form an end-to-end differentiable reconstruction pipeline.
    
    TRAINING:
    - Input: simulated measurement y (from forward model)
    - Output: reconstructed image x̂
    - Loss: ||x̂ - x_true||² + λ_tv·TV(x̂) + λ_perceptual·Perceptual(x̂, x_true)
    - Optimizer: Adam
    
    The network learns:
    - Optimal ρ_k for each stage
    - Optimal dual step sizes
    - CNN denoiser weights (the most powerful part!)
    
    AT INFERENCE:
    Much faster than classical ADMM because:
    - Only K=10 stages (vs 100+ classical iterations)
    - Learned parameters are already optimal
    - Single forward pass, no iterative convergence needed
    """
    
    def __init__(self,
                 psf: np.ndarray,
                 num_stages: int = 10,
                 in_channels: int = 1,
                 num_features: int = 64,
                 num_blocks_per_stage: int = 3,
                 share_weights: bool = False):
        """
        Args:
            psf: Point Spread Function as numpy array
            num_stages: Number of ADMM stages (K). More = better but slower
            in_channels: Image channels (1=grayscale, 3=RGB)
            num_features: CNN feature channels per stage
            num_blocks_per_stage: Residual blocks per denoiser
            share_weights: If True, all stages share the same CNN weights
                          (fewer parameters but potentially less flexible)
        """
        super().__init__()
        
        self.num_stages = num_stages
        self.in_channels = in_channels
        
        # Store PSF FFT as non-trainable buffers
        psf_tensor = torch.from_numpy(psf).float()
        psf_fft = torch.fft.rfft2(psf_tensor)
        
        self.register_buffer('psf_fft', psf_fft)
        self.register_buffer('psf_fft_conj', torch.conj(psf_fft))
        self.register_buffer('psf_fft_abs_sq', torch.abs(psf_fft) ** 2)
        
        # Create ADMM stages
        if share_weights:
            # Weight sharing: one set of weights for all stages
            # Fewer parameters, but each stage uses same denoiser
            stage = UnrolledADMMStage(in_channels, num_features, 
                                      num_blocks_per_stage)
            self.stages = nn.ModuleList([stage] * num_stages)
        else:
            # Independent weights per stage (recommended)
            # Each stage can specialize: early stages handle large errors,
            # later stages refine fine details
            self.stages = nn.ModuleList([
                UnrolledADMMStage(
                    in_channels, num_features, num_blocks_per_stage,
                    init_rho=1.0 * (1.5 ** i)  # Increasing ρ schedule
                )
                for i in range(num_stages)
            ])
        
        # Optional: learnable initialization
        # Instead of initializing x=0, learn a better starting point
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, in_channels, 3, padding=1),
        )
    
    def forward(self, measurement: torch.Tensor, 
                return_intermediates: bool = False) -> torch.Tensor:
        """
        Reconstruct image from measurement.
        
        Args:
            measurement: Sensor measurement, shape (B, C, H, W)
            return_intermediates: If True, return list of z at each stage
            
        Returns:
            z: Reconstructed image, shape (B, C, H, W)
            intermediates: (optional) List of intermediate reconstructions
        """
        B, C, H, W = measurement.shape
        
        # Pre-compute FFT of measurement (shared across all stages)
        measurement_fft = []
        for c in range(C):
            measurement_fft.append(torch.fft.rfft2(measurement[:, c, :, :]))
        
        # Initialize ADMM variables
        # Use learned initialization for z, zeros for x and u
        z = self.init_conv(measurement)  # Learned init (better than zeros!)
        x = torch.zeros_like(measurement)
        u = torch.zeros_like(measurement)
        
        intermediates = []
        
        # Run through all stages (unrolled ADMM iterations)
        for k, stage in enumerate(self.stages):
            x, z, u = stage(measurement_fft, x, z, u,
                           self.psf_fft, self.psf_fft_conj, self.psf_fft_abs_sq)
            
            if return_intermediates:
                intermediates.append(z.detach().clone())
        
        if return_intermediates:
            return z, intermediates
        return z
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_architecture(self):
        """Print a summary of the network architecture."""
        print("=" * 60)
        print("  Unrolled ADMM Network Architecture")
        print("=" * 60)
        print(f"  Number of stages: {self.num_stages}")
        print(f"  Input channels: {self.in_channels}")
        print(f"  Total parameters: {self.count_parameters():,}")
        print()
        for i, stage in enumerate(self.stages):
            rho = stage.rho.item()
            step = stage.dual_step.item()
            denoiser_params = sum(p.numel() for p in stage.denoiser.parameters())
            print(f"  Stage {i}: ρ={rho:.4f}, dual_step={step:.4f}, "
                  f"denoiser_params={denoiser_params:,}")
        print("=" * 60)


class SimpleUNet(nn.Module):
    """
    Simple U-Net for direct (non-unrolled) learned reconstruction.
    
    This is a baseline: directly map measurement → reconstruction
    without any physics (no PSF, no forward model in the loop).
    
    Included for comparison to show that physics-informed approaches
    (unrolled ADMM) outperform pure data-driven methods!
    
    Architecture:
        Encoder: Conv blocks with downsampling
        Bottleneck: Deep feature processing
        Decoder: Conv blocks with upsampling + skip connections
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, 
                 base_features: int = 32):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 4, base_features * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_features * 8, base_features * 4)
        
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_features * 4, base_features * 2)
        
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, 2, stride=2)
        self.dec1 = self._conv_block(base_features * 2, base_features)
        
        self.final = nn.Conv2d(base_features, out_channels, 1)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        out = torch.sigmoid(self.final(d1))
        return out


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Deep Reconstruction Network Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a dummy PSF
    from forward_model import PSFGenerator
    psf_gen = PSFGenerator(size=64, seed=42)
    psf = psf_gen.gaussian_mixture(num_components=20)
    
    # Build network
    net = UnrolledADMMNetwork(
        psf=psf,
        num_stages=5,
        in_channels=1,
        num_features=32,
        num_blocks_per_stage=2
    ).to(device)
    
    net.print_architecture()
    
    # Test forward pass
    dummy_measurement = torch.randn(2, 1, 64, 64).to(device)
    output = net(dummy_measurement)
    print(f"\nInput shape:  {dummy_measurement.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test with intermediates
    output, intermediates = net(dummy_measurement, return_intermediates=True)
    print(f"\nNumber of intermediate outputs: {len(intermediates)}")
    
    print("\n✅ Deep reconstruction network test passed!")

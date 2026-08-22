# =============================================================================
# forward_model.py — DiffuserCam Forward Model & PSF Simulation
# =============================================================================
#
# This module implements the physics of a lensless phase-mask camera.
#
# KEY CONCEPTS:
#   - A DiffuserCam replaces the lens with a phase diffuser
#   - The diffuser's effect on a point source = Point Spread Function (PSF)
#   - For shift-invariant systems: measurement = PSF * scene + noise
#   - Convolution in spatial domain = multiplication in Fourier domain
#
# MATH:
#   Forward model:  y = A·x + n
#   Where A is convolution with PSF h:  y = h * x + n
#   In Fourier domain:  Y(f) = H(f)·X(f) + N(f)
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional


class PSFGenerator:
    """
    Generates Point Spread Functions (PSFs) for lensless imaging simulation.
    
    A PSF describes how a single point of light spreads across the sensor
    after passing through the phase diffuser. Think of it as the camera's
    "fingerprint" — it completely characterizes the optical system.
    
    We provide several PSF types:
    1. Gaussian Mixture — Sum of random Gaussians (simple approximation)
    2. Random Phase Mask — Physically-motivated via Fourier optics
    3. Caustic Pattern — Simulates the caustic patterns seen in real DiffuserCams
    """
    
    def __init__(self, size: int = 256, seed: int = 42):
        """
        Args:
            size: PSF image size (size x size pixels)
            seed: Random seed for reproducibility
        """
        self.size = size
        self.rng = np.random.RandomState(seed)
    
    def gaussian_mixture(self, num_components: int = 50, 
                         min_sigma: float = 1.0, 
                         max_sigma: float = 15.0) -> np.ndarray:
        """
        Generate a PSF as a mixture of random Gaussians.
        
        This is the simplest PSF model. Real diffuser PSFs look like
        random caustic patterns, but a Gaussian mixture captures the
        key property: light from one point spreads across the whole sensor.
        
        Args:
            num_components: Number of Gaussian blobs
            min_sigma: Minimum standard deviation of each Gaussian
            max_sigma: Maximum standard deviation of each Gaussian
            
        Returns:
            psf: Normalized PSF array of shape (size, size)
        """
        psf = np.zeros((self.size, self.size), dtype=np.float64)
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:self.size, 0:self.size]
        
        for _ in range(num_components):
            # Random center position
            cx = self.rng.uniform(0, self.size)
            cy = self.rng.uniform(0, self.size)
            
            # Random spread (sigma) and amplitude
            sigma = self.rng.uniform(min_sigma, max_sigma)
            amplitude = self.rng.uniform(0.5, 1.0)
            
            # 2D Gaussian: G(x,y) = A * exp(-((x-cx)² + (y-cy)²) / (2σ²))
            gaussian = amplitude * np.exp(
                -((x_coords - cx)**2 + (y_coords - cy)**2) / (2 * sigma**2)
            )
            psf += gaussian
        
        # Normalize so PSF sums to 1 (energy conservation!)
        # This is physically important: total light energy is preserved
        psf = psf / psf.sum()
        
        return psf.astype(np.float32)
    
    def random_phase_mask(self, feature_size: float = 8.0) -> np.ndarray:
        """
        Generate a PSF by simulating a random phase mask using Fourier optics.
        
        PHYSICS:
        A phase diffuser adds a random phase φ(x,y) to the incoming wavefront.
        The PSF is the squared magnitude of the Fourier transform of the
        phase-only transmission function:
        
            t(x,y) = exp(j·φ(x,y))        (phase-only mask)
            PSF = |FFT{t(x,y)}|²           (Fraunhofer diffraction)
        
        This is more physically accurate than Gaussian mixtures!
        
        Args:
            feature_size: Controls the spatial scale of phase features.
                         Larger = smoother phase → more concentrated PSF
                         Smaller = rougher phase → more spread-out PSF
                         
        Returns:
            psf: Normalized PSF array of shape (size, size)
        """
        # Step 1: Generate random phase pattern
        # Start with random noise, then smooth it to control feature size
        random_phase = self.rng.randn(self.size, self.size)
        
        # Smooth the phase to create spatially-correlated features
        # (Real diffusers have structure at specific length scales)
        kernel_size = int(feature_size) * 2 + 1
        random_phase = cv2.GaussianBlur(
            random_phase.astype(np.float32), 
            (kernel_size, kernel_size), 
            feature_size
        )
        
        # Scale phase to [0, 2π] — one full wave of optical path difference
        phase = 2 * np.pi * (random_phase - random_phase.min()) / (random_phase.max() - random_phase.min())
        
        # Step 2: Create the phase-only transmission function
        # t(x,y) = exp(j·φ(x,y))  — this has magnitude 1 everywhere
        transmission = np.exp(1j * phase)
        
        # Step 3: Propagate to sensor via Fourier transform (Fraunhofer approx)
        # The far-field diffraction pattern is the FT of the aperture function
        field_at_sensor = np.fft.fftshift(np.fft.fft2(transmission))
        
        # Step 4: Intensity = |field|² (we measure intensity, not field!)
        psf = np.abs(field_at_sensor) ** 2
        
        # Normalize
        psf = psf / psf.sum()
        
        return psf.astype(np.float32)
    
    def caustic_pattern(self, num_features: int = 200, 
                        smoothing: float = 3.0) -> np.ndarray:
        """
        Generate a PSF that mimics caustic patterns seen in real DiffuserCams.
        
        Caustics are the bright, web-like patterns you see when light passes
        through a textured transparent material (like the bottom of a swimming
        pool). DiffuserCam PSFs look similar!
        
        Method: We simulate caustics by refracting random "rays" through a
        bumpy surface and accumulating where they land.
        
        Args:
            num_features: Number of random refractive features
            smoothing: Gaussian smoothing applied to final pattern
            
        Returns:
            psf: Normalized PSF array of shape (size, size)
        """
        # Start with a flat intensity map
        psf = np.zeros((self.size, self.size), dtype=np.float64)
        
        # Generate random "lens" surface heights
        surface = self.rng.randn(self.size, self.size).astype(np.float32)
        surface = cv2.GaussianBlur(surface, (31, 31), 5.0)
        
        # Compute gradients (surface slopes → ray deflections)
        grad_x = np.gradient(surface, axis=1)
        grad_y = np.gradient(surface, axis=0)
        
        # Create coordinate grids for ray positions
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        
        # Deflect rays by the surface gradient (Snell's law approximation)
        deflection_strength = 10.0
        new_x = xx + deflection_strength * grad_x
        new_y = yy + deflection_strength * grad_y
        
        # Clip to valid range
        new_x = np.clip(new_x, 0, self.size - 1).astype(int)
        new_y = np.clip(new_y, 0, self.size - 1).astype(int)
        
        # Accumulate ray landings → caustic pattern
        np.add.at(psf, (new_y, new_x), 1.0)
        
        # Smooth slightly to remove single-pixel artifacts
        psf = cv2.GaussianBlur(psf.astype(np.float32), (0, 0), smoothing)
        
        # Normalize
        psf = psf / psf.sum()
        
        return psf.astype(np.float32)
    
    def load_measured_psf(self, filepath: str) -> np.ndarray:
        """
        Load a real measured PSF from file.
        
        In practice, you calibrate a DiffuserCam by placing a point light
        source in front of it and capturing the resulting pattern.
        
        Args:
            filepath: Path to PSF image file
            
        Returns:
            psf: Normalized PSF array
        """
        psf = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if psf is None:
            raise FileNotFoundError(f"Could not load PSF from: {filepath}")
        
        psf = cv2.resize(psf, (self.size, self.size)).astype(np.float32)
        
        # Normalize
        psf = psf / psf.sum()
        
        return psf


class DiffuserCamForwardModel(nn.Module):
    """
    Differentiable forward model for a DiffuserCam.
    
    Implements: y = H·x + n
    
    Where H is the convolution with the PSF, implemented efficiently
    in the Fourier domain:
        Y = H_fft ⊙ X_fft + N
    
    This module is differentiable (thanks to PyTorch!) so it can be used
    in end-to-end training of reconstruction networks.
    
    WHY FOURIER DOMAIN?
    Direct convolution with an NxN PSF on an NxN image costs O(N⁴).
    FFT-based convolution costs O(N² log N). For 256x256 images:
        Direct: ~4.3 billion operations
        FFT:    ~1.1 million operations  (3900x faster!)
    """
    
    def __init__(self, psf: np.ndarray, noise_std: float = 0.01):
        """
        Args:
            psf: Point Spread Function as numpy array (H x W)
            noise_std: Standard deviation of Gaussian noise to add
        """
        super().__init__()
        
        self.noise_std = noise_std
        
        # Convert PSF to torch tensor and store as buffer
        # (buffers are saved with the model but not trained)
        psf_tensor = torch.from_numpy(psf).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('psf', psf_tensor)
        
        # Pre-compute the FFT of the PSF (it doesn't change!)
        # This saves computation during forward passes
        # We use rfft2 for real-valued inputs (more efficient than fft2)
        psf_for_fft = psf_tensor.squeeze()
        self.register_buffer('psf_fft', torch.fft.rfft2(psf_for_fft))
        
        # Also store the conjugate for use in reconstruction
        self.register_buffer('psf_fft_conj', torch.conj(self.psf_fft))
        
        # Pre-compute |H(f)|² for Wiener filtering
        self.register_buffer('psf_fft_abs_sq', torch.abs(self.psf_fft) ** 2)
    
    def forward(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """
        Apply the forward model: y = PSF * x + noise
        
        Args:
            x: Clean scene image, shape (B, 1, H, W) or (B, 3, H, W)
            add_noise: Whether to add Gaussian noise
            
        Returns:
            y: Simulated sensor measurement, same shape as x
        """
        B, C, H, W = x.shape
        
        # Process each channel independently
        # (Color images: convolve R, G, B separately with same PSF)
        measurements = []
        for c in range(C):
            x_channel = x[:, c, :, :]  # (B, H, W)
            
            # Step 1: FFT of the scene
            X_fft = torch.fft.rfft2(x_channel)
            
            # Step 2: Multiply in Fourier domain (= convolution in spatial domain)
            Y_fft = self.psf_fft.unsqueeze(0) * X_fft  # (B, H, W//2+1)
            
            # Step 3: Inverse FFT to get spatial-domain measurement
            y_channel = torch.fft.irfft2(Y_fft, s=(H, W))
            
            measurements.append(y_channel)
        
        # Stack channels back together
        y = torch.stack(measurements, dim=1)  # (B, C, H, W)
        
        # Step 4: Add noise (simulating sensor noise)
        if add_noise and self.noise_std > 0:
            noise = torch.randn_like(y) * self.noise_std
            y = y + noise
        
        # Step 5: Clip to valid range [0, 1] (sensor can't measure negative light)
        y = torch.clamp(y, 0.0, 1.0)
        
        return y
    
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the adjoint (transpose) of the forward model: A^T · y
        
        In Fourier domain: x_adj = IFFT{ conj(H) · FFT{y} }
        
        The adjoint is NOT the inverse! It's used as a starting point
        for iterative reconstruction algorithms.
        
        Think of it as "approximately un-blurring" — it correlates the
        measurement with the PSF.
        
        Args:
            y: Measurement, shape (B, C, H, W)
            
        Returns:
            x_adj: Adjoint output, same shape as y
        """
        B, C, H, W = y.shape
        
        channels = []
        for c in range(C):
            Y_fft = torch.fft.rfft2(y[:, c, :, :])
            X_adj_fft = self.psf_fft_conj.unsqueeze(0) * Y_fft
            x_adj_channel = torch.fft.irfft2(X_adj_fft, s=(H, W))
            channels.append(x_adj_channel)
        
        return torch.stack(channels, dim=1)
    
    def get_psf_fft(self) -> torch.Tensor:
        """Return the precomputed FFT of the PSF."""
        return self.psf_fft
    
    def get_psf_numpy(self) -> np.ndarray:
        """Return the PSF as a numpy array for visualization."""
        return self.psf.squeeze().cpu().numpy()


def create_test_scene(size: int = 256, scene_type: str = 'natural') -> np.ndarray:
    """
    Create or load a test scene for forward model simulation.
    
    Args:
        size: Image size
        scene_type: Type of test scene
            - 'checkerboard': Sharp edges (good for testing ringing artifacts)
            - 'resolution_target': Multiple frequencies (tests resolution)
            - 'gradient': Smooth gradient (tests low-frequency recovery)
            - 'natural': Download/create a natural-looking test image
            
    Returns:
        scene: Test scene array of shape (size, size), values in [0, 1]
    """
    if scene_type == 'checkerboard':
        # Checkerboard pattern — great for seeing convolution effects
        block_size = size // 8
        scene = np.zeros((size, size), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    scene[i*block_size:(i+1)*block_size, 
                          j*block_size:(j+1)*block_size] = 1.0
                    
    elif scene_type == 'resolution_target':
        # Concentric circles at different frequencies
        scene = np.zeros((size, size), dtype=np.float32)
        cy, cx = size // 2, size // 2
        yy, xx = np.mgrid[0:size, 0:size]
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        
        # Multiple frequency rings
        for freq in [0.5, 1.0, 2.0, 4.0, 8.0]:
            scene += 0.2 * (1 + np.cos(2 * np.pi * freq * r / size)).astype(np.float32)
        
        scene = np.clip(scene / scene.max(), 0, 1)
        
    elif scene_type == 'gradient':
        # Smooth gradient
        scene = np.linspace(0, 1, size, dtype=np.float32)
        scene = np.outer(scene, scene)
        
    elif scene_type == 'natural':
        # Create a synthetic "natural-ish" scene with smooth blobs
        scene = np.zeros((size, size), dtype=np.float32)
        rng = np.random.RandomState(123)
        yy, xx = np.mgrid[0:size, 0:size]
        
        for _ in range(15):
            cx, cy = rng.uniform(0, size, 2)
            sigma = rng.uniform(size * 0.05, size * 0.2)
            amplitude = rng.uniform(0.3, 1.0)
            blob = amplitude * np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*sigma**2))
            scene += blob.astype(np.float32)
        
        scene = np.clip(scene / scene.max(), 0, 1)
    
    else:
        raise ValueError(f"Unknown scene type: {scene_type}")
    
    return scene


# =============================================================================
# Quick test / demo when running this file directly
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  DiffuserCam Forward Model Demo")
    print("=" * 60)
    
    # Generate a PSF
    print("\n[1] Generating PSF (random phase mask)...")
    psf_gen = PSFGenerator(size=256, seed=42)
    psf = psf_gen.random_phase_mask(feature_size=8.0)
    print(f"    PSF shape: {psf.shape}, sum: {psf.sum():.4f}")
    
    # Create a test scene
    print("\n[2] Creating test scene...")
    scene = create_test_scene(size=256, scene_type='natural')
    print(f"    Scene shape: {scene.shape}, range: [{scene.min():.2f}, {scene.max():.2f}]")
    
    # Build the forward model
    print("\n[3] Building forward model...")
    model = DiffuserCamForwardModel(psf, noise_std=0.01)
    
    # Simulate measurement
    print("\n[4] Simulating measurement (y = PSF * x + noise)...")
    scene_tensor = torch.from_numpy(scene).float().unsqueeze(0).unsqueeze(0)
    measurement = model(scene_tensor)
    print(f"    Measurement shape: {measurement.shape}")
    print(f"    Measurement range: [{measurement.min():.4f}, {measurement.max():.4f}]")
    
    # Save results
    print("\n[5] Saving results...")
    cv2.imwrite("results/psf.png", (psf / psf.max() * 255).astype(np.uint8))
    cv2.imwrite("results/scene.png", (scene * 255).astype(np.uint8))
    meas_np = measurement.squeeze().cpu().numpy()
    cv2.imwrite("results/measurement.png", (meas_np / meas_np.max() * 255).astype(np.uint8))
    
    print("\n✅ Done! Check the results/ directory.")
    print("   - psf.png: The simulated Point Spread Function")
    print("   - scene.png: The original test scene")
    print("   - measurement.png: The simulated sensor measurement")

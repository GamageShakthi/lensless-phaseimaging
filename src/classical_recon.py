# =============================================================================
# classical_recon.py — Classical Reconstruction Algorithms
# =============================================================================
#
# This module implements classical (non-learning) approaches to solving
# the inverse problem:  Given y = PSF * x + n, recover x.
#
# METHODS:
#   1. Wiener Deconvolution — Frequency-domain optimal linear filter
#   2. ADMM — Proximal splitting with Total Variation regularization
#
# WHY CLASSICAL FIRST?
#   Understanding classical methods is essential before moving to deep
#   learning approaches. The deep networks we'll build later are actually
#   "unrolled" versions of these classical algorithms!
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import time


class WienerDeconvolution:
    """
    Wiener Deconvolution — The Simplest Reconstruction Method
    
    IDEA:
    If measurement Y(f) = H(f)·X(f), then naively X(f) = Y(f)/H(f).
    But this fails because:
      1. H(f) can be zero at some frequencies → division by zero!
      2. Noise gets amplified at frequencies where H(f) is small
    
    SOLUTION:
    The Wiener filter adds a regularization term:
    
        X̂(f) = H*(f) / (|H(f)|² + λ) · Y(f)
    
    Where:
      - H*(f) = complex conjugate of H(f)
      - |H(f)|² = power spectrum of PSF
      - λ = regularization parameter (noise-to-signal ratio estimate)
    
    When λ = 0: naive inverse (noisy)
    When λ → ∞: over-regularized (blurry)
    Sweet spot: λ ≈ noise_variance / signal_variance
    
    PROS: Very fast (single FFT pair), closed-form solution
    CONS: Assumes Gaussian noise, can't enforce non-negativity, limited quality
    """
    
    def __init__(self, psf_fft: torch.Tensor, regularization: float = 0.01):
        """
        Args:
            psf_fft: FFT of the PSF, shape (H, W//2+1), complex
            regularization: λ parameter (higher = smoother, lower = sharper but noisier)
        """
        self.psf_fft = psf_fft
        self.regularization = regularization
        
        # Pre-compute the Wiener filter kernel
        # W(f) = H*(f) / (|H(f)|² + λ)
        psf_conj = torch.conj(psf_fft)
        psf_power = torch.abs(psf_fft) ** 2
        
        self.wiener_filter = psf_conj / (psf_power + regularization)
    
    def reconstruct(self, measurement: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct image from measurement using Wiener filtering.
        
        Args:
            measurement: Sensor measurement, shape (B, C, H, W)
            
        Returns:
            reconstruction: Recovered image, shape (B, C, H, W)
        """
        B, C, H, W = measurement.shape
        
        channels = []
        for c in range(C):
            # FFT of measurement
            Y_fft = torch.fft.rfft2(measurement[:, c, :, :])
            
            # Apply Wiener filter
            X_fft = self.wiener_filter.unsqueeze(0) * Y_fft
            
            # Inverse FFT to get spatial domain result
            x_recon = torch.fft.irfft2(X_fft, s=(H, W))
            channels.append(x_recon)
        
        result = torch.stack(channels, dim=1)
        
        # Clip to valid range (Wiener can produce values outside [0,1])
        result = torch.clamp(result, 0.0, 1.0)
        
        return result


class TVDenoiser:
    """
    Total Variation (TV) Proximal Operator
    
    TV regularization promotes piecewise-constant images (sharp edges
    with flat regions). It's defined as:
    
        TV(x) = Σ_i √(|∇x_h(i)|² + |∇x_v(i)|²)    (isotropic TV)
    
    Where ∇x_h and ∇x_v are horizontal and vertical gradients.
    
    The proximal operator of TV (used in ADMM) solves:
        prox_{λ·TV}(v) = argmin_x  ½||x - v||² + λ·TV(x)
    
    We approximate this using iterative soft-thresholding on the gradients.
    """
    
    @staticmethod
    def compute_tv(x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Total Variation of an image.
        
        Args:
            x: Image tensor, shape (B, C, H, W)
            
        Returns:
            tv: Scalar TV value
        """
        # Horizontal gradient: difference between adjacent pixels
        diff_h = x[:, :, :, 1:] - x[:, :, :, :-1]
        # Vertical gradient
        diff_v = x[:, :, 1:, :] - x[:, :, :-1, :]
        
        # Isotropic TV: sqrt(|∇h|² + |∇v|²), but we need same size
        # Use anisotropic TV (simpler): |∇h| + |∇v|
        tv = torch.abs(diff_h).sum() + torch.abs(diff_v).sum()
        
        return tv
    
    @staticmethod
    def prox_tv(v: torch.Tensor, lambda_tv: float, 
                num_iters: int = 20) -> torch.Tensor:
        """
        Proximal operator for TV using Chambolle's projection algorithm.
        
        This solves:  argmin_x  ½||x - v||² + λ·TV(x)
        
        Chambolle's algorithm works by solving the dual problem:
        it finds the optimal "gradient field" p and computes x from it.
        
        Args:
            v: Input image to denoise, shape (B, C, H, W)
            lambda_tv: TV regularization strength
            num_iters: Number of iterations for Chambolle's algorithm
            
        Returns:
            x_denoised: TV-denoised image
        """
        B, C, H, W = v.shape
        
        # Initialize dual variables (gradient field)
        p_h = torch.zeros(B, C, H, W - 1, device=v.device)
        p_v = torch.zeros(B, C, H - 1, W, device=v.device)
        
        tau = 0.25  # Step size (must be ≤ 1/8 for convergence)
        
        for _ in range(num_iters):
            # Compute divergence of p (adjoint of gradient)
            div_p = torch.zeros_like(v)
            div_p[:, :, :, 1:] += p_h
            div_p[:, :, :, :-1] -= p_h
            div_p[:, :, 1:, :] += p_v
            div_p[:, :, :-1, :] -= p_v
            
            # Current estimate
            x_hat = v - lambda_tv * div_p
            
            # Compute gradients of x_hat
            grad_h = x_hat[:, :, :, 1:] - x_hat[:, :, :, :-1]
            grad_v = x_hat[:, :, 1:, :] - x_hat[:, :, :-1, :]
            
            # Update dual variables with projection onto unit ball
            # (This is the key step of Chambolle's algorithm)
            p_h_new = p_h + tau * grad_h / lambda_tv
            p_v_new = p_v + tau * grad_v / lambda_tv
            
            # Project: ensure |p| ≤ 1 (dual feasibility)
            # We need to compute the magnitude at each pixel
            # Pad to same size for magnitude computation
            mag_h = torch.abs(p_h_new)
            mag_v = torch.abs(p_v_new)
            
            p_h = p_h_new / torch.clamp(mag_h, min=1.0)
            p_v = p_v_new / torch.clamp(mag_v, min=1.0)
        
        # Final reconstruction from dual solution
        div_p = torch.zeros_like(v)
        div_p[:, :, :, 1:] += p_h
        div_p[:, :, :, :-1] -= p_h
        div_p[:, :, 1:, :] += p_v
        div_p[:, :, :-1, :] -= p_v
        
        x_denoised = v - lambda_tv * div_p
        
        return x_denoised


class ADMMReconstructor:
    """
    ADMM (Alternating Direction Method of Multipliers) Reconstruction
    
    ADMM is a powerful optimization algorithm that splits a hard problem
    into easier sub-problems. For image reconstruction with TV prior:
    
    We want to solve:
        min_x  ½||Ax - y||² + λ·TV(x)
         ^         ^              ^
         |    data fidelity    regularizer
         
    ADMM introduces an auxiliary variable z and reformulates as:
        min_{x,z}  ½||Ax - y||² + λ·TV(z)
        subject to: x = z
    
    The augmented Lagrangian (with dual variable u and penalty ρ):
        L(x, z, u) = ½||Ax - y||² + λ·TV(z) + ρ/2·||x - z + u||²
    
    ADMM alternates three updates:
        1. x-update: min_x  ½||Ax - y||² + ρ/2·||x - z + u||²
           → Quadratic in x, solved in Fourier domain!
        2. z-update: min_z  λ·TV(z) + ρ/2·||x - z + u||²
           → This is the TV proximal operator!
        3. u-update: u ← u + (x - z)
           → Simple gradient ascent on dual variable
    
    WHY ADMM IS GREAT:
    - Decomposes hard problem into easy sub-problems
    - Each sub-problem has an efficient solution
    - Guaranteed convergence for convex problems
    - The deep learning version "unrolls" these iterations into a network!
    """
    
    def __init__(self, 
                 psf_fft: torch.Tensor,
                 psf_fft_conj: torch.Tensor,
                 psf_fft_abs_sq: torch.Tensor,
                 rho: float = 1.0,
                 lambda_tv: float = 0.01,
                 num_iters: int = 50,
                 tv_inner_iters: int = 20,
                 verbose: bool = True):
        """
        Args:
            psf_fft: FFT of PSF
            psf_fft_conj: Conjugate of PSF FFT
            psf_fft_abs_sq: |H(f)|² — squared magnitude of PSF FFT
            rho: ADMM penalty parameter (controls convergence speed)
            lambda_tv: TV regularization weight
            num_iters: Number of ADMM iterations
            tv_inner_iters: Iterations for TV proximal operator
            verbose: Print progress
        """
        self.psf_fft = psf_fft
        self.psf_fft_conj = psf_fft_conj
        self.psf_fft_abs_sq = psf_fft_abs_sq
        self.rho = rho
        self.lambda_tv = lambda_tv
        self.num_iters = num_iters
        self.tv_inner_iters = tv_inner_iters
        self.verbose = verbose
    
    def reconstruct(self, measurement: torch.Tensor,
                    ground_truth: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[dict]]:
        """
        Reconstruct image using ADMM with TV regularization.
        
        Args:
            measurement: Sensor measurement, shape (B, C, H, W)
            ground_truth: Optional GT for tracking metrics during iterations
            
        Returns:
            x: Reconstructed image, shape (B, C, H, W)
            history: List of dicts with convergence metrics per iteration
        """
        B, C, H, W = measurement.shape
        device = measurement.device
        
        # Initialize variables
        x = torch.zeros_like(measurement)  # Primal variable
        z = torch.zeros_like(measurement)  # Auxiliary variable
        u = torch.zeros_like(measurement)  # Dual variable (scaled form)
        
        # Pre-compute FFT of measurement (doesn't change across iterations)
        Y_fft_channels = []
        for c in range(C):
            Y_fft_channels.append(torch.fft.rfft2(measurement[:, c, :, :]))
        
        # Pre-compute the denominator for x-update (doesn't change!)
        # x-update in Fourier: X(f) = (H*(f)·Y(f) + ρ·FFT{z-u}) / (|H(f)|² + ρ)
        denom = self.psf_fft_abs_sq + self.rho  # shape: (H, W//2+1)
        
        history = []
        start_time = time.time()
        
        for iteration in range(self.num_iters):
            # ========================================================
            # Step 1: x-update (data fidelity + quadratic penalty)
            # ========================================================
            # Solve: min_x ½||Ax - y||² + ρ/2·||x - (z - u)||²
            # In Fourier domain, this has a closed-form solution!
            
            x_channels = []
            for c in range(C):
                # FFT of (z - u) for the penalty term
                v_fft = torch.fft.rfft2((z - u)[:, c, :, :])
                
                # Numerator: H*(f)·Y(f) + ρ·V(f)
                numerator = self.psf_fft_conj.unsqueeze(0) * Y_fft_channels[c] + self.rho * v_fft
                
                # Solve: X(f) = numerator / denominator
                X_fft = numerator / denom.unsqueeze(0)
                
                # Back to spatial domain
                x_channel = torch.fft.irfft2(X_fft, s=(H, W))
                x_channels.append(x_channel)
            
            x = torch.stack(x_channels, dim=1)
            
            # ========================================================
            # Step 2: z-update (TV proximal operator)
            # ========================================================
            # Solve: min_z λ·TV(z) + ρ/2·||z - (x + u)||²
            # This is the proximal operator of (λ/ρ)·TV evaluated at (x + u)
            
            z = TVDenoiser.prox_tv(
                x + u, 
                lambda_tv=self.lambda_tv / self.rho,
                num_iters=self.tv_inner_iters
            )
            
            # Enforce non-negativity (physical constraint: intensities ≥ 0)
            z = torch.clamp(z, 0.0, 1.0)
            
            # ========================================================
            # Step 3: u-update (dual variable ascent)
            # ========================================================
            # u ← u + (x - z)
            # This enforces the constraint x = z over iterations
            
            u = u + (x - z)
            
            # ========================================================
            # Track convergence
            # ========================================================
            primal_residual = torch.norm(x - z).item()
            
            iter_info = {
                'iteration': iteration,
                'primal_residual': primal_residual,
                'time': time.time() - start_time,
            }
            
            # If ground truth is available, compute quality metrics
            if ground_truth is not None:
                mse = F.mse_loss(z, ground_truth).item()
                psnr = -10 * np.log10(mse + 1e-10)
                iter_info['psnr'] = psnr
            
            history.append(iter_info)
            
            if self.verbose and (iteration % 10 == 0 or iteration == self.num_iters - 1):
                msg = f"  ADMM iter {iteration:3d}/{self.num_iters}: "
                msg += f"primal_res={primal_residual:.6f}"
                if 'psnr' in iter_info:
                    msg += f", PSNR={iter_info['psnr']:.2f} dB"
                print(msg)
        
        return z, history


class GradientDescentReconstructor:
    """
    Simple Gradient Descent Reconstruction (for comparison).
    
    Minimizes: ½||Ax - y||² + λ·TV(x)
    
    Using gradient descent: x_{k+1} = x_k - α·∇f(x_k)
    
    Where: ∇(½||Ax - y||²) = A^T(Ax - y) = H**(Hx - y) in Fourier domain
    
    This is simpler but slower than ADMM. Included for educational comparison.
    """
    
    def __init__(self,
                 psf_fft: torch.Tensor,
                 psf_fft_conj: torch.Tensor,
                 step_size: float = 0.01,
                 lambda_tv: float = 0.001,
                 num_iters: int = 100):
        self.psf_fft = psf_fft
        self.psf_fft_conj = psf_fft_conj
        self.step_size = step_size
        self.lambda_tv = lambda_tv
        self.num_iters = num_iters
    
    def reconstruct(self, measurement: torch.Tensor) -> torch.Tensor:
        """Simple gradient descent reconstruction."""
        B, C, H, W = measurement.shape
        
        # Initialize with zeros (or adjoint)
        x = torch.zeros_like(measurement, requires_grad=True)
        
        optimizer = torch.optim.Adam([x], lr=self.step_size)
        
        for i in range(self.num_iters):
            optimizer.zero_grad()
            
            # Forward model: Ax
            Ax_channels = []
            for c in range(C):
                X_fft = torch.fft.rfft2(x[:, c, :, :])
                AX_fft = self.psf_fft.unsqueeze(0) * X_fft
                ax_c = torch.fft.irfft2(AX_fft, s=(H, W))
                Ax_channels.append(ax_c)
            Ax = torch.stack(Ax_channels, dim=1)
            
            # Data fidelity loss
            data_loss = 0.5 * F.mse_loss(Ax, measurement, reduction='sum')
            
            # TV regularization
            tv_loss = self.lambda_tv * TVDenoiser.compute_tv(x)
            
            # Total loss
            loss = data_loss + tv_loss
            loss.backward()
            optimizer.step()
            
            # Project to valid range
            with torch.no_grad():
                x.clamp_(0.0, 1.0)
            
            if i % 20 == 0:
                print(f"  GD iter {i}: loss={loss.item():.4f}, "
                      f"data={data_loss.item():.4f}, tv={tv_loss.item():.4f}")
        
        return x.detach()


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    from forward_model import PSFGenerator, DiffuserCamForwardModel, create_test_scene
    
    print("=" * 60)
    print("  Classical Reconstruction Demo")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create PSF and forward model
    psf_gen = PSFGenerator(size=128, seed=42)
    psf = psf_gen.random_phase_mask(feature_size=5.0)
    model = DiffuserCamForwardModel(psf, noise_std=0.01)
    model = model.to(device)
    
    # Create test scene and measurement
    scene = create_test_scene(size=128, scene_type='natural')
    scene_tensor = torch.from_numpy(scene).float().unsqueeze(0).unsqueeze(0).to(device)
    measurement = model(scene_tensor)
    
    # --- Wiener Deconvolution ---
    print("\n[1] Wiener Deconvolution...")
    t0 = time.time()
    wiener = WienerDeconvolution(model.psf_fft, regularization=0.001)
    wiener_recon = wiener.reconstruct(measurement)
    t1 = time.time()
    
    mse = F.mse_loss(wiener_recon, scene_tensor).item()
    psnr = -10 * np.log10(mse + 1e-10)
    print(f"    Time: {t1-t0:.3f}s, PSNR: {psnr:.2f} dB")
    
    # --- ADMM ---
    print("\n[2] ADMM with TV regularization...")
    t0 = time.time()
    admm = ADMMReconstructor(
        model.psf_fft, model.psf_fft_conj, model.psf_fft_abs_sq,
        rho=1.0, lambda_tv=0.005, num_iters=30, verbose=True
    )
    admm_recon, history = admm.reconstruct(measurement, ground_truth=scene_tensor)
    t1 = time.time()
    
    mse = F.mse_loss(admm_recon, scene_tensor).item()
    psnr = -10 * np.log10(mse + 1e-10)
    print(f"    Time: {t1-t0:.3f}s, Final PSNR: {psnr:.2f} dB")
    
    print("\n✅ Classical reconstruction complete!")

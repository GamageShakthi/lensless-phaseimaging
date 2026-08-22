# =============================================================================
# utils.py — Utility Functions for Visualization, I/O, and Metrics
# =============================================================================

import torch
import numpy as np
import cv2
import os
from typing import Optional, List, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (for saving figures)
import matplotlib.pyplot as plt


def load_image(filepath: str, size: Optional[int] = None, 
               grayscale: bool = True) -> np.ndarray:
    """
    Load an image from file and preprocess for reconstruction.
    
    Args:
        filepath: Path to image file
        size: Resize to (size, size) if specified
        grayscale: Convert to grayscale if True
        
    Returns:
        image: Numpy array in [0, 1] range
    """
    if grayscale:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")
    
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img


def save_image(image: np.ndarray, filepath: str):
    """
    Save an image to file.
    
    Args:
        image: Numpy array, either [0,1] float or [0,255] uint8
        filepath: Output path
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', 
                exist_ok=True)
    
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(filepath, image)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a numpy array for visualization.
    
    Args:
        tensor: Shape (B, C, H, W) or (C, H, W) or (H, W)
        
    Returns:
        numpy array in (H, W) or (H, W, C) format
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch element
    
    img = tensor.detach().cpu().numpy()
    
    if img.shape[0] in [1, 3]:  # C, H, W → H, W, C
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = img.squeeze(2)
    
    return np.clip(img, 0, 1)


def numpy_to_tensor(image: np.ndarray, 
                    device: str = 'cpu') -> torch.Tensor:
    """
    Convert numpy image to PyTorch tensor.
    
    Args:
        image: Shape (H, W) or (H, W, C), values in [0, 1]
        device: Target device
        
    Returns:
        tensor: Shape (1, C, H, W)
    """
    if image.ndim == 2:
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    else:
        tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    
    return tensor.to(device)


def plot_comparison(images: List[np.ndarray], 
                    titles: List[str],
                    figsize: Tuple[int, int] = None,
                    save_path: Optional[str] = None,
                    suptitle: Optional[str] = None):
    """
    Plot multiple images side by side for comparison.
    
    Args:
        images: List of numpy images
        titles: List of titles for each image
        figsize: Figure size (width, height)
        save_path: If specified, save the figure
        suptitle: Super title for the whole figure
    """
    n = len(images)
    if figsize is None:
        figsize = (4 * n, 4)
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                    exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved comparison figure to: {save_path}")
    
    plt.close()


def plot_convergence(history: List[dict], 
                     metric: str = 'psnr',
                     save_path: Optional[str] = None):
    """
    Plot convergence curve from ADMM iteration history.
    
    Args:
        history: List of dicts from ADMM reconstructor
        metric: Which metric to plot ('psnr', 'primal_residual')
        save_path: If specified, save the figure
    """
    iterations = [h['iteration'] for h in history]
    values = [h.get(metric, 0) for h in history]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, values, 'b-o', markersize=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(metric.upper().replace('_', ' '))
    ax.set_title(f'ADMM Convergence: {metric}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_psf_analysis(psf: np.ndarray, save_path: Optional[str] = None):
    """
    Comprehensive PSF visualization and analysis.
    
    Shows:
    1. PSF in spatial domain
    2. PSF in log scale (to see faint features)
    3. PSF frequency response (MTF)
    4. Cross-section profile
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # 1. Spatial domain (linear scale)
    axes[0, 0].imshow(psf, cmap='hot')
    axes[0, 0].set_title('PSF (Linear Scale)')
    axes[0, 0].axis('off')
    
    # 2. Spatial domain (log scale)
    psf_log = np.log10(psf + 1e-10)
    axes[0, 1].imshow(psf_log, cmap='hot')
    axes[0, 1].set_title('PSF (Log Scale)')
    axes[0, 1].axis('off')
    
    # 3. Modulation Transfer Function (frequency response)
    mtf = np.abs(np.fft.fftshift(np.fft.fft2(psf)))
    mtf_log = np.log10(mtf + 1e-10)
    axes[1, 0].imshow(mtf_log, cmap='viridis')
    axes[1, 0].set_title('MTF (Log Scale)')
    axes[1, 0].axis('off')
    
    # 4. Cross-section through center
    center = psf.shape[0] // 2
    axes[1, 1].plot(psf[center, :], 'b-', label='Horizontal')
    axes[1, 1].plot(psf[:, center], 'r--', label='Vertical')
    axes[1, 1].set_title('PSF Cross-Section')
    axes[1, 1].set_xlabel('Pixel')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Point Spread Function Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_loss_curves(train_losses: List[float], 
                     val_losses: Optional[List[float]] = None,
                     save_path: Optional[str] = None):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(train_losses, 'b-', label='Training Loss', alpha=0.8)
    if val_losses:
        ax.plot(val_losses, 'r-', label='Validation Loss', alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def create_gif_from_intermediates(intermediates: List[np.ndarray],
                                   save_path: str,
                                   duration_ms: int = 200):
    """
    Create a GIF showing the reconstruction evolving over iterations.
    
    Args:
        intermediates: List of intermediate reconstruction images
        save_path: Output GIF path
        duration_ms: Duration per frame in milliseconds
    """
    try:
        from PIL import Image
        
        frames = []
        for img in intermediates:
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(img))
        
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0
        )
        print(f"  Saved animation to: {save_path}")
    except ImportError:
        print("  Warning: PIL not available. Cannot create GIF.")


def print_metrics(prediction: torch.Tensor, target: torch.Tensor, 
                  method_name: str = ""):
    """Print PSNR and SSIM metrics."""
    from src.losses import compute_psnr, compute_ssim
    
    psnr = compute_psnr(prediction, target)
    ssim = compute_ssim(prediction, target)
    
    prefix = f"[{method_name}] " if method_name else ""
    print(f"  {prefix}PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
    
    return {'psnr': psnr, 'ssim': ssim}

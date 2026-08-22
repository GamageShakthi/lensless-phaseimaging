# =============================================================================
# train.py — Training Loop for the Unrolled ADMM Network
# =============================================================================
#
# This module handles the full training pipeline:
#   1. Dataset creation (synthetic measurements from natural images)
#   2. Training loop with loss computation
#   3. Validation and checkpointing
#   4. Learning rate scheduling
#
# TRAINING STRATEGY:
#   We generate synthetic training data on-the-fly:
#   - Take a natural image x
#   - Simulate measurement y = PSF * x + noise
#   - Train the network to reconstruct x from y
#   - Loss = MSE(x̂, x) + λ_tv·TV(x̂) + λ_perc·Perceptual(x̂, x)
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import time
from typing import Optional, Tuple, Dict, List

from src.forward_model import DiffuserCamForwardModel, PSFGenerator
from src.deep_recon import UnrolledADMMNetwork
from src.losses import CombinedLoss, compute_psnr, compute_ssim
from src.utils import plot_loss_curves, plot_comparison, tensor_to_numpy


class SyntheticLenslessDataset(Dataset):
    """
    Dataset that generates synthetic lensless measurements on-the-fly.
    
    For each sample:
    1. Load (or generate) a clean image
    2. Apply the forward model (convolution + noise)
    3. Return (measurement, clean_image) pair
    
    This means we don't need real DiffuserCam captures for training!
    We just need natural images + a known PSF.
    """
    
    def __init__(self, 
                 image_dir: Optional[str] = None,
                 psf: Optional[np.ndarray] = None,
                 num_samples: int = 500,
                 image_size: int = 128,
                 noise_std: float = 0.01,
                 grayscale: bool = True):
        """
        Args:
            image_dir: Directory containing training images. 
                      If None, generates synthetic images.
            psf: Point Spread Function. If None, generates one.
            num_samples: Number of training samples
            image_size: Size of images (image_size x image_size)
            noise_std: Noise level for forward model
            grayscale: Use grayscale images
        """
        self.image_size = image_size
        self.noise_std = noise_std
        self.num_samples = num_samples
        self.grayscale = grayscale
        
        # Load or generate PSF
        if psf is None:
            psf_gen = PSFGenerator(size=image_size, seed=42)
            self.psf = psf_gen.random_phase_mask(feature_size=5.0)
        else:
            self.psf = psf
        
        # Build forward model
        self.forward_model = DiffuserCamForwardModel(self.psf, noise_std=noise_std)
        
        # Load or generate clean images
        self.images = []
        if image_dir and os.path.exists(image_dir):
            # Load real images
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = [
                os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if os.path.splitext(f)[1].lower() in extensions
            ]
            for fpath in image_files[:num_samples]:
                flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
                img = cv2.imread(fpath, flag)
                if img is not None:
                    img = cv2.resize(img, (image_size, image_size))
                    img = img.astype(np.float32) / 255.0
                    self.images.append(img)
        
        # If not enough real images, pad with synthetic ones
        if len(self.images) < num_samples:
            rng = np.random.RandomState(12345)
            for i in range(num_samples - len(self.images)):
                img = self._generate_synthetic_image(rng)
                self.images.append(img)
    
    def _generate_synthetic_image(self, rng: np.random.RandomState) -> np.ndarray:
        """Generate a synthetic natural-looking image with smooth blobs."""
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        yy, xx = np.mgrid[0:self.image_size, 0:self.image_size]
        
        # Add random smooth blobs
        num_blobs = rng.randint(5, 20)
        for _ in range(num_blobs):
            cx = rng.uniform(0, self.image_size)
            cy = rng.uniform(0, self.image_size)
            sigma = rng.uniform(self.image_size * 0.03, self.image_size * 0.15)
            amplitude = rng.uniform(0.2, 1.0)
            blob = amplitude * np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*sigma**2))
            img += blob
        
        # Sometimes add geometric shapes
        if rng.random() > 0.5:
            # Add rectangle
            x1, y1 = rng.randint(0, self.image_size//2, 2)
            x2, y2 = x1 + rng.randint(10, self.image_size//3), y1 + rng.randint(10, self.image_size//3)
            x2, y2 = min(x2, self.image_size), min(y2, self.image_size)
            img[y1:y2, x1:x2] += rng.uniform(0.3, 0.8)
        
        return np.clip(img / max(img.max(), 1e-5), 0, 1).astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a (measurement, ground_truth) pair.
        
        Returns:
            measurement: shape (1, H, W)
            ground_truth: shape (1, H, W)
        """
        clean = self.images[idx]
        
        # Add channel dimension if grayscale
        if clean.ndim == 2:
            clean = clean[np.newaxis, :, :]  # (1, H, W)
        
        clean_tensor = torch.from_numpy(clean).float()
        
        # Apply forward model
        with torch.no_grad():
            measurement = self.forward_model(clean_tensor.unsqueeze(0))
            measurement = measurement.squeeze(0)
        
        return measurement, clean_tensor


class Trainer:
    """
    Training manager for the unrolled ADMM network.
    
    Handles:
    - Training loop with gradient accumulation
    - Validation and metrics
    - Checkpointing (save/load model)
    - Learning rate scheduling
    - Logging
    """
    
    def __init__(self,
                 model: UnrolledADMMNetwork,
                 train_dataset: SyntheticLenslessDataset,
                 val_dataset: Optional[SyntheticLenslessDataset] = None,
                 learning_rate: float = 1e-4,
                 batch_size: int = 4,
                 lambda_tv: float = 0.01,
                 lambda_perceptual: float = 0.0,
                 use_perceptual: bool = False,
                 checkpoint_dir: str = 'pretrained',
                 device: str = 'auto'):
        """
        Args:
            model: The unrolled ADMM network
            train_dataset: Training dataset
            val_dataset: Validation dataset
            learning_rate: Initial learning rate
            batch_size: Batch size for training
            lambda_tv: TV loss weight
            lambda_perceptual: Perceptual loss weight
            use_perceptual: Whether to use VGG perceptual loss
            checkpoint_dir: Directory to save model checkpoints
            device: 'cuda', 'cpu', or 'auto'
        """
        # Device selection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Training on: {self.device}")
        
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True
            )
        
        # Loss function
        self.criterion = CombinedLoss(
            lambda_data=1.0,
            lambda_tv=lambda_tv,
            lambda_perceptual=lambda_perceptual,
            use_perceptual=use_perceptual,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        
        # Learning rate scheduler
        # ReduceLROnPlateau: reduce LR when validation loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # History tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (measurements, targets) in enumerate(self.train_loader):
            measurements = measurements.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions = self.model(measurements)
            
            # Compute loss
            losses = self.criterion(predictions, targets, measurements)
            loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                psnr = compute_psnr(predictions.detach(), targets)
                print(f"  Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.6f} | PSNR: {psnr:.2f} dB")
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0, 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0
        
        for measurements, targets in self.val_loader:
            measurements = measurements.to(self.device)
            targets = targets.to(self.device)
            
            predictions = self.model(measurements)
            losses = self.criterion(predictions, targets)
            
            total_loss += losses['total'].item()
            total_psnr += compute_psnr(predictions, targets)
            total_ssim += compute_ssim(predictions[:, 0:1], targets[:, 0:1])
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_psnr = total_psnr / max(num_batches, 1)
        avg_ssim = total_ssim / max(num_batches, 1)
        
        return avg_loss, avg_psnr, avg_ssim
    
    def train(self, num_epochs: int = 50, save_every: int = 10):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
        """
        print("=" * 60)
        print("  Starting Training")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Model parameters: {self.model.count_parameters():,}")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_psnr, val_ssim = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss if val_loss > 0 else train_loss)
            
            elapsed = time.time() - t0
            
            print(f"\n{'='*60}")
            print(f"  Epoch {epoch}/{num_epochs} ({elapsed:.1f}s)")
            print(f"  Train Loss: {train_loss:.6f}")
            if val_loss > 0:
                print(f"  Val Loss:   {val_loss:.6f} | PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}")
            
            # Save best model
            if val_loss > 0 and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(os.path.join(self.checkpoint_dir, 'best_model.pth'))
                print(f"  ★ New best model saved! (val_loss={val_loss:.6f})")
            
            # Regular checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, f'checkpoint_epoch{epoch}.pth')
                )
            
            print(f"{'='*60}\n")
        
        # Save final model and loss curves
        self.save_checkpoint(os.path.join(self.checkpoint_dir, 'final_model.pth'))
        plot_loss_curves(
            self.train_losses, 
            self.val_losses if self.val_losses[0] > 0 else None,
            save_path=os.path.join(self.checkpoint_dir, 'loss_curves.png')
        )
        
        print("Training complete!")
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Loaded checkpoint from: {filepath}")

# =============================================================================
# losses.py — Physics-Guided Loss Functions
# =============================================================================
#
# Loss functions are how we tell the neural network what "good" means.
# In computational imaging, we combine multiple loss terms:
#
#   L_total = L_data + λ_tv · L_TV + λ_perc · L_perceptual
#
# Each term captures a different aspect of image quality:
#   - L_data: Pixel-level accuracy (MSE)
#   - L_TV: Edge preservation & smoothness (Total Variation)
#   - L_perceptual: Visual/perceptual quality (VGG features)
#
# WHY MULTIPLE LOSSES?
#   MSE alone produces blurry results because it averages over possibilities.
#   TV adds sharpness but can create "staircase" artifacts.
#   Perceptual loss captures high-level structure but can add texture artifacts.
#   The combination balances all three!
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Dict


class DataFidelityLoss(nn.Module):
    """
    Data Fidelity Loss — Measures how well reconstruction matches measurement.
    
    Basic version: L = ||x̂ - x_true||²  (MSE between reconstruction and GT)
    
    Physics-informed version: L = ||A·x̂ - y||²  (measurement consistency)
    This ensures the reconstruction, when re-imaged through the forward model,
    produces the same measurement we observed.
    """
    
    def __init__(self, loss_type: str = 'mse'):
        """
        Args:
            loss_type: 'mse' (Mean Squared Error) or 'l1' (Mean Absolute Error)
                      MSE penalizes large errors more, L1 is more robust to outliers
        """
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, prediction: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction: Reconstructed image, shape (B, C, H, W)
            target: Ground truth image, shape (B, C, H, W)
            
        Returns:
            loss: Scalar loss value
        """
        if self.loss_type == 'mse':
            return F.mse_loss(prediction, target)
        elif self.loss_type == 'l1':
            return F.l1_loss(prediction, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class TotalVariationLoss(nn.Module):
    """
    Total Variation (TV) Loss — Promotes Piecewise-Smooth Images
    
    TV measures the total amount of "variation" in an image. Minimizing it
    encourages the network to produce smooth regions with sharp edges.
    
    Isotropic TV:   TV(x) = Σ √(|∇x_h|² + |∇x_v|²)
    Anisotropic TV: TV(x) = Σ (|∇x_h| + |∇x_v|)
    
    WHEN TO USE:
    - Natural images are mostly smooth with edges → TV is appropriate
    - Too much TV → "cartoon-like" appearance
    - Too little TV → noisy output
    
    INTUITION:
    Think of TV as a "budget" for edges. The network must spend this budget
    on real edges in the scene, not on noise artifacts.
    """
    
    def __init__(self, mode: str = 'isotropic'):
        """
        Args:
            mode: 'isotropic' or 'anisotropic'
        """
        super().__init__()
        self.mode = mode
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute TV loss.
        
        Args:
            x: Image tensor, shape (B, C, H, W)
            
        Returns:
            tv_loss: Scalar TV value (averaged over batch)
        """
        # Compute image gradients
        diff_h = x[:, :, :, 1:] - x[:, :, :, :-1]  # Horizontal differences
        diff_v = x[:, :, 1:, :] - x[:, :, :-1, :]  # Vertical differences
        
        if self.mode == 'isotropic':
            # Isotropic: sqrt(|∇h|² + |∇v|²)
            # Need to handle the size mismatch between diff_h and diff_v
            # Crop to same size
            min_h = min(diff_h.shape[2], diff_v.shape[2])
            min_w = min(diff_h.shape[3], diff_v.shape[3])
            
            dh = diff_h[:, :, :min_h, :min_w]
            dv = diff_v[:, :, :min_h, :min_w]
            
            # Smooth L1 approximation to avoid non-differentiability at 0
            tv = torch.sqrt(dh ** 2 + dv ** 2 + 1e-8).mean()
        
        elif self.mode == 'anisotropic':
            # Anisotropic: |∇h| + |∇v| (simpler, separable)
            tv = torch.abs(diff_h).mean() + torch.abs(diff_v).mean()
        
        else:
            raise ValueError(f"Unknown TV mode: {self.mode}")
        
        return tv


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss (VGG Feature Loss) — Measures Visual Similarity
    
    Instead of comparing images pixel-by-pixel (MSE), we compare their
    high-level features extracted by a pre-trained VGG network.
    
    WHY PERCEPTUAL LOSS?
    Two images can have the same MSE but look very different:
    - A slightly shifted image has high MSE but looks identical
    - A blurry image can have low MSE but looks terrible
    
    Perceptual loss captures "does this LOOK like the target?" by comparing
    intermediate feature maps from VGG16:
    
        L_perceptual = Σ_l ||φ_l(x̂) - φ_l(x_true)||²
    
    Where φ_l is the feature map at layer l of VGG16.
    
    LAYER SELECTION:
    - Early layers (relu1_2): Match textures and edges
    - Middle layers (relu2_2, relu3_3): Match patterns and structure
    - Deep layers (relu4_3): Match semantic content
    
    We use a combination for balanced perceptual quality.
    """
    
    def __init__(self, layers: Optional[list] = None, 
                 normalize_input: bool = True):
        """
        Args:
            layers: Which VGG layers to extract features from.
                   Default: relu1_2, relu2_2, relu3_3, relu4_3
            normalize_input: Whether to normalize input to VGG's expected range
        """
        super().__init__()
        
        self.normalize_input = normalize_input
        
        # Load pre-trained VGG16 features
        # We only need the feature extractor, not the classifier
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Define which layers to use for feature extraction
        # VGG16 feature layers:
        #   0-3:   conv1_1, relu, conv1_2, relu, pool    → relu1_2 = index 3
        #   4-8:   conv2_1, relu, conv2_2, relu, pool    → relu2_2 = index 8
        #   9-15:  conv3_1...conv3_3, relu, pool          → relu3_3 = index 15
        #   16-22: conv4_1...conv4_3, relu, pool          → relu4_3 = index 22
        
        if layers is None:
            self.layer_indices = [3, 8, 15, 22]
        else:
            self.layer_indices = layers
        
        # Extract VGG features up to the deepest layer we need
        max_layer = max(self.layer_indices) + 1
        self.features = nn.Sequential(*list(vgg.features.children())[:max_layer])
        
        # Freeze VGG weights — we don't want to train VGG!
        for param in self.features.parameters():
            param.requires_grad = False
        
        # VGG normalization (ImageNet statistics)
        self.register_buffer('mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', 
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to VGG's expected range."""
        return (x - self.mean) / self.std
    
    def _extract_features(self, x: torch.Tensor) -> list:
        """Extract feature maps at specified layers."""
        # If grayscale, repeat to 3 channels (VGG expects RGB)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        if self.normalize_input:
            x = self._normalize(x)
        
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        
        return features
    
    def forward(self, prediction: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            prediction: Reconstructed image
            target: Ground truth image
            
        Returns:
            loss: Perceptual loss (sum of MSE at each feature layer)
        """
        pred_features = self._extract_features(prediction)
        target_features = self._extract_features(target)
        
        loss = 0.0
        for pf, tf in zip(pred_features, target_features):
            # MSE between feature maps at each layer
            loss += F.mse_loss(pf, tf)
        
        return loss


class MeasurementConsistencyLoss(nn.Module):
    """
    Measurement Consistency Loss — Physics-Informed Constraint
    
    Instead of just comparing x̂ to x_true, we also check:
    "If I re-image x̂ through the camera, do I get back the measurement y?"
    
        L_consistency = ||A·x̂ - y||²
    
    This doesn't require ground truth! It uses the physics of the imaging
    system as a self-supervised constraint.
    
    WHY THIS MATTERS:
    - Can be used for fine-tuning without ground truth
    - Ensures physical plausibility of the reconstruction
    - Acts as a regularizer that prevents hallucination
    """
    
    def __init__(self, psf_fft: torch.Tensor):
        """
        Args:
            psf_fft: FFT of the PSF
        """
        super().__init__()
        self.register_buffer('psf_fft', psf_fft)
    
    def forward(self, reconstruction: torch.Tensor, 
                measurement: torch.Tensor) -> torch.Tensor:
        """
        Compute measurement consistency.
        
        Args:
            reconstruction: Reconstructed image x̂, shape (B, C, H, W)
            measurement: Original measurement y, shape (B, C, H, W)
            
        Returns:
            loss: ||A·x̂ - y||²
        """
        B, C, H, W = reconstruction.shape
        
        # Re-image through forward model
        re_measured_channels = []
        for c in range(C):
            X_fft = torch.fft.rfft2(reconstruction[:, c, :, :])
            AX_fft = self.psf_fft.unsqueeze(0) * X_fft
            ax = torch.fft.irfft2(AX_fft, s=(H, W))
            re_measured_channels.append(ax)
        
        re_measured = torch.stack(re_measured_channels, dim=1)
        
        return F.mse_loss(re_measured, measurement)


class CombinedLoss(nn.Module):
    """
    Combined Loss — Weighted Sum of All Loss Terms
    
    L_total = λ_data · L_data + λ_tv · L_TV + λ_perc · L_perceptual + λ_consist · L_consistency
    
    The weights (λ) control the trade-off between different objectives:
    - High λ_data: Prioritize pixel accuracy (but may be blurry)
    - High λ_tv: Prioritize sharpness (but may look "cartoon-like")
    - High λ_perc: Prioritize visual quality (but may add artifacts)
    - High λ_consist: Prioritize physical consistency
    
    Finding the right balance is part of the art of computational imaging!
    """
    
    def __init__(self,
                 lambda_data: float = 1.0,
                 lambda_tv: float = 0.01,
                 lambda_perceptual: float = 0.1,
                 lambda_consistency: float = 0.0,
                 data_loss_type: str = 'l1',
                 tv_mode: str = 'isotropic',
                 psf_fft: Optional[torch.Tensor] = None,
                 use_perceptual: bool = True):
        """
        Args:
            lambda_data: Weight for data fidelity loss
            lambda_tv: Weight for TV regularization
            lambda_perceptual: Weight for perceptual loss
            lambda_consistency: Weight for measurement consistency
            data_loss_type: 'mse' or 'l1'
            tv_mode: 'isotropic' or 'anisotropic'
            psf_fft: Required if lambda_consistency > 0
            use_perceptual: Whether to use perceptual loss (requires VGG download)
        """
        super().__init__()
        
        self.lambda_data = lambda_data
        self.lambda_tv = lambda_tv
        self.lambda_perceptual = lambda_perceptual
        self.lambda_consistency = lambda_consistency
        
        # Initialize loss components
        self.data_loss = DataFidelityLoss(data_loss_type)
        self.tv_loss = TotalVariationLoss(tv_mode)
        
        self.perceptual_loss = None
        if use_perceptual and lambda_perceptual > 0:
            self.perceptual_loss = PerceptualLoss()
        
        self.consistency_loss = None
        if lambda_consistency > 0 and psf_fft is not None:
            self.consistency_loss = MeasurementConsistencyLoss(psf_fft)
    
    def forward(self, 
                prediction: torch.Tensor,
                target: torch.Tensor,
                measurement: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            prediction: Reconstructed image
            target: Ground truth image
            measurement: Original measurement (for consistency loss)
            
        Returns:
            losses: Dictionary with individual and total loss values
        """
        losses = {}
        
        # Data fidelity
        losses['data'] = self.data_loss(prediction, target)
        total = self.lambda_data * losses['data']
        
        # Total Variation
        losses['tv'] = self.tv_loss(prediction)
        total = total + self.lambda_tv * losses['tv']
        
        # Perceptual loss
        if self.perceptual_loss is not None:
            losses['perceptual'] = self.perceptual_loss(prediction, target)
            total = total + self.lambda_perceptual * losses['perceptual']
        
        # Measurement consistency
        if self.consistency_loss is not None and measurement is not None:
            losses['consistency'] = self.consistency_loss(prediction, measurement)
            total = total + self.lambda_consistency * losses['consistency']
        
        losses['total'] = total
        
        return losses


# =============================================================================
# Helper function for PSNR and SSIM metrics
# =============================================================================

def compute_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    PSNR = 10 · log₁₀(MAX² / MSE)
    
    Higher = better. Typical values:
    - < 20 dB: Poor quality
    - 20-30 dB: Acceptable
    - 30-40 dB: Good
    - > 40 dB: Excellent
    """
    mse = F.mse_loss(prediction, target).item()
    if mse < 1e-10:
        return 100.0  # Perfect reconstruction
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def compute_ssim(prediction: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    SSIM compares images based on:
    - Luminance (mean brightness)
    - Contrast (variance)
    - Structure (correlation)
    
    Range: [-1, 1], where 1 = identical images
    
    SSIM is better than PSNR for measuring perceptual quality because
    it's more aligned with human visual perception.
    """
    C1 = 0.01 ** 2  # Stabilization constants
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = torch.outer(g, g)
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0)
    
    # Ensure same device
    window = window.to(prediction.device)
    
    # Compute means
    mu1 = F.conv2d(prediction, window, padding=window_size // 2)
    mu2 = F.conv2d(target, window, padding=window_size // 2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(prediction ** 2, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(prediction * target, window, padding=window_size // 2) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Loss Functions Test")
    print("=" * 60)
    
    # Create dummy images
    target = torch.rand(2, 1, 64, 64)
    prediction = target + 0.1 * torch.randn_like(target)
    prediction = prediction.clamp(0, 1)
    
    # Test data fidelity
    data_loss = DataFidelityLoss('mse')
    print(f"MSE Loss: {data_loss(prediction, target):.6f}")
    
    # Test TV
    tv_loss = TotalVariationLoss('isotropic')
    print(f"TV Loss: {tv_loss(prediction):.6f}")
    
    # Test PSNR and SSIM
    print(f"PSNR: {compute_psnr(prediction, target):.2f} dB")
    print(f"SSIM: {compute_ssim(prediction, target):.4f}")
    
    # Test combined loss (without perceptual to avoid VGG download)
    combined = CombinedLoss(
        lambda_data=1.0, lambda_tv=0.01, 
        lambda_perceptual=0.0, use_perceptual=False
    )
    losses = combined(prediction, target)
    for name, value in losses.items():
        print(f"  {name}: {value.item():.6f}")
    
    print("\n✅ Loss functions test passed!")

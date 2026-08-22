# =============================================================================
# Lensless Phase-Mask Imaging — Package Initialization
# =============================================================================
#
# This package implements a physics-informed reconstruction pipeline
# for lensless cameras using phase-mask diffusers (DiffuserCam).
#
# Modules:
#   forward_model   — PSF simulation & differentiable forward model
#   classical_recon — Wiener deconvolution & ADMM reconstruction  
#   deep_recon      — Unrolled ADMM neural network
#   losses          — TV, perceptual, and combined loss functions
#   utils           — Visualization and I/O utilities
#   train           — Training pipeline
#
# =============================================================================

__version__ = "0.1.0"
__author__ = "Student Researcher"

from src.forward_model import PSFGenerator, DiffuserCamForwardModel, create_test_scene
from src.classical_recon import WienerDeconvolution, ADMMReconstructor
from src.deep_recon import UnrolledADMMNetwork, SimpleUNet
from src.losses import CombinedLoss, TotalVariationLoss, compute_psnr, compute_ssim

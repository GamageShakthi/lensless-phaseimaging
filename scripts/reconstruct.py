"""
reconstruct.py — Reconstruct Images from Lensless Measurements
==============================================================

Usage:
    # Wiener deconvolution
    python scripts/reconstruct.py --method wiener --reg 0.001

    # ADMM with TV
    python scripts/reconstruct.py --method admm --num_iters 50

    # Learned (unrolled ADMM network)
    python scripts/reconstruct.py --method learned --checkpoint pretrained/best_model.pth

    # Compare all methods
    python scripts/reconstruct.py --method all
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import time

from src.forward_model import PSFGenerator, DiffuserCamForwardModel, create_test_scene
from src.classical_recon import WienerDeconvolution, ADMMReconstructor
from src.deep_recon import UnrolledADMMNetwork
from src.losses import compute_psnr, compute_ssim
from src.utils import (save_image, plot_comparison, plot_convergence, 
                       tensor_to_numpy, numpy_to_tensor)


def reconstruct_wiener(measurement, model, reg=0.001):
    """Wiener deconvolution reconstruction."""
    print("\n--- Wiener Deconvolution ---")
    t0 = time.time()
    wiener = WienerDeconvolution(model.psf_fft, regularization=reg)
    recon = wiener.reconstruct(measurement)
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.3f}s")
    return recon, elapsed


def reconstruct_admm(measurement, model, num_iters=50, rho=1.0, 
                     lambda_tv=0.005, ground_truth=None):
    """ADMM reconstruction with TV prior."""
    print("\n--- ADMM Reconstruction ---")
    t0 = time.time()
    admm = ADMMReconstructor(
        model.psf_fft, model.psf_fft_conj, model.psf_fft_abs_sq,
        rho=rho, lambda_tv=lambda_tv, num_iters=num_iters, verbose=True
    )
    recon, history = admm.reconstruct(measurement, ground_truth=ground_truth)
    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.3f}s")
    return recon, history, elapsed


def reconstruct_learned(measurement, psf, checkpoint_path, num_stages=8):
    """Learned (unrolled ADMM) reconstruction."""
    print("\n--- Learned Reconstruction ---")
    
    device = measurement.device
    net = UnrolledADMMNetwork(
        psf=psf, num_stages=num_stages, in_channels=1, 
        num_features=64, num_blocks_per_stage=3
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    
    t0 = time.time()
    with torch.no_grad():
        recon = net(measurement)
    elapsed = time.time() - t0
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Model parameters: {net.count_parameters():,}")
    return recon, elapsed


def main():
    parser = argparse.ArgumentParser(description='Reconstruct from lensless measurement')
    parser.add_argument('--method', type=str, default='all',
                        choices=['wiener', 'admm', 'learned', 'all'],
                        help='Reconstruction method')
    parser.add_argument('--measurement', type=str, default=None,
                        help='Path to measurement .npy file')
    parser.add_argument('--psf_file', type=str, default=None,
                        help='Path to PSF .npy file')
    parser.add_argument('--size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--noise', type=float, default=0.01,
                        help='Noise level (for synthetic measurement)')
    parser.add_argument('--reg', type=float, default=0.001,
                        help='Wiener regularization')
    parser.add_argument('--num_iters', type=int, default=50,
                        help='ADMM iterations')
    parser.add_argument('--checkpoint', type=str, default='pretrained/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load or generate data
    if args.measurement and args.psf_file:
        meas_np = np.load(args.measurement)
        psf = np.load(args.psf_file)
        measurement = torch.from_numpy(meas_np).float().unsqueeze(0).unsqueeze(0).to(device)
        ground_truth = None
    else:
        print("No measurement provided — generating synthetic data...")
        psf_gen = PSFGenerator(size=args.size, seed=42)
        psf = psf_gen.random_phase_mask(feature_size=5.0)
        model = DiffuserCamForwardModel(psf, noise_std=args.noise).to(device)
        scene = create_test_scene(size=args.size, scene_type='natural')
        scene_tensor = torch.from_numpy(scene).float().unsqueeze(0).unsqueeze(0).to(device)
        measurement = model(scene_tensor)
        ground_truth = scene_tensor
    
    # Build forward model for reconstruction
    fwd_model = DiffuserCamForwardModel(psf, noise_std=0).to(device)
    
    results = {}
    images = [tensor_to_numpy(measurement)]
    titles = ['Measurement']
    
    if ground_truth is not None:
        images.append(tensor_to_numpy(ground_truth))
        titles.append('Ground Truth')
    
    # --- Wiener ---
    if args.method in ['wiener', 'all']:
        recon, t = reconstruct_wiener(measurement, fwd_model, reg=args.reg)
        results['wiener'] = {'recon': recon, 'time': t}
        images.append(tensor_to_numpy(recon))
        
        if ground_truth is not None:
            psnr = compute_psnr(recon, ground_truth)
            ssim = compute_ssim(recon, ground_truth)
            results['wiener']['psnr'] = psnr
            results['wiener']['ssim'] = ssim
            titles.append(f'Wiener\nPSNR={psnr:.1f}dB')
        else:
            titles.append('Wiener')
        
        save_image(tensor_to_numpy(recon), os.path.join(args.output, 'recon_wiener.png'))
    
    # --- ADMM ---
    if args.method in ['admm', 'all']:
        recon, history, t = reconstruct_admm(
            measurement, fwd_model, num_iters=args.num_iters,
            ground_truth=ground_truth
        )
        results['admm'] = {'recon': recon, 'time': t, 'history': history}
        images.append(tensor_to_numpy(recon))
        
        if ground_truth is not None:
            psnr = compute_psnr(recon, ground_truth)
            ssim = compute_ssim(recon, ground_truth)
            results['admm']['psnr'] = psnr
            results['admm']['ssim'] = ssim
            titles.append(f'ADMM\nPSNR={psnr:.1f}dB')
        else:
            titles.append('ADMM')
        
        save_image(tensor_to_numpy(recon), os.path.join(args.output, 'recon_admm.png'))
        
        if any('psnr' in h for h in history):
            plot_convergence(history, metric='psnr',
                           save_path=os.path.join(args.output, 'admm_convergence.png'))
    
    # --- Learned ---
    if args.method in ['learned', 'all']:
        if os.path.exists(args.checkpoint):
            recon, t = reconstruct_learned(measurement, psf, args.checkpoint)
            results['learned'] = {'recon': recon, 'time': t}
            images.append(tensor_to_numpy(recon))
            
            if ground_truth is not None:
                psnr = compute_psnr(recon, ground_truth)
                ssim = compute_ssim(recon, ground_truth)
                results['learned']['psnr'] = psnr
                results['learned']['ssim'] = ssim
                titles.append(f'Learned\nPSNR={psnr:.1f}dB')
            else:
                titles.append('Learned')
            
            save_image(tensor_to_numpy(recon), os.path.join(args.output, 'recon_learned.png'))
        else:
            print(f"\n⚠ Checkpoint not found: {args.checkpoint}")
            print("  Train the network first: python scripts/train_network.py")
    
    # Save comparison
    plot_comparison(
        images, titles,
        save_path=os.path.join(args.output, 'reconstruction_comparison.png'),
        suptitle='Reconstruction Results'
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for method, info in results.items():
        line = f"  {method:10s}: time={info['time']:.3f}s"
        if 'psnr' in info:
            line += f" | PSNR={info['psnr']:.2f}dB | SSIM={info['ssim']:.4f}"
        print(line)
    print("=" * 60)


if __name__ == '__main__':
    main()

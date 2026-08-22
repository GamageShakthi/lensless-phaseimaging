"""
simulate_measurement.py — Generate Synthetic DiffuserCam Measurements
=====================================================================

Usage:
    python scripts/simulate_measurement.py --scene_type natural --psf_type random_phase_mask
    python scripts/simulate_measurement.py --image path/to/image.png --noise 0.02
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch

from src.forward_model import PSFGenerator, DiffuserCamForwardModel, create_test_scene
from src.utils import save_image, plot_comparison, plot_psf_analysis, load_image


def main():
    parser = argparse.ArgumentParser(description='Simulate a DiffuserCam measurement')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (if None, generates synthetic scene)')
    parser.add_argument('--scene_type', type=str, default='natural',
                        choices=['natural', 'checkerboard', 'resolution_target', 'gradient'],
                        help='Type of synthetic scene to generate')
    parser.add_argument('--psf_type', type=str, default='random_phase_mask',
                        choices=['gaussian_mixture', 'random_phase_mask', 'caustic_pattern'],
                        help='Type of PSF to simulate')
    parser.add_argument('--size', type=int, default=256,
                        help='Image size (size x size)')
    parser.add_argument('--noise', type=float, default=0.01,
                        help='Gaussian noise standard deviation')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("  DiffuserCam Measurement Simulation")
    print("=" * 60)
    
    # Generate PSF
    print(f"\n[1] Generating PSF ({args.psf_type})...")
    psf_gen = PSFGenerator(size=args.size, seed=args.seed)
    
    if args.psf_type == 'gaussian_mixture':
        psf = psf_gen.gaussian_mixture()
    elif args.psf_type == 'random_phase_mask':
        psf = psf_gen.random_phase_mask()
    elif args.psf_type == 'caustic_pattern':
        psf = psf_gen.caustic_pattern()
    
    print(f"    PSF shape: {psf.shape}, sum: {psf.sum():.4f}")
    
    # Save PSF and analysis
    save_image(psf / psf.max(), os.path.join(args.output, 'psf.png'))
    plot_psf_analysis(psf, save_path=os.path.join(args.output, 'psf_analysis.png'))
    np.save(os.path.join(args.output, 'psf.npy'), psf)
    
    # Load or create scene
    print(f"\n[2] Creating scene...")
    if args.image:
        scene = load_image(args.image, size=args.size)
        print(f"    Loaded image from: {args.image}")
    else:
        scene = create_test_scene(size=args.size, scene_type=args.scene_type)
        print(f"    Generated synthetic scene ({args.scene_type})")
    
    save_image(scene, os.path.join(args.output, 'scene.png'))
    
    # Apply forward model
    print(f"\n[3] Applying forward model (noise_std={args.noise})...")
    model = DiffuserCamForwardModel(psf, noise_std=args.noise)
    scene_tensor = torch.from_numpy(scene).float().unsqueeze(0).unsqueeze(0)
    
    measurement = model(scene_tensor)
    meas_np = measurement.squeeze().cpu().numpy()
    
    save_image(meas_np, os.path.join(args.output, 'measurement.png'))
    np.save(os.path.join(args.output, 'measurement.npy'), meas_np)
    
    # Save comparison figure
    print(f"\n[4] Saving results to: {args.output}")
    plot_comparison(
        [scene, psf / psf.max(), meas_np],
        ['Original Scene', 'PSF', 'Measurement'],
        save_path=os.path.join(args.output, 'simulation_overview.png'),
        suptitle='DiffuserCam Forward Model Simulation'
    )
    
    print("\n✅ Done!")
    print(f"   Scene:       {os.path.join(args.output, 'scene.png')}")
    print(f"   PSF:         {os.path.join(args.output, 'psf.png')}")
    print(f"   Measurement: {os.path.join(args.output, 'measurement.png')}")
    print(f"   PSF data:    {os.path.join(args.output, 'psf.npy')}")


if __name__ == '__main__':
    main()

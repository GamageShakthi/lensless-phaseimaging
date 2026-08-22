"""
train_network.py — Train the Unrolled ADMM Reconstruction Network
=================================================================

Usage:
    python scripts/train_network.py --epochs 50 --batch_size 4
    python scripts/train_network.py --config configs/default.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import yaml

from src.forward_model import PSFGenerator
from src.deep_recon import UnrolledADMMNetwork
from src.train import SyntheticLenslessDataset, Trainer


def main():
    parser = argparse.ArgumentParser(description='Train the unrolled ADMM network')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory with training images')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {args.config}")
    
    # Extract settings with overrides
    img_cfg = config.get('image', {})
    psf_cfg = config.get('psf', {})
    fwd_cfg = config.get('forward_model', {})
    net_cfg = config.get('network', {})
    train_cfg = config.get('training', {})
    loss_cfg = config.get('loss', {})
    out_cfg = config.get('output', {})
    
    image_size = img_cfg.get('size', 128)
    noise_std = fwd_cfg.get('noise_std', 0.01)
    epochs = args.epochs or train_cfg.get('epochs', 50)
    batch_size = args.batch_size or train_cfg.get('batch_size', 4)
    lr = args.lr or train_cfg.get('learning_rate', 1e-4)
    
    print("=" * 60)
    print("  Training Configuration")
    print("=" * 60)
    print(f"  Image size: {image_size}")
    print(f"  Noise: {noise_std}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print("=" * 60)
    
    # Generate PSF
    psf_gen = PSFGenerator(size=image_size, seed=psf_cfg.get('seed', 42))
    psf_type = psf_cfg.get('type', 'random_phase_mask')
    
    if psf_type == 'gaussian_mixture':
        psf = psf_gen.gaussian_mixture(num_components=psf_cfg.get('num_components', 50))
    elif psf_type == 'random_phase_mask':
        psf = psf_gen.random_phase_mask(feature_size=psf_cfg.get('feature_size', 5.0))
    elif psf_type == 'caustic_pattern':
        psf = psf_gen.caustic_pattern()
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SyntheticLenslessDataset(
        image_dir=args.image_dir,
        psf=psf,
        num_samples=train_cfg.get('num_train_samples', 500),
        image_size=image_size,
        noise_std=noise_std
    )
    
    val_dataset = SyntheticLenslessDataset(
        psf=psf,
        num_samples=train_cfg.get('num_val_samples', 50),
        image_size=image_size,
        noise_std=noise_std
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create model
    print("\nBuilding model...")
    model = UnrolledADMMNetwork(
        psf=psf,
        num_stages=net_cfg.get('num_stages', 8),
        in_channels=img_cfg.get('channels', 1),
        num_features=net_cfg.get('num_features', 64),
        num_blocks_per_stage=net_cfg.get('num_blocks_per_stage', 3),
        share_weights=net_cfg.get('share_weights', False)
    )
    model.print_architecture()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=lr,
        batch_size=batch_size,
        lambda_tv=loss_cfg.get('lambda_tv', 0.01),
        lambda_perceptual=loss_cfg.get('lambda_perceptual', 0.0),
        use_perceptual=loss_cfg.get('lambda_perceptual', 0.0) > 0,
        checkpoint_dir=out_cfg.get('checkpoint_dir', 'pretrained')
    )
    
    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train!
    trainer.train(
        num_epochs=epochs,
        save_every=train_cfg.get('save_every', 10)
    )


if __name__ == '__main__':
    main()

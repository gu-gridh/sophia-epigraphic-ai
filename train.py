#!/usr/bin/env python3
"""
Training script for SOPHIA model.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from sophia.training import SophiaTrainer
from sophia.data import create_dataloaders


def main():
    parser = argparse.ArgumentParser(description='Train SOPHIA model')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--data_dir', default='./data', help='Data directory')
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override data paths if provided
    if args.data_dir:
        config['data']['train_csv'] = os.path.join(args.data_dir, 'train_dataset.csv')
        config['data']['val_csv'] = os.path.join(args.data_dir, 'val_dataset.csv')
        config['data']['images_dir'] = os.path.join(args.data_dir, 'images')
        config['data']['annotations_dir'] = os.path.join(args.data_dir, 'annotations')
    
    # Debug mode
    if args.debug:
        config['training']['num_epochs'] = 2
        config['training']['batch_size'] = 2
        config['use_wandb'] = False
        print("DEBUG MODE: Reduced epochs and batch size")
    
    print("Configuration:")
    print(json.dumps(config, indent=2))
    
    # Check data availability
    train_csv = config['data']['train_csv']
    val_csv = config['data']['val_csv']
    images_dir = config['data']['images_dir']
    annotations_dir = config['data']['annotations_dir']
    
    if not all(os.path.exists(p) for p in [train_csv, val_csv, images_dir, annotations_dir]):
        print("ERROR: Some data files are missing!")
        print(f"Train CSV: {train_csv} - {'✓' if os.path.exists(train_csv) else '✗'}")
        print(f"Val CSV: {val_csv} - {'✓' if os.path.exists(val_csv) else '✗'}")
        print(f"Images dir: {images_dir} - {'✓' if os.path.exists(images_dir) else '✗'}")
        print(f"Annotations dir: {annotations_dir} - {'✓' if os.path.exists(annotations_dir) else '✗'}")
        return 1
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Validation loader: {len(val_loader)} batches")
    
    # Create trainer
    print("Initializing trainer...")
    trainer = SophiaTrainer(config)
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs']
    )
    
    print("Training completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Multi-Channel Graffiti Recognition Training
==========================================

Comprehensive 12-channel vision processing with advanced transformer decoder.
Quick training script that uses the unified train_sophia.py with multi-channel model settings.
"""

import subprocess
import sys
import os

def main():
    print(" Starting Multi-Channel Model Training")
    print(" Configuration: 15 epochs, batch size 6, comprehensive architecture")
    print(" Features: 12-channel vision + 6-layer transformer + language conditioning")
    print("-" * 60)
    
    # Run the unified training script with multi-channel model settings
    cmd = [
        sys.executable, 'train_sophia.py',
        '--model_type', 'multichannel',
        '--epochs', '15',
        '--batch_size', '6',
        '--lr', '1e-4'
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f" Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == '__main__':
    main()

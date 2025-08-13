#!/bin/bash
# Manual SOPHIA Environment Setup (Much Faster)

echo "SOPHIA - Manual Environment Setup"
echo "================================="

# Create a basic conda environment
echo "Creating basic conda environment..."
conda create -n sophia-ai python=3.10 -y

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate sophia-ai

# Install PyTorch with CUDA support via conda (fastest for PyTorch)
echo "Installing PyTorch with CUDA..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install everything else via pip (much faster)
echo "Installing core packages via pip..."
pip install transformers pandas numpy pillow opencv-python matplotlib jupyter notebook

echo "Installing additional ML packages..."
pip install scipy scikit-learn wandb tensorboard tqdm requests

echo "Installing development tools..."
pip install pytest black flake8

echo "Installing optional packages..."
pip install shapely geopandas albumentations

echo "Creating project directories..."
mkdir -p data/{train,val,test,images,annotations,processed}
mkdir -p checkpoints logs outputs

echo "Testing installation..."
python -c "
import torch
import transformers
import pandas as pd
import numpy as np
from PIL import Image
import cv2

print('SUCCESS: PyTorch:', torch.__version__)
print('SUCCESS: Transformers:', transformers.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('Using CPU')
print('SUCCESS: All packages imported!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "SUCCESS: SOPHIA environment is ready!"
    echo "To activate: conda activate sophia-ai"
else
    echo "ERROR: Installation failed"
fi

#!/bin/bash
# SOPHIA Environment Setup Script

echo "SOPHIA - Epigraphic AI Setup"
echo "============================"

# Function to check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo "ERROR: Conda is not installed. Please install Anaconda or Miniconda first."
        echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    echo "SUCCESS: Conda found: $(conda --version)"
}

# Function to create conda environment
create_environment() {
    echo ""
    echo "Creating conda environment 'sophia-ai'..."
    
    if conda env list | grep -q "sophia-ai"; then
        echo "WARNING: Environment 'sophia-ai' already exists."
        read -p "Do you want to remove and recreate it? (y/N): " recreate
        if [[ $recreate =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            conda env remove -n sophia-ai -y
        else
            echo "INFO: Using existing environment."
            return 0
        fi
    fi
    
    echo "Creating new environment from environment.yml..."
    conda env create -f environment.yml
    
    if [ $? -eq 0 ]; then
        echo "SUCCESS: Environment created successfully!"
    else
        echo "ERROR: Failed to create environment. Trying alternative method..."
        echo "Creating basic environment and installing packages..."
        
        conda create -n sophia-ai python=3.10 -y
        conda activate sophia-ai
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        pip install -r requirements.txt
    fi
}

# Function to setup development tools
setup_dev_tools() {
    echo ""
    echo "Setting up development tools..."
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate sophia-ai
    
    # Install pre-commit hooks (optional)
    if command -v pre-commit &> /dev/null; then
        echo "Setting up pre-commit hooks..."
        pre-commit install
    fi
    
    # Create directories
    echo "Creating project directories..."
    mkdir -p data/{train,val,test,images,annotations,processed}
    mkdir -p checkpoints
    mkdir -p logs
    mkdir -p outputs
    
    echo "SUCCESS: Development environment setup complete!"
}

# Function to test installation
test_installation() {
    echo ""
    echo "Testing installation..."
    
    eval "$(conda shell.bash hook)"
    conda activate sophia-ai
    
    python -c "
import torch
import torchvision
import transformers
import pandas as pd
import numpy as np
from PIL import Image
import cv2

print('SUCCESS: PyTorch:', torch.__version__)
print('SUCCESS: Torchvision:', torchvision.__version__)
print('SUCCESS: Transformers:', transformers.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU Memory:', f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('Using CPU')

print('SUCCESS: All core packages imported successfully!')
"
    
    if [ $? -eq 0 ]; then
        echo "SUCCESS: Installation test passed!"
    else
        echo "ERROR: Installation test failed. Please check the error messages above."
        return 1
    fi
}

# Function to display usage instructions
show_usage() {
    echo ""
    echo "Getting Started with SOPHIA"
    echo "==========================="
    echo ""
    echo "1. Activate the environment:"
    echo "   conda activate sophia-ai"
    echo ""
    echo "2. Prepare your data:"
    echo "   python scripts/prepare_dataset.py \\"
    echo "     --csv_path /path/to/combined_dataset.csv \\"
    echo "     --annotations_dir /path/to/annotations/ \\"
    echo "     --images_dir /path/to/images/ \\"
    echo "     --output_dir ./data"
    echo ""
    echo "3. Train the model:"
    echo "   python train.py --config configs/sophia_base.json"
    echo ""
    echo "4. Run inference:"
    echo "   python predict.py \\"
    echo "     --model_path ./checkpoints/best_model.pt \\"
    echo "     --config configs/sophia_base.json \\"
    echo "     --image inscription.jpg"
    echo ""
    echo "For more details, see README.md"
    echo ""
}

# Main execution
main() {
    check_conda
    create_environment
    setup_dev_tools
    test_installation
    
    if [ $? -eq 0 ]; then
        show_usage
        echo "SOPHIA environment is ready!"
        echo "Run 'conda activate sophia-ai' to get started."
    else
        echo "WARNING: Setup completed with some issues. Please review the output above."
    fi
}

# Run main function
main "$@"

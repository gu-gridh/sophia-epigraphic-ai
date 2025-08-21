# SOPHIA Training System Overview

## Architecture

The SOPHIA training system now has a clean, modular structure with two main model architectures:

### 1. **Multi-Channel Model** (`models_multichannel.py`)
- **12-channel vision encoder** (4 image types × 3 RGB channels)
- **6-layer transformer decoder**
- **Language + writing system conditioning**
- **~286M parameters**
- **Best for:** Comprehensive feature extraction from all image modalities

### 2. **Enhanced Model** (`models_enhanced.py`)
- **Channel compression + simplified CNN**
- **4-layer transformer decoder**
- **Simplified language conditioning (language only)**
- **~133M parameters**
- **Best for:** Faster training with good performance

## 🚀 Training Scripts

### Unified Training Script
```bash
# Main training script with full control
python train_sophia.py --model_type [enhanced|multichannel] --epochs N --batch_size N

# Show available models and usage
python train_sophia.py --info

# Examples
python train_sophia.py --model_type enhanced --epochs 12 --batch_size 8
python train_sophia.py --model_type multichannel --epochs 15 --batch_size 6
```

### Quick Training Shortcuts
```bash
# Enhanced model with optimized settings
python train_enhanced_graffiti.py

# Multi-channel model with optimized settings  
python train_multichannel_graffiti.py
```

## Model Comparison

| Feature | Enhanced Model | Multi-Channel Model |
|---------|----------------|---------------------|
| Parameters | 133M | 286M |
| Training Speed |  Fast |  Slower |
| Memory Usage |  Low |  High |
| Vision Processing | Channel compression | Full 12-channel |
| Language Conditioning | Simplified | Full (Lang + Writing) |
| Transformer Layers | 4 | 6 |
| Best Use Case | Quick experiments | Maximum performance |

##  Language Conditioning

### Enhanced Model
- Simplified prefix: `[LANGUAGE] transcription`
- Example: `[GREEK] ΑΛΕΞΑΝΔΡΟΣ`

### Multi-Channel Model  
- Full prefix: `[LANGUAGE][WRITING_SYSTEM] transcription`
- Example: `[GREEK][GREEK_MAJUSCULE] ΑΛΕΞΑΝΔΡΟΣ`

##  File Structure

```
sophia-epigraphic-ai/
├── train_sophia.py              # Unified training script
├── train_enhanced_graffiti.py   # Enhanced model shortcut
├── train_multichannel_graffiti.py  # Multi-channel shortcut
├── models_enhanced.py           # Enhanced model architecture
├── models_multichannel.py       # Multi-channel model architecture
├── evaluate_sophia.py           # Model evaluation
└── models/                      # Saved model checkpoints
    ├── best_enhanced_model.pth
    └── best_multichannel_model.pth
```

## Usage Examples

### Quick Start - Enhanced Model
```bash
python train_enhanced_graffiti.py
```

### Custom Training
```bash
# Train enhanced model for 20 epochs
python train_sophia.py --model_type enhanced --epochs 20 --batch_size 16

# Train multi-channel model with small batch size
python train_sophia.py --model_type multichannel --epochs 10 --batch_size 4
```

### Model Information
```bash
python train_sophia.py --info
```
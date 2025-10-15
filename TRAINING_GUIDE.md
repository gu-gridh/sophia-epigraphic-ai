# Training Guide - Saint Sophia Graffiti Recognition# SOPHIA Training System Overview



## Quick Start## Architecture



### 1. Basic Training CommandsThe SOPHIA training system now has a clean, modular structure with two main model architectures:



```bash### 1. **Multi-Channel Model** (`models_multichannel.py`)

# Multi-Channel model (default: both RTI + Korniienko)- **12-channel vision encoder** (4 image types × 3 RGB channels)

python train.py --model multichannel --epochs 15 --batch_size 6- **6-layer transformer decoder**

- **Language + writing system conditioning**

# Enhanced model- **~286M parameters**

python train.py --model enhanced --epochs 12 --batch_size 8- **Best for:** Comprehensive feature extraction from all image modalities



# Transformer model### 2. **Enhanced Model** (`models_enhanced.py`)

python train.py --model transformer --epochs 20 --batch_size 4- **Channel compression + simplified CNN**

```- **4-layer transformer decoder**

- **Simplified language conditioning (language only)**

### 2. Phase-Based Training- **~133M parameters**

- **Best for:** Faster training with good performance

**Phase 1: Korniienko-Only** (939 samples available NOW)

```bash## 🚀 Training Scripts

python train.py --model enhanced --use_korniienko --no_rti --epochs 10 --batch_size 8

```### Unified Training Script

```bash

**Phase 2: RTI-Only** (after cropping completes)# Main training script with full control

```bashpython train_sophia.py --model_type [enhanced|multichannel] --epochs N --batch_size N

python train.py --model enhanced --use_rti --no_korniienko --epochs 10 --batch_size 6

```# Show available models and usage

python train_sophia.py --info

**Phase 3: Full Multi-Modal** (best performance expected)

```bash# Examples

python train.py --model enhanced --use_rti --use_korniienko --epochs 15 --batch_size 6python train_sophia.py --model_type enhanced --epochs 12 --batch_size 8

```python train_sophia.py --model_type multichannel --epochs 15 --batch_size 6

```

## Model Comparison

### Quick Training Shortcuts

### Data Requirements```bash

# Enhanced model with optimized settings

| Model | RTI Support | Korniienko Support | Memory | Recommended Batch Size |python train_enhanced_graffiti.py

|-------|-------------|-------------------|--------|----------------------|

| **multichannel** | ✓ (12 channels) | ✓ (photo + drawing) | High | 4-6 |# Multi-channel model with optimized settings  

| **enhanced** | ✓ (12 channels) | ✓ (photo + drawing) | Medium | 6-8 |python train_multichannel_graffiti.py

| **transformer** | ✓ (12 channels) | ✓ (photo + drawing) | Medium | 4-6 |```



### Model Characteristics## Model Comparison



**Multi-Channel** (70.3M params)| Feature | Enhanced Model | Multi-Channel Model |

- **Strengths**: Comprehensive multi-channel RTI processing|---------|----------------|---------------------|

- **Best for**: Maximum feature extraction from RTI| Parameters | 133M | 286M |

- **Training time**: Slowest| Training Speed |  Fast |  Slower |

- **Command**: `--model multichannel --batch_size 6`| Memory Usage |  Low |  High |

| Vision Processing | Channel compression | Full 12-channel |

**Enhanced** (58.0M params)| Language Conditioning | Simplified | Full (Lang + Writing) |

- **Strengths**: Deep ResNet + attention mechanisms| Transformer Layers | 4 | 6 |

- **Best for**: Balanced performance/speed| Best Use Case | Quick experiments | Maximum performance |

- **Training time**: Medium

- **Command**: `--model enhanced --batch_size 8`##  Language Conditioning



**Transformer** (50.8M params)### Enhanced Model

- **Strengths**: Multi-task learning, attention fusion- Simplified prefix: `[LANGUAGE] transcription`

- **Best for**: Best expected accuracy- Example: `[GREEK] ΑΛΕΞΑΝΔΡΟΣ`

- **Training time**: Fast (smallest model)

- **Command**: `--model transformer --batch_size 4 --lr 5e-5`### Multi-Channel Model  

- Full prefix: `[LANGUAGE][WRITING_SYSTEM] transcription`

## Advanced Options- Example: `[GREEK][GREEK_MAJUSCULE] ΑΛΕΞΑΝΔΡΟΣ`



### Custom Data Paths##  File Structure

```bash

python train.py --model enhanced \```

    --data_dir /path/to/data \sophia-epigraphic-ai/

    --train_csv data/my_train.csv \├── train_sophia.py              # Unified training script

    --val_csv data/my_val.csv \├── train_enhanced_graffiti.py   # Enhanced model shortcut

    --epochs 15├── train_multichannel_graffiti.py  # Multi-channel shortcut

```├── models_enhanced.py           # Enhanced model architecture

├── models_multichannel.py       # Multi-channel model architecture

### Learning Rate Tuning├── evaluate_sophia.py           # Model evaluation

```bash└── models/                      # Saved model checkpoints

# Lower LR for transformer (more stable)    ├── best_enhanced_model.pth

python train.py --model transformer --lr 5e-5 --epochs 20    └── best_multichannel_model.pth

```

# Higher LR for CNN models

python train.py --model enhanced --lr 2e-4 --epochs 12## Usage Examples

```

### Quick Start - Enhanced Model

### Resume Training```bash

```bashpython train_enhanced_graffiti.py

python train.py --model enhanced \```

    --resume checkpoints/enhanced/20251015_120000/checkpoint_epoch_10.pt \

    --epochs 20### Custom Training

``````bash

# Train enhanced model for 20 epochs

## Expected Performancepython train_sophia.py --model_type enhanced --epochs 20 --batch_size 16



Based on similar OCR/HTR tasks:# Train multi-channel model with small batch size

python train_sophia.py --model_type multichannel --epochs 10 --batch_size 4

| Phase | Modalities | Expected CER | Expected WER |```

|-------|-----------|--------------|--------------|

| Phase 1 | Korniienko only | 20-30% | 40-50% |### Model Information

| Phase 2 | RTI only | 15-25% | 35-45% |```bash

| Phase 3 | RTI + Korniienko | **10-20%** | **25-35%** |python train_sophia.py --info

```
**Target**: >50% word-level accuracy (WER < 50%)

## Next Steps

1. ✅ Unified training script created (`train.py`)
2. 📋 **Start Phase 1 training** (Korniienko-only)
3. ⏳ Wait for RTI cropping to complete
4. 📋 Create evaluation script (`evaluate.py`)
5. 📋 Run full comparison across all models and phases

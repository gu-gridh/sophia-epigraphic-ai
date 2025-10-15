# Saint Sophia Graffiti Recognition

Multimodal deep learning framework for automatic transcription of ancient inscriptions from Saint Sophia Cathedral in Kyiv, Ukraine. Combines RTI imaging and Korniienko photo/drawing documentation with transformer-based models.

## Features

- **Multi-modal architecture**: RTI images (12 channels) + Korniienko photos/drawings
- **3 model architectures**: Multi-Channel CNN, Enhanced CNN, Transformer (50-70M params)
- **Multilingual support**: Greek, Latin, Cyrillic scripts
- **Unified training pipeline**: Single script for all models and modalities
- **Comprehensive evaluation**: CER, WER, per-language metrics, error analysis


## Quick Start

### Installation

```bash
git clone https://github.com/gu-gridh/sophia-epigraphic-ai.git
cd sophia-epigraphic-ai
pip install -r requirements.txt
```

### Training

**Phase 1: Korniienko-only** (Available NOW - 939 samples)
```bash
python train.py --model enhanced --use_korniienko --no_rti --epochs 5 --batch_size 8
```

**Phase 2: RTI-only** (After cropping completes)
```bash
python train.py --model enhanced --use_rti --no_korniienko --epochs 10 --batch_size 8
```

**Phase 3: Full multi-modal** (Best performance)
```bash
python train.py --model transformer --use_rti --use_korniienko --epochs 20 --batch_size 8
```

### Evaluation

```bash
python evaluate.py \
    --model enhanced \
    --checkpoint checkpoints/enhanced/phase1/best_model.pt \
    --use_korniienko --no_rti \
    --output_dir evaluation_results/enhanced_phase1
```

**Outputs**: metrics.json, predictions.csv, error_analysis.csv, EVALUATION_REPORT.md

## Model Architectures

| Model | Parameters | Input Modalities |
|-------|------------|------------------|
| Multi-Channel CNN | 70.3M | RTI + Korniienko |
| Enhanced CNN | 58.0M | RTI + Korniienko |
| Transformer | 50.8M | RTI + Korniienko |

All models support flexible modality selection via command-line flags.

## Documentation

- **TRAINING_GUIDE.md** - Complete training documentation
- **EVALUATION_GUIDE.md** - Evaluation metrics and analysis
- **SESSION_SUMMARY.md** - Technical implementation details

## Project Structure

```
sophia-epigraphic-ai/
├── train.py                    # Unified training script
├── evaluate.py                 # Comprehensive evaluation
├── data/                       # Train/val/test CSV files
├── models/                     # Model architectures
│   ├── models_multichannel.py  # Multi-Channel CNN
│   ├── models_enhanced.py      # Enhanced CNN
│   └── models_transformer.py   # Transformer
├── checkpoints/                # Saved model checkpoints
├── evaluation_results/         # Evaluation outputs
└── cropped_images_hq/          # RTI images (4 types)
```


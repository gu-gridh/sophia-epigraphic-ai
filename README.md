# Saint Sophia Graffiti Recognition

Multimodal deep learning framework for automatic transcription of ancient inscriptions from Saint Sophia Cathedral in Kyiv, Ukraine. Combines RTI imaging and Korniienko photo/drawing documentation with transformer-based models.

## Dataset

| Metric | Count |
|--------|-------|
| Total inscriptions with valid transcription | 1,854 |
| Training-ready samples (with images) | **1,720** |
| Korniienko photos | 1,602 |
| Korniienko drawings | 1,607 |
| IIIF panel crops | 538 |

**Language distribution:**
- Church Slavonic: 872 (55%)
- Ukrainian: 370 (23%)
- Polish: 155 (10%)
- Ancient Greek: 81 (5%)
- Armenian: 26 (2%)
- Latin, Greek, Mixed, Russian: ~5%

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

### Data Collection

**Step 1: Prepare dataset from API and local images**
```bash
# Fetch fresh data from API (requires network access)
python scripts/prepare_dataset.py --fetch --download-iiif

# Or use cached data with existing images
python scripts/prepare_dataset.py --download-iiif
```

**Step 2: Download Korniienko images (if not present)**
```bash
# Korniienko images should be in data/korniienkoimages/
# Expected naming: Korniienko_v{vol}_{page}_{idx}_photo_{id}.png
#                  Korniienko_v{vol}_{page}_{idx}_drawing_{id}.png
ls data/korniienkoimages/*.png | wc -l  # Should show ~12,000 files
```

**Step 3: Download IIIF panel crops for inscriptions without dedicated images**
```bash
python scripts/download_iiif_crops.py
```

**Step 4: Verify dataset**
```bash
python -c "
from train import SophiaMultiModalDataset, CharacterTokenizer
tokenizer = CharacterTokenizer()
dataset = SophiaMultiModalDataset(
    csv_file='data/complete_dataset.csv',
    data_dir='data',
    tokenizer=tokenizer,
    use_rti=False,
    use_korniienko=True,
    model_type='enhanced',
    split='train'
)
print(f'Training samples: {len(dataset)}')
"
```

### Training

**Korniienko-only** (Recommended - 1,720 samples)
```bash
python train.py --model enhanced --use_korniienko --no_rti --epochs 30 --batch_size 8
```

**Cross-validation** (For robust evaluation)
```bash
python cross_validate.py --model enhanced --folds 5 --epochs 30 --use_korniienko
```

**Full multi-modal** (With RTI images)
```bash
python train.py --model transformer --use_rti --use_korniienko --epochs 30 --batch_size 8
```

### Evaluation

```bash
python evaluate.py \
    --model enhanced \
    --checkpoint checkpoints/enhanced/best_model.pt \
    --use_korniienko --no_rti \
    --output_dir evaluation_results/enhanced
```

**Outputs**: metrics.json, predictions.csv, error_analysis.csv, EVALUATION_REPORT.md

## Results

**5-Fold Cross-Validation (Enhanced CNN, Korniienko-only):**
| Metric | Mean ± Std |
|--------|------------|
| CER | 7.05% ± 0.70% |
| Sequence Accuracy | 52.19% ± 1.47% |

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
├── cross_validate.py           # K-fold cross-validation
├── scripts/
│   ├── prepare_dataset.py      # Data collection pipeline
│   └── download_iiif_crops.py  # IIIF image downloader
├── data/
│   ├── complete_dataset.csv    # Main dataset file
│   ├── korniienkoimages/       # Korniienko photos/drawings
│   └── iiif_crops/             # IIIF panel crops
├── models/                     # Model architectures
│   ├── models_multichannel.py  # Multi-Channel CNN
│   ├── models_enhanced.py      # Enhanced CNN
│   └── models_transformer.py   # Transformer
├── checkpoints/                # Saved model checkpoints
└── evaluation_results/         # Evaluation outputs
```

## API Reference

**Saint Sophia Inscriptions API:**
- Base URL: `https://saintsophia.dh.gu.se/api/inscriptions/inscription/`
- IIIF Image Server: `https://img.dh.gu.se/saintsophia/static/inscriptions/iiif/`

## Citation

```bibtex
@InProceedings{10.1007/978-3-032-36042-7_6,
author="Karimi, Aram
and Westin, Jonathan
and Almevik, Gunnar",
title="Multi-channel Deep Learning for Medieval Inscription Recognition: A Study of Saint Sophia Cathedral Graffiti",
booktitle="Document Analysis and Recognition -- ICDAR 2026",
year="2027",
publisher="Springer Nature Switzerland",
pages="87--104",
isbn="978-3-032-36042-7"
}
```


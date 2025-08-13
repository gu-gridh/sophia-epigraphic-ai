# SOPHIA - Epigraphic AI

Multimodal deep learning framework for reading ancient inscriptions from Saint Sophia Cathedral in Kyiv, Ukraine. This tool combines computer vision with spatial annotation data to automatically transcribe Byzantine graffiti and historical texts, advancing digital humanities research through AI-powered archaeological text recognition.

## Overview

SOPHIA is inspired by state-of-the-art approaches like [ITHACA](https://github.com/google-deepmind/ithaca), [Predicting the Past](https://github.com/google-deepmind/predictingthepast), and [Ancient Text Restoration](https://github.com/sommerschield/ancient-text-restoration), but specifically designed for the unique challenges of graffiti and spatial annotation integration.

### Key Innovation

Unlike traditional OCR systems, SOPHIA leverages **spatial annotation metadata** alongside visual features to achieve better recognition accuracy. Each training example combines:

- High-resolution inscription images
- Geometric annotation data (bounding boxes, polygons, coordinates)
- Historical context (dating, language, surface information)
- Transcription ground truth

## Architecture

```text
Input Modalities:
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Images    │    │   Spatial    │    │    Text     │
│             │    │ Annotations  │    │  Context    │
└─────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Vision    │    │   Spatial    │    │    Text     │
│  Encoder    │    │   Encoder    │    │  Encoder    │
└─────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │
       └─────────▼─────────┘───────────────────┘
                 │
                 ▼
         ┌─────────────┐
         │ Multimodal  │
         │   Fusion    │
         └─────────────┘
                 │
                 ▼
         ┌─────────────┐
         │ Transformer │
         │   Decoder   │
         └─────────────┘
                 │
                 ▼
         ┌─────────────┐
         │ Inscription │
         │Transcription│
         └─────────────┘
```

## Quick Start

### 1. Installation

```bash
git clone https://github.com/gu-gridh/sophia-epigraphic-ai.git
cd sophia-epigraphic-ai
pip install -r requirements.txt
```

### 2. Data Preparation

First, collect your data using the Saint Sophia data tools:

```bash
# In your Saint Sophia backend directory
cd saintsophia-backend/data_tools
python export_inscriptions.py
python download_annotations.py inscriptions_*.csv
python create_dataset.py inscriptions_*.csv
```

Then prepare the data for SOPHIA training:

```bash
# In SOPHIA directory
python scripts/prepare_dataset.py \
  --csv_path /path/to/combined_dataset_*.csv \
  --annotations_dir /path/to/annotations/ \
  --images_dir /path/to/images/ \
  --output_dir ./data \
  --validate
```

### 3. Training

```bash
python train.py --config configs/sophia_base.json --data_dir ./data
```

### 4. Inference

Single image:
```bash
python predict.py \
  --model_path ./checkpoints/best_model.pt \
  --config configs/sophia_base.json \
  --image /path/to/inscription.jpg \
  --annotation /path/to/annotation.json
```

Batch processing:
```bash
python predict.py \
  --model_path ./checkpoints/best_model.pt \
  --config configs/sophia_base.json \
  --images_dir ./data/images \
  --output results.json
```

## Data Integration

SOPHIA seamlessly integrates with the Saint Sophia data collection pipeline:

```bash
# Complete workflow from database to trained model

# 1. Export from database (Saint Sophia backend)
cd saintsophia-backend/data_tools
python collect_data.py  # All-in-one data collection

# 2. Prepare for SOPHIA (this repository)
cd sophia-epigraphic-ai
python scripts/prepare_dataset.py \
  --csv_path ../saintsophia-backend/combined_dataset_*.csv \
  --annotations_dir ../saintsophia-backend/annotations/ \
  --images_dir /path/to/your/images/ \
  --output_dir ./data

# 3. Train SOPHIA model
python train.py --config configs/sophia_base.json

# 4. Use for inference
python predict.py --model_path ./checkpoints/best_model.pt --config configs/sophia_base.json --image new_inscription.jpg
```

## Model Features

### Vision Component
- **Multi-scale feature extraction** using pre-trained CNN backbones
- **Spatial pyramid pooling** for handling variable inscription sizes
- **Attention mechanisms** for focusing on text regions

### Spatial Component
- **Geometric encoding** of annotation coordinates
- **Spatial relationship modeling** between multiple annotations
- **Layout-aware feature fusion**

### Text Component
- **Multilingual support** (Greek, Latin, Cyrillic scripts)
- **Historical language modeling** with specialized embeddings
- **Character-level processing** for damaged text handling

### Training Innovations
- **Annotation-guided learning**: Each annotation becomes a training example
- **Multi-task objectives**: Transcription + dating + restoration confidence
- **Data augmentation** for limited historical datasets

## Dataset Structure

After preparation, your dataset will have this structure:

```
data/
├── train_dataset.csv          # Training split
├── val_dataset.csv            # Validation split  
├── test_dataset.csv           # Test split
├── images/                    # All inscription images
│   ├── 123.jpg
│   └── 456.jpg
├── annotations/               # All annotation JSON files
│   ├── annotation_123.json
│   └── annotation_456.json
└── dataset_stats.json         # Dataset statistics
```

Each CSV row contains:
- **All inscription fields** (23 fields: transcription, metadata, translations, etc.)
- **Annotation metadata** (index, coordinates, geometry type)
- **File paths** for images and annotations

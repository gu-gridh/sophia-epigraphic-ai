# Saint Sophia Graffiti Recognition - Complete Documentation

This directory contains all the scripts for preparing data and training the graffiti recognition model.

---

## Data Preparation Pipeline

Run these scripts in order to prepare your training data:

### 1. **`prepare_crop_dataset.py`** - Create Train/Val/Test Splits
- **Input**: `inscriptions_graffiti_*.csv`
- **Output**: `train_dataset.csv`, `val_dataset.csv`, `test_dataset.csv`


### 2. **`clean_text.py`** - Clean Text for Training
- **Input**: `inscriptions_graffiti_*.csv`
- **Output**: `inscriptions_graffiti_cleaned.csv`
- **Removes**:
  - HTML tags and entities (preserves `</p>` as line breaks)
  - Editorial symbols: `()`, `[]`, `{}`, `<>` and their content
  - Abbreviation markers: `҃`, `͠`, `ⷠ`, combining diacriticals
  - Punctuation: `,`, `?`, `+`, quotes (but NOT single dashes)
  - Pipe notation `|` → spaces
  - Gap markers (3+ dashes: `---`)
- **Preserves**:
  - Line breaks (multi-line inscriptions)
  - Single/double dashes (e.g., "ни-")
- **Result**: 1,602 clean transcriptions (76.2%), 94.7% size reduction

### 3. **`crop_images.py`** - Extract High-Quality Image Crops
- **Input**: Split datasets + original panel images
- **Output**: Cropped PNG images (4 types per inscription)
  - `original` - Standard RGB photograph
  - `blended` - RTI blended texture map
  - `normal` - RTI normal map (surface normals)
  - `texture` - RTI texture map
- **Features**:
  - Lossless PNG format (compress_level=6)
  - Original crop dimensions preserved (adaptive sizing)
  - LANCZOS resampling for quality
  - Graceful error handling for corrupted/missing files
  - Progress tracking with statistics
- **Run Time**: ~6 hours for full dataset
- **Naming**: `{inscription_id}_{image_type}.png`

### 4. **`create_datasets.py`** - Build Comprehensive Training Datasets
- **Input**: Split datasets + cleaned inscriptions + Korniienko catalog + cropped images
- **Output**: Comprehensive CSV files with 50+ columns
  - `train_comprehensive.csv` 
  - `val_comprehensive.csv`
  - `test_comprehensive.csv`
  - `complete_dataset.csv`
- **Includes**:
  - RTI cropped image paths (all 4 types) + availability flags
  - Cleaned text (transcription, translation, romanisation)
  - Korniienko local image paths (photo + drawing)
  - Metadata (dates, languages, locations, bounding boxes)
  - Binary flags (has_transcription, has_korniienko, has_all_images, etc.)

## Quick Start

### Step 1: Prepare Dataset Splits

```bash
python prepare_crop_dataset.py
```

### Step 2: Clean Text Data

```bash
python clean_text.py
```

### Step 3: Crop Images (long-running process)

```bash
# Run in background
nohup python crop_images.py > crop_log.txt 2>&1 &

# Monitor progress
tail -f crop_log.txt

# Or check cropped image count
ls -1 ../data/cropped_images_hq/train/original/ | wc -l
```

### Step 4: Create Comprehensive Datasets

```bash
python create_datasets.py
```

## Output Files

After running all scripts, you'll have:

### In `../data/`:
- **Split datasets (basic)**:
  - `train_dataset.csv` (1,470 inscriptions)
  - `val_dataset.csv` (315 inscriptions)
  - `test_dataset.csv` (316 inscriptions)

- **Comprehensive datasets (with all metadata)**:
  - `train_comprehensive.csv`
  - `val_comprehensive.csv`
  - `test_comprehensive.csv`
  - `complete_dataset.csv` (combined)

- **Cropped images**:
  - `cropped_images_hq/train/{original,blended,normal,texture}/`
  - `cropped_images_hq/val/{original,blended,normal,texture}/`
  - `cropped_images_hq/test/{original,blended,normal,texture}/`

---

## Comprehensive Dataset Schema

The `*_comprehensive.csv` files contain 50+ columns organized as follows:

### Identification
- `id` - Unique inscription ID
- `title` - Inscription title
- `split` - train/val/test

### RTI Images (Cropped)
- `original_image` - Path to cropped original image
- `blended_image` - Path to cropped blended map
- `normal_image` - Path to cropped normal map
- `texture_image` - Path to cropped texture map
- `num_available_images` - Count (0-4)
- `has_all_images` - Binary flag

### Korniienko References (Local Paths)
- `korniienko_photo` - Path: `korniienkoimages/Korniienko_v{vol}_{plate}_{num}_photo_{title}.png`
- `korniienko_drawing` - Path: `korniienkoimages/Korniienko_v{vol}_{plate}_{num}_drawing_{title}.png`
- `korniienko_count` - Number of Korniienko images
- `korniienko_bibliography` - Volume reference
- `korniienko_plate` - Plate number
- `korniienko_year` - Publication year
- `has_korniienko` - Binary flag

### Location & Bounding Box
- `panel_id`, `panel_title`, `panel_room` - Physical location
- `bbox_x`, `bbox_y` - Top-left coordinates (0-1 range)
- `bbox_width`, `bbox_height` - Dimensions (0-1 range)
- `position_on_surface` - Original IIIF format
- `elevation` - Height from floor (mm)

### Text Content (Cleaned - No HTML)
- `transcription_clean` - Cyrillic/Greek transcription (line breaks preserved)
- `interpretative_edition_clean` - Scholarly interpretation
- `romanisation_clean` - Latin alphabet transliteration
- `translation_eng_clean` - English translation
- `translation_ukr_clean` - Ukrainian translation
- `comments_eng_clean`, `comments_ukr_clean` - Commentary

### Metadata
- `language_name` - Language (Church Slavonic, Ukrainian, Polish, Ancient Greek, etc.)
- `writing_system_name` - Script (Cyrillic, Latin, Greek)
- `type_of_inscription` - Type (Textual graffiti, etc.)
- `min_year`, `max_year` - Date range (1010-1715 CE)
- `date_range` - Years between min/max
- `date_midpoint` - Average date
- `height`, `width` - Physical dimensions (mm)

### Binary Flags
- `has_transcription` - 1 if transcription exists
- `has_translation` - 1 if English translation exists
- `has_romanisation` - 1 if romanisation exists
- `has_korniienko` - 1 if Korniienko images available
- `has_all_images` - 1 if all 4 RTI images available

### Original Data (with HTML)
- `transcription`, `interpretative_edition`, etc. - Original formatting preserved

---

## Utility Scripts

- **`prepare_dataset.py`** - Legacy dataset preparation (older format)
- **`collect_data.py`** - Collect data from backend API (not in scripts/ directory)

---

## Tips & Best Practices

- **Run cropping in background**: `nohup python crop_images.py > crop_log.txt 2>&1 &`
- **Monitor progress**: `tail -f crop_log.txt` or count files in output directories
- **After cropping completes**: Re-run `create_datasets.py` to update image counts
- **Use comprehensive CSVs**: They contain all metadata needed for training
- **Text cleaning**: Already preserves line breaks and single dashes per expert guidance

---

## Troubleshooting

### Cropping Issues
- **Corrupted files**: Script continues and logs errors (e.g., 121-10a, 113-18a, 113-06b, 114-19b)
- **Missing image types**: Script saves whatever is available (`require_all_types=False`)
- **Progress check**: `ls -1 ../data/cropped_images_hq/train/original/ | wc -l`

### Memory Issues
- Large images handled by PIL with `Image.MAX_IMAGE_PIXELS = None`
- Cropping processes one image at a time (low memory footprint)

### Path Issues
- Scripts assume they're run from the `scripts/` directory
- Data files should be in `../data/` (one level up)
- Korniienko images in `../data/korniienkoimages/`

---

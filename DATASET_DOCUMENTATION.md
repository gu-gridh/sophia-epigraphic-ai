# Saint Sophia Graffiti Dataset Creation & Image Cropping Documentation

## Overview
This documentation describes the complete process of creating training datasets from Saint Sophia Cathedral graffiti inscriptions and preparing multi-type images for AI model training. The project focuses on automated transcription of historical graffiti using multi-modal computer vision.

## Dataset Creation Process

### Source Data
- **Main File**: `inscriptions_with_graffiti_data.csv`
- **Total Records**: 2,226 graffiti annotations
- **Languages**: Church Slavonic, Ukrainian, Polish, Ancient Greek, Armenian, Latin, Russian
- **Writing Systems**: Cyrillic, Latin, Greek, Armenian, Mixed scripts, Glagolitic

### Key Fields Used
```
id                    - Unique annotation identifier
position_on_surface   - Bounding box coordinates (pct:x,y,width,height)
panel_title          - Image panel name (e.g., "118-02")
panel_room           - Room number in cathedral
type_of_inscription  - Classification (Textual graffiti, etc.)
genres               - Content category (Prayer, Commemoration, etc.)
tags                 - Additional metadata tags
elevation            - Physical height position
language             - Language of inscription
writing_system       - Script used
transcription        - Raw text with HTML formatting
interpretative_edition - Scholar interpretation
romanisation         - Latin transliteration
conditions           - Preservation state
```

### Data Processing Steps

#### 1. HTML Tag Cleaning
**Problem**: Transcriptions contain HTML formatting
```html
Original: <p>IWO HOWSKI</p>\r\n\r\n<p>H 1620</p>
Cleaned:  IWO HOWSKI H 1620
```

**Solution**: `clean_transcription()` function removes:
- HTML tags (`<p>`, `<span>`, etc.)
- HTML entities (`&nbsp;`, `&amp;`)
- Line breaks (`\r\n`, `\n`)
- Extra whitespace

#### 2. Bounding Box Extraction
**Format**: `pct:14.69,54.72,19.24,0.91` (percentage coordinates)
**Conversion**: 
- Parse coordinates from string
- Convert percentages to 0-1 range
- Store as separate columns: `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height`

#### 3. Data Filtering
- **Before filtering**: 2,226 records
- **After bbox validation**: Records with valid coordinates only
- **After transcription cleaning**: 1,439 records (removed empty transcriptions)
- **Final dataset**: 1,439 usable annotations

#### 4. Dataset Splitting
```
Train:      1,007 records (70%)
Validation:   216 records (15%)
Test:         216 records (15%)
```
- **Stratified by**: `panel_room` to ensure room distribution
- **Random seed**: 42 (for reproducibility)

### Output Files
```
data/train_dataset.csv    - Training annotations
data/val_dataset.csv      - Validation annotations  
data/test_dataset.csv     - Test annotations
```

## Image Cropping Process

### Why Crop Images?

#### 1. **Focus on Relevant Content**
- Original images are very large (1370040000+ pixels)
- Graffiti annotations are small regions within these images
- Cropping extracts only the inscription area for training

#### 2. **Computational Efficiency**
- Reduces memory usage during training
- Faster data loading and processing
- Standardized input size (224x224 pixels)

#### 3. **Multi-Modal Input**
- Each annotation uses 4 different image types
- Provides richer visual information for recognition

### Image Types Used

#### 1. **Original Image** (`panel-name.jpg`)
- Standard RGB photograph
- High-resolution capture of cathedral wall
- Shows inscription in natural lighting

#### 2. **Blended Map** (`panel-name_blended_map_texture_level_grey.png`)
- Texture-enhanced visualization
- Emphasizes surface details and depth
- Better visibility of worn inscriptions

#### 3. **Normal Map** (`panel-name_normal_map.png`)
- Surface normal information
- Shows 3D surface geometry
- Reveals carved or etched text details

#### 4. **Texture Map** (`panel-name_texture_map.png`)
- Enhanced texture information
- Highlights material variations
- Improves readability of faded text

### Cropping Implementation

#### File Structure
```
data/original/
├── 118-02.jpg
├── 118-02_blended_map_texture_level_grey.jpg
├── 118-02_normal_map.jpg
├── 118-02_texture_map.jpg
└── ...
```

#### Processing Steps
1. **Load annotation data** from CSV files
2. **For each annotation**:
   - Find all 4 image types for the panel
   - Extract bbox coordinates
   - Crop region from each image type
   - Resize to 224x224 pixels
   - Save with annotation ID naming

#### Output Structure
```
data/cropped_images/
├── train/
│   ├── original/     # 126_original.jpg, 338_original.jpg, ...
│   ├── blended/      # 126_blended.jpg, 338_blended.jpg, ...
│   ├── normal/       # 126_normal.jpg, 338_normal.jpg, ...
│   └── texture/      # 126_texture.jpg, 338_texture.jpg, ...
├── val/
│   ├── original/
│   ├── blended/
│   ├── normal/
│   └── texture/
└── test/
    ├── original/
    ├── blended/
    ├── normal/
    └── texture/
```

### Technical Considerations

#### 1. **Large Image Handling**
```python
Image.MAX_IMAGE_PIXELS = None  # Bypass PIL size limits
```

#### 2. **File Matching Logic**
- Handles naming variations (decimated versions)
- Prioritizes non-decimated files when available
- Falls back to available alternatives

#### 3. **Error Handling**
- Creates blank images for missing files
- Logs missing image types
- Continues processing despite individual failures

## 🔧 Usage

### Creating Datasets
```bash
python create_datasets.py
```

### Cropping Images
```bash
python crop_images.py
```

### Custom Parameters
```python
from create_datasets import create_datasets

# Custom dataset creation
train_data, val_data, test_data = create_datasets(
    input_file='data/inscriptions_with_graffiti_data.csv',
    output_dir='data',
    test_size=0.3,      # 30% for test+validation
    val_ratio=0.5,      # Equal test/validation split
    random_state=42     # Reproducible results
)
```

## Statistics

### Dataset Distribution
#### It will be updated with final statistics
```
Languages:
- Church Slavonic: 550 (54.6%)
- Ukrainian: 199 (19.8%)
- Polish: 89 (8.8%)
- Ancient Greek: 51 (5.1%)
- Others: 118 (11.7%)

Writing Systems:
- Cyrillic: 766 (76.1%)
- Latin: 113 (11.2%)
- Greek: 56 (5.6%)
- Others: 72 (7.1%)

Top Image Panels:
- 118-02: 106 annotations
- 115-16: 66 annotations
- 208-02: 40 annotations
```

## Next Steps

After dataset creation and cropping:

1. **Model Architecture**: Design multi-modal encoder for 4 image types
2. **Data Loading**: Create PyTorch DataLoader for cropped images
3. **Training Pipeline**: Implement vision-to-text training loop
4. **Evaluation**: Test on multiple languages and scripts
5. **Inference**: Deploy for new graffiti recognition

## Files Created

### Scripts
- `create_datasets.py` - Dataset creation from main CSV
- `crop_images.py` - Image cropping for all annotations
- `example_create_datasets.py` - Usage example

### Data Files
- `train_dataset.csv` - Training annotations with metadata
- `val_dataset.csv` - Validation annotations
- `test_dataset.csv` - Test annotations
- `data/cropped_images/` - Cropped image regions for training

This documentation ensures reproducibility and understanding of the data preparation process for the Saint Sophia graffiti recognition system.

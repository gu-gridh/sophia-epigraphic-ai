#!/usr/bin/env python3
"""
Dataset Creation Script for Saint Sophia Graffiti Recognition
Creates train/validation/test datasets from the main graffiti data file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re

def parse_position(pos_str):
    """
    Parse position_on_surface string to extract bbox coordinates.
    
    Args:
        pos_str: String in format "pct:x,y,width,height"
        
    Returns:
        tuple: (x, y, width, height) as floats in 0-1 range, or (None, None, None, None)
    """
    if pd.isna(pos_str) or not pos_str.startswith('pct:'):
        return None, None, None, None
    try:
        coords = pos_str.replace('pct:', '').split(',')
        if len(coords) == 4:
            x, y, width, height = [float(c) for c in coords]
            # Convert percentages to 0-1 range
            return x/100, y/100, width/100, height/100
    except:
        return None, None, None, None
    return None, None, None, None

def clean_transcription(text):
    """
    Clean HTML tags and formatting from transcription text.
    
    Args:
        text: Raw transcription text with HTML tags
        
    Returns:
        str: Clean text without HTML tags and extra whitespace
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', str(text))
    
    # Remove common HTML entities
    clean_text = clean_text.replace('&nbsp;', ' ')
    clean_text = clean_text.replace('&amp;', '&')
    clean_text = clean_text.replace('&lt;', '<')
    clean_text = clean_text.replace('&gt;', '>')
    clean_text = clean_text.replace('&quot;', '"')
    
    # Remove \r\n and extra whitespace
    clean_text = clean_text.replace('\\r\\n', ' ')
    clean_text = clean_text.replace('\r\n', ' ')
    clean_text = clean_text.replace('\n', ' ')
    clean_text = clean_text.replace('\r', ' ')
    
    # Clean up multiple spaces and strip
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text

def create_datasets(input_file='data/inscriptions_with_graffiti_data_20250811_124547.csv',
                   output_dir='data',
                   test_size=0.3,
                   val_ratio=0.5,
                   random_state=42):
    """
    Create train/validation/test datasets from main graffiti data.
    
    Args:
        input_file: Path to main CSV file
        output_dir: Directory to save datasets
        test_size: Proportion for test+validation (default 0.3 = 30%)
        val_ratio: Ratio of val/(test+val) (default 0.5 = equal test/val)
        random_state: Random seed for reproducibility
    """
    
    print(f'=== Creating Datasets from {input_file} ===')
    
    # Load the main dataset
    df = pd.read_csv(input_file)
    print(f'Total records loaded: {len(df)}')
    
    # Select key fields as specified
    key_fields = ['id', 'position_on_surface', 'panel_title', 'panel_room', 'type_of_inscription', 
                  'genres', 'tags', 'elevation', 'language', 'writing_system', 'transcription', 
                  'interpretative_edition', 'romanisation', 'conditions']
    
    # Create dataset with key fields
    dataset = df[key_fields].copy()
    
    # Parse position_on_surface to extract bbox coordinates
    print('Parsing bbox coordinates...')
    bbox_data = dataset['position_on_surface'].apply(parse_position)
    dataset['bbox_x'] = [b[0] for b in bbox_data]
    dataset['bbox_y'] = [b[1] for b in bbox_data]
    dataset['bbox_width'] = [b[2] for b in bbox_data]
    dataset['bbox_height'] = [b[3] for b in bbox_data]
    
    # Add image_name column (panel_title + .jpg)
    dataset['image_name'] = dataset['panel_title'] + '.jpg'
    
    # Filter out records with missing essential data
    print(f'\nFiltering records...')
    print(f'Before filtering: {len(dataset)}')
    
    # Keep records with valid bbox and transcription
    valid_bbox = dataset['bbox_x'].notna() & dataset['bbox_y'].notna()
    has_transcription = dataset['transcription'].notna() & (dataset['transcription'] != '')
    dataset_filtered = dataset[valid_bbox & has_transcription].copy()
    
    print(f'After filtering (valid bbox + transcription): {len(dataset_filtered)}')
    
    # Clean transcription text (remove HTML tags)
    print('Cleaning transcription text...')
    dataset_filtered['transcription_clean'] = dataset_filtered['transcription'].apply(clean_transcription)
    
    # Update transcription column with cleaned version
    dataset_filtered['transcription'] = dataset_filtered['transcription_clean']
    dataset_filtered = dataset_filtered.drop('transcription_clean', axis=1)
    
    # Filter out empty transcriptions after cleaning
    has_clean_transcription = dataset_filtered['transcription'] != ''
    dataset_filtered = dataset_filtered[has_clean_transcription].copy()
    
    print(f'After cleaning transcription: {len(dataset_filtered)}')
    
    # Check stratification column
    if 'panel_room' in dataset_filtered.columns and dataset_filtered['panel_room'].notna().sum() > 0:
        stratify_col = dataset_filtered['panel_room']
        print(f'Stratifying by panel_room: {len(stratify_col.value_counts())} unique rooms')
    else:
        stratify_col = None
        print('No stratification (panel_room not available)')
    
    # Split into train/val/test
    train_data, temp_data = train_test_split(
        dataset_filtered, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratify_col
    )
    
    if stratify_col is not None:
        # Stratify validation/test split as well
        temp_stratify = temp_data['panel_room']
        val_data, test_data = train_test_split(
            temp_data, 
            test_size=val_ratio, 
            random_state=random_state, 
            stratify=temp_stratify
        )
    else:
        val_data, test_data = train_test_split(
            temp_data, 
            test_size=val_ratio, 
            random_state=random_state
        )
    
    print(f'\n=== Dataset Splits ===')
    print(f'Train: {len(train_data)} records ({len(train_data)/len(dataset_filtered)*100:.1f}%)')
    print(f'Validation: {len(val_data)} records ({len(val_data)/len(dataset_filtered)*100:.1f}%)')
    print(f'Test: {len(test_data)} records ({len(test_data)/len(dataset_filtered)*100:.1f}%)')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    train_path = os.path.join(output_dir, 'train_dataset.csv')
    val_path = os.path.join(output_dir, 'val_dataset.csv')
    test_path = os.path.join(output_dir, 'test_dataset.csv')
    
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f'\n=== Datasets created successfully ===')
    print(f'Files saved:')
    print(f'- {train_path}')
    print(f'- {val_path}')
    print(f'- {test_path}')
    
    # Show analysis
    print(f'\n=== Data Analysis ===')
    print(f'Languages in training set:')
    lang_counts = train_data['language'].value_counts()
    print(lang_counts.head(10))
    
    print(f'\nWriting systems in training set:')
    ws_counts = train_data['writing_system'].value_counts()
    print(ws_counts.head(10))
    
    print(f'\nTop panel_titles (images) in training set:')
    panel_counts = train_data['panel_title'].value_counts()
    print(panel_counts.head(10))
    
    # Show sample from each dataset
    print(f'\n=== Sample from train dataset ===')
    sample_cols = ['id', 'image_name', 'bbox_x', 'bbox_y', 'transcription', 'language']
    print(train_data[sample_cols].head(3))
    
    return train_data, val_data, test_data

def main():
    """Main function to create datasets."""
    try:
        train_data, val_data, test_data = create_datasets()
        print(f'\n Dataset creation completed successfully!')
        
    except Exception as e:
        print(f' Error creating datasets: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

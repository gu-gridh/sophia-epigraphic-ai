#!/usr/bin/env python3
"""
Comprehensive Dataset Creation Script for Saint Sophia Graffiti Recognition

This script handles:
1. Splitting data into train/val/test sets (if needed)
2. Creating comprehensive datasets with:
   - Cropped image filenames (4 types per inscription)
   - Inscription metadata (transcription, translation, etc.)
   - Korniienko reference images
   - Cleaned text for training
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path


def main():
    """Main function to create comprehensive datasets."""
    try:
        print("="*60)
        print("COMPREHENSIVE DATASET CREATION")
        print("="*60)
        print("\nThis script creates comprehensive training datasets with:")
        print("  1. Cropped image filenames (4 types per inscription)")
        print("  2. Inscription metadata (transcription, translation, etc.)")
        print("  3. Korniienko reference images")
        print("  4. Cleaned text for training")
        print("="*60)
        
        # Create comprehensive datasets for each split
        train_df = create_comprehensive_dataset('train')
        val_df = create_comprehensive_dataset('val')
        test_df = create_comprehensive_dataset('test')
        
        # Create combined dataset
        combined_df = create_combined_dataset()
        
        print("\n" + "="*60)
        print("✓ DATASET CREATION COMPLETE")
        print("="*60)
        print("\nCreated files:")
        print("  - ../data/train_comprehensive.csv")
        print("  - ../data/val_comprehensive.csv")
        print("  - ../data/test_comprehensive.csv")
        print("  - ../data/complete_dataset.csv")
        print("\nReady for text cleaning and model training!")
        
    except Exception as e:
        print(f'✗ Error creating datasets: {e}')
        import traceback
        traceback.print_exc()


def check_cropped_images(inscription_id, split_name, base_dir='../data/cropped_images_hq'):
    """
    Check which cropped image types exist for an inscription.
    
    Args:
        inscription_id: ID of the inscription
        split_name: 'train', 'val', or 'test'
        base_dir: Base directory with cropped images
        
    Returns:
        dict: Mapping of image type to filename (or None if missing)
    """
    image_types = ['original', 'blended', 'normal', 'texture']
    image_paths = {}
    
    for img_type in image_types:
        # Check for PNG first (our high-quality format)
        png_path = os.path.join(base_dir, split_name, img_type, f"{inscription_id}_{img_type}.png")
        jpg_path = os.path.join(base_dir, split_name, img_type, f"{inscription_id}_{img_type}.jpg")
        
        if os.path.exists(png_path):
            image_paths[f'{img_type}_image'] = f"cropped_images_hq/{split_name}/{img_type}/{inscription_id}_{img_type}.png"
        elif os.path.exists(jpg_path):
            image_paths[f'{img_type}_image'] = f"cropped_images_hq/{split_name}/{img_type}/{inscription_id}_{img_type}.jpg"
        else:
            image_paths[f'{img_type}_image'] = None
    
    # Count available images
    image_paths['num_available_images'] = sum(1 for v in image_paths.values() if v is not None and v != 0)
    
    return image_paths


def get_korniienko_images(inscription_id, korniienko_df):
    """
    Get Korniienko reference images for an inscription.
    
    Args:
        inscription_id: ID of the inscription
        korniienko_df: DataFrame with Korniienko images
        
    Returns:
        dict: Korniienko image information with local file paths
    """
    # Filter for this inscription
    images = korniienko_df[korniienko_df['inscription_id'] == inscription_id]
    
    result = {
        'korniienko_count': len(images),
        'korniienko_photo': None,
        'korniienko_drawing': None,
        'korniienko_bibliography': None,
        'korniienko_plate': None,
        'korniienko_year': None
    }
    
    if len(images) > 0:
        # Get photo and drawing - extract filename from URL and use local path
        photo = images[images['type_of_image'] == 'Photograph']
        drawing = images[images['type_of_image'] == 'Drawing']
        
        if len(photo) > 0:
            # Extract filename from URL and create local path
            url = photo.iloc[0]['url']
            filename = url.split('/')[-1]  # Get filename from URL
            result['korniienko_photo'] = f"korniienkoimages/{filename}"
            
        if len(drawing) > 0:
            # Extract filename from URL and create local path
            url = drawing.iloc[0]['url']
            filename = url.split('/')[-1]  # Get filename from URL
            result['korniienko_drawing'] = f"korniienkoimages/{filename}"
        
        # Get metadata from first image
        first_img = images.iloc[0]
        result['korniienko_bibliography'] = first_img['bibliography']
        result['korniienko_plate'] = first_img['plate']
        result['korniienko_year'] = first_img['year']
    
    return result


def clean_html_tags(text):
    """Remove HTML tags from text."""
    if pd.isna(text) or text == '':
        return ''
    # Remove <p> tags and other HTML
    text = re.sub(r'<[^>]+>', '', str(text))
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_language_text(lang_dict):
    """Extract language text from dictionary string."""
    if pd.isna(lang_dict) or lang_dict == '':
        return ''
    try:
        # Try to parse as dict
        if isinstance(lang_dict, str):
            # Extract text field using regex
            match = re.search(r"'text':\s*'([^']*)'", lang_dict)
            if match:
                return match.group(1)
        return str(lang_dict)
    except:
        return ''


def create_comprehensive_dataset(split_name):
    """
    Create comprehensive dataset for a split (train/val/test).
    
    Args:
        split_name: 'train', 'val', or 'test'
    """
    print(f"\n{'='*60}")
    print(f"Creating comprehensive dataset for: {split_name}")
    print(f"{'='*60}")
    
    # Load data
    print("Loading data files...")
    dataset_path = f'../data/{split_name}_dataset.csv'
    inscriptions_path = '../data/inscriptions_graffiti_cleaned.csv'  # Use cleaned version
    korniienko_path = '../data/korniienko_images_20251014_124018.csv'
    
    # Read datasets
    split_df = pd.read_csv(dataset_path)
    inscriptions_df = pd.read_csv(inscriptions_path)
    korniienko_df = pd.read_csv(korniienko_path)
    
    print(f"  - Split dataset: {len(split_df)} annotations")
    print(f"  - Full inscriptions: {len(inscriptions_df)} records")
    print(f"  - Korniienko images: {len(korniienko_df)} images")
    
    # Merge with full inscription data (to get any additional fields)
    print("\nMerging datasets...")
    merged_df = split_df.copy()
    
    # Create comprehensive records
    print("Processing inscriptions...")
    records = []
    
    for idx, row in merged_df.iterrows():
        inscription_id = row['id']
        
        # Base record with all original fields
        record = row.to_dict()
        
        # Add cropped image paths
        image_info = check_cropped_images(inscription_id, split_name)
        record.update(image_info)
        
        # Add Korniienko reference images
        korniienko_info = get_korniienko_images(inscription_id, korniienko_df)
        record.update(korniienko_info)
        
        # Use pre-cleaned text fields if available (from clean_text.py)
        # Otherwise fall back to cleaning on the fly
        if 'transcription_clean' in row and pd.notna(row.get('transcription_clean')):
            record['transcription_clean'] = row['transcription_clean']
        else:
            record['transcription_clean'] = clean_html_tags(row.get('transcription', ''))
        
        if 'interpretative_edition_clean' in row and pd.notna(row.get('interpretative_edition_clean')):
            record['interpretative_edition_clean'] = row['interpretative_edition_clean']
        else:
            record['interpretative_edition_clean'] = clean_html_tags(row.get('interpretative_edition', ''))
        
        if 'romanisation_clean' in row and pd.notna(row.get('romanisation_clean')):
            record['romanisation_clean'] = row['romanisation_clean']
        else:
            record['romanisation_clean'] = clean_html_tags(row.get('romanisation', ''))
        
        # Clean other text fields
        record['translation_eng_clean'] = clean_html_tags(row.get('translation_eng', ''))
        record['translation_ukr_clean'] = clean_html_tags(row.get('translation_ukr', ''))
        record['comments_eng_clean'] = clean_html_tags(row.get('comments_eng', ''))
        record['comments_ukr_clean'] = clean_html_tags(row.get('comments_ukr', ''))
        
        # Extract language names
        record['language_name'] = extract_language_text(row.get('language', ''))
        record['writing_system_name'] = extract_language_text(row.get('writing_system', ''))
        
        # Add date range calculations
        if pd.notna(row.get('min_year')) and pd.notna(row.get('max_year')):
            record['date_range'] = int(row['max_year'] - row['min_year'])
            record['date_midpoint'] = int((row['min_year'] + row['max_year']) / 2)
        else:
            record['date_range'] = None
            record['date_midpoint'] = None
        
        # Binary flags
        record['has_transcription'] = 1 if record['transcription_clean'] != '' else 0
        record['has_translation'] = 1 if record['translation_eng_clean'] != '' else 0
        record['has_romanisation'] = 1 if record['romanisation_clean'] != '' else 0
        record['has_korniienko'] = 1 if record['korniienko_count'] > 0 else 0
        record['has_all_images'] = 1 if record['num_available_images'] == 4 else 0
        
        records.append(record)
    
    # Create final dataframe
    final_df = pd.DataFrame(records)
    
    # Reorder columns for better readability
    priority_cols = [
        'id', 'title', 'split',
        # Image files
        'original_image', 'blended_image', 'normal_image', 'texture_image', 
        'num_available_images', 'has_all_images',
        # Korniienko references
        'has_korniienko', 'korniienko_count', 'korniienko_photo', 'korniienko_drawing',
        'korniienko_bibliography', 'korniienko_plate', 'korniienko_year',
        # Panel info
        'panel_id', 'panel_title', 'panel_room',
        # Bbox info
        'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
        # Text content (cleaned)
        'transcription_clean', 'interpretative_edition_clean', 'romanisation_clean',
        'translation_eng_clean', 'translation_ukr_clean',
        'has_transcription', 'has_translation', 'has_romanisation',
        # Metadata
        'type_of_inscription', 'language_name', 'writing_system_name',
        'min_year', 'max_year', 'date_range', 'date_midpoint',
        'elevation', 'height', 'width',
        # Original text (with HTML)
        'transcription', 'interpretative_edition', 'romanisation',
        'translation_eng', 'translation_ukr',
        'comments_eng_clean', 'comments_ukr_clean',
        'comments_eng', 'comments_ukr',
        # URLs
        'iiif_url', 'position_on_surface',
        # Other fields
        'inscriber', 'language', 'writing_system', 'published'
    ]
    
    # Add split column
    final_df['split'] = split_name
    
    # Reorder columns (keep extra columns at the end)
    existing_priority = [col for col in priority_cols if col in final_df.columns]
    other_cols = [col for col in final_df.columns if col not in existing_priority]
    final_df = final_df[existing_priority + other_cols]
    
    # Save dataset
    output_path = f'../data/{split_name}_comprehensive.csv'
    final_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved comprehensive dataset: {output_path}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Dataset Statistics for {split_name}")
    print(f"{'='*60}")
    print(f"Total inscriptions: {len(final_df)}")
    print(f"\nImage availability:")
    print(f"  - All 4 image types: {final_df['has_all_images'].sum()} ({final_df['has_all_images'].sum()/len(final_df)*100:.1f}%)")
    print(f"  - Average images per inscription: {final_df['num_available_images'].mean():.2f}")
    print(f"  - At least 1 image: {(final_df['num_available_images'] > 0).sum()}")
    
    print(f"\nText content:")
    print(f"  - With transcription: {final_df['has_transcription'].sum()} ({final_df['has_transcription'].sum()/len(final_df)*100:.1f}%)")
    print(f"  - With translation: {final_df['has_translation'].sum()} ({final_df['has_translation'].sum()/len(final_df)*100:.1f}%)")
    print(f"  - With romanisation: {final_df['has_romanisation'].sum()} ({final_df['has_romanisation'].sum()/len(final_df)*100:.1f}%)")
    
    print(f"\nKorniienko references:")
    print(f"  - With Korniienko: {final_df['has_korniienko'].sum()} ({final_df['has_korniienko'].sum()/len(final_df)*100:.1f}%)")
    print(f"  - With photo: {final_df['korniienko_photo'].notna().sum()}")
    print(f"  - With drawing: {final_df['korniienko_drawing'].notna().sum()}")
    
    print(f"\nLanguages:")
    if 'language_name' in final_df.columns:
        lang_counts = final_df['language_name'].value_counts()
        for lang, count in lang_counts.head(5).items():
            print(f"  - {lang}: {count}")
    
    print(f"\nDate ranges:")
    if final_df['date_midpoint'].notna().sum() > 0:
        print(f"  - Earliest: {int(final_df['min_year'].min())}")
        print(f"  - Latest: {int(final_df['max_year'].max())}")
        print(f"  - Average date: {int(final_df['date_midpoint'].mean())}")
    
    return final_df


def create_combined_dataset():
    """Combine all splits into one master dataset."""
    print(f"\n{'='*60}")
    print(f"Creating combined master dataset")
    print(f"{'='*60}")
    
    # Load all comprehensive datasets
    train_df = pd.read_csv('../data/train_comprehensive.csv')
    val_df = pd.read_csv('../data/val_comprehensive.csv')
    test_df = pd.read_csv('../data/test_comprehensive.csv')
    
    # Combine
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Save
    output_path = '../data/complete_dataset.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"✓ Saved combined dataset: {output_path}")
    
    print(f"\nCombined dataset statistics:")
    print(f"  - Total inscriptions: {len(combined_df)}")
    print(f"  - Train: {len(train_df)} ({len(train_df)/len(combined_df)*100:.1f}%)")
    print(f"  - Val: {len(val_df)} ({len(val_df)/len(combined_df)*100:.1f}%)")
    print(f"  - Test: {len(test_df)} ({len(test_df)/len(combined_df)*100:.1f}%)")
    
    return combined_df

if __name__ == '__main__':
    main()

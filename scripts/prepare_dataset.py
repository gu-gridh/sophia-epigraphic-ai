#!/usr/bin/env python3
"""
Data preparation script for SOPHIA.
Converts Saint Sophia dataset to SOPHIA training format.
"""

import os
import sys
import pandas as pd
import json
import argparse
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


def setup_data_directories(data_dir: str):
    """Create necessary data directories."""
    directories = [
        'train', 'val', 'test',
        'images', 'annotations',
        'processed'
    ]
    
    for directory in directories:
        dir_path = Path(data_dir) / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")


def convert_saint_sophia_dataset(
    csv_path: str,
    annotations_dir: str,
    images_dir: str,
    output_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.1
):
    """
    Convert Saint Sophia dataset to SOPHIA format.
    
    Args:
        csv_path: Path to combined dataset CSV
        annotations_dir: Directory containing annotation JSON files
        images_dir: Directory containing inscription images
        output_dir: Output directory for processed data
        test_size: Fraction of data for testing
        val_size: Fraction of data for validation
    """
    
    print("Loading Saint Sophia dataset.")
    data = pd.read_csv(csv_path)
    
    print(f"Loaded {len(data)} samples")
    
    # Filter for complete samples (with transcription)
    complete_data = data[
        (data['transcription'].notna()) & 
        (data['transcription'] != '') &
        (data['annotation_index'] >= 0)  # Has annotation data
    ].copy()
    
    print(f"Found {len(complete_data)} complete samples with transcriptions and annotations")
    
    # Check data availability
    print("\nChecking data availability.")
    available_data = []
    
    for idx, row in complete_data.iterrows():
        inscription_id = row['id']
        
        # Check for annotation file
        annotation_file = Path(annotations_dir) / f"annotation_{inscription_id}.json"
        has_annotation = annotation_file.exists()
        
        # Check for image file
        image_file = find_image_file(images_dir, inscription_id, row.get('panel_title', ''))
        has_image = image_file is not None
        
        if has_annotation and has_image:
            row_data = row.to_dict()
            row_data['image_path'] = image_file
            row_data['annotation_path'] = str(annotation_file)
            available_data.append(row_data)
        
        if len(available_data) % 100 == 0:
            print(f"Processed {len(available_data)} available samples.")
    
    print(f"Found {len(available_data)} samples with both images and annotations")
    
    if len(available_data) == 0:
        print("No complete samples found! Check your data paths.")
        return False
    
    # Convert to DataFrame
    available_df = pd.DataFrame(available_data)
    
    # Split data
    print("\nSplitting dataset.")
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        available_df, 
        test_size=test_size, 
        random_state=42,
        stratify=available_df.get('language', None) if 'language' in available_df.columns else None
    )
    
    # Second split: train vs val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),  # Adjust for already split test set
        random_state=42,
        stratify=train_val_df.get('language', None) if 'language' in train_val_df.columns else None
    )
    
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples") 
    print(f"Test: {len(test_df)} samples")
    
    # Setup output directories
    setup_data_directories(output_dir)
    
    # Copy images and annotations to organized structure
    print("\nCopying data files...")
    
    def copy_data_files(df, split_name):
        images_out_dir = Path(output_dir) / 'images'
        annotations_out_dir = Path(output_dir) / 'annotations'
        
        for idx, row in df.iterrows():
            inscription_id = row['id']
            
            # Copy image
            src_image = row['image_path']
            dst_image = images_out_dir / f"{inscription_id}.jpg"
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)
            
            # Copy annotation
            src_annotation = row['annotation_path']
            dst_annotation = annotations_out_dir / f"annotation_{inscription_id}.json"
            if not dst_annotation.exists():
                shutil.copy2(src_annotation, dst_annotation)
    
    # Copy files for each split
    copy_data_files(train_df, 'train')
    copy_data_files(val_df, 'val')
    copy_data_files(test_df, 'test')
    
    # Save split CSVs
    output_path = Path(output_dir)
    train_df.to_csv(output_path / 'train_dataset.csv', index=False)
    val_df.to_csv(output_path / 'val_dataset.csv', index=False)
    test_df.to_csv(output_path / 'test_dataset.csv', index=False)
    
    # Create dataset statistics
    stats = {
        'total_samples': len(available_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'unique_inscriptions': available_df['id'].nunique(),
        'languages': available_df['language'].value_counts().to_dict() if 'language' in available_df.columns else {},
        'average_transcription_length': available_df['transcription'].str.len().mean(),
        'annotation_types': {}
    }
    
    # Save statistics
    with open(output_path / 'dataset_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nDataset preparation completed!")
    print(f"Output directory: {output_dir}")
    print(f"Dataset statistics saved to: {output_path / 'dataset_stats.json'}")
    
    return True


def find_image_file(images_dir: str, inscription_id: str, panel_title: str = '') -> str:
    """Find image file for inscription."""
    images_path = Path(images_dir)
    
    # Try multiple naming conventions
    possible_names = [
        f"{inscription_id}.jpg",
        f"{inscription_id}.png",
        f"{inscription_id}.jpeg",
        f"inscription_{inscription_id}.jpg",
        f"inscription_{inscription_id}.png"
    ]
    
    if panel_title:
        possible_names.extend([
            f"{panel_title}.jpg",
            f"{panel_title}.png",
            f"{panel_title}.jpeg"
        ])
    
    for name in possible_names:
        image_path = images_path / name
        if image_path.exists():
            return str(image_path)
    
    return None


def validate_dataset(output_dir: str):
    """Validate the prepared dataset."""
    print("\nValidating dataset.")
    
    output_path = Path(output_dir)
    
    splits = ['train', 'val', 'test']
    for split in splits:
        csv_path = output_path / f'{split}_dataset.csv'
        if not csv_path.exists():
            print(f"ERROR: Missing {split}_dataset.csv")
            continue
        
        df = pd.read_csv(csv_path)
        print(f"{split.upper()}: {len(df)} samples")
        
        # Check file existence
        missing_images = 0
        missing_annotations = 0
        
        for _, row in df.iterrows():
            inscription_id = row['id']
            
            image_path = output_path / 'images' / f"{inscription_id}.jpg"
            if not image_path.exists():
                missing_images += 1
            
            annotation_path = output_path / 'annotations' / f"annotation_{inscription_id}.json"
            if not annotation_path.exists():
                missing_annotations += 1
        
        if missing_images > 0:
            print(f"  WARNING: {missing_images} missing images")
        if missing_annotations > 0:
            print(f"  WARNING: {missing_annotations} missing annotations")
        
        if missing_images == 0 and missing_annotations == 0:
            print(f"  All files present")
    
    print("Validation completed!")


def main():
    parser = argparse.ArgumentParser(description='Prepare Saint Sophia dataset for SOPHIA training')
    parser.add_argument('--csv_path', required=True, help='Path to combined dataset CSV')
    parser.add_argument('--annotations_dir', required=True, help='Directory containing annotation JSON files')
    parser.add_argument('--images_dir', required=True, help='Directory containing inscription images')
    parser.add_argument('--output_dir', default='./data', help='Output directory for processed data')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data for testing')
    parser.add_argument('--val_size', type=float, default=0.1, help='Fraction of data for validation')
    parser.add_argument('--validate', action='store_true', help='Validate dataset after preparation')
    
    args = parser.parse_args()
    
    # Check input paths
    if not os.path.exists(args.csv_path):
        print(f"ERROR: CSV file not found: {args.csv_path}")
        return 1
    
    if not os.path.exists(args.annotations_dir):
        print(f"ERROR: Annotations directory not found: {args.annotations_dir}")
        return 1
    
    if not os.path.exists(args.images_dir):
        print(f"ERROR: Images directory not found: {args.images_dir}")
        return 1
    
    # Convert dataset
    success = convert_saint_sophia_dataset(
        csv_path=args.csv_path,
        annotations_dir=args.annotations_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    if not success:
        return 1
    
    # Validate if requested
    if args.validate:
        validate_dataset(args.output_dir)
    
    print("\n" + "="*50)
    print("Data preparation completed successfully!")
    print(f"Ready to train SOPHIA with data in: {args.output_dir}")
    print("="*50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

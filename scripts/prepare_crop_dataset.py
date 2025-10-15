#!/usr/bin/env python3
"""
Prepare dataset for cropping by parsing position_on_surface into bbox coordinates.
Converts IIIF format (pct:x,y,w,h) to normalized bbox values.
"""

import pandas as pd
import numpy as np
import os

def parse_position_on_surface(position_str):
    """
    Parse IIIF position string into bbox coordinates.
    
    Args:
        position_str: String like "pct:5.50,48.65,66.62,5.69"
        
    Returns:
        tuple: (bbox_x, bbox_y, bbox_width, bbox_height) in 0-1 range
    """
    try:
        if pd.isna(position_str) or not position_str:
            return None, None, None, None
        
        # Remove 'pct:' prefix
        if position_str.startswith('pct:'):
            position_str = position_str[4:]
        
        # Split by comma
        parts = position_str.split(',')
        if len(parts) != 4:
            return None, None, None, None
        
        # Convert to float and normalize (already in percentage 0-100)
        x, y, w, h = [float(p) / 100.0 for p in parts]
        
        return x, y, w, h
        
    except Exception as e:
        print(f"Error parsing position: {position_str} - {e}")
        return None, None, None, None

def prepare_dataset(input_csv, output_csv):
    """
    Prepare dataset by parsing position_on_surface into bbox coordinates.
    
    Args:
        input_csv: Path to input CSV (inscriptions_graffiti_*.csv)
        output_csv: Path to output CSV with bbox columns
    """
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Total inscriptions: {len(df)}")
    
    # Parse position_on_surface into bbox columns
    print("Parsing position_on_surface into bbox coordinates...")
    bbox_data = df['position_on_surface'].apply(parse_position_on_surface)
    
    # Add bbox columns
    df['bbox_x'] = bbox_data.apply(lambda x: x[0] if x else None)
    df['bbox_y'] = bbox_data.apply(lambda x: x[1] if x else None)
    df['bbox_width'] = bbox_data.apply(lambda x: x[2] if x else None)
    df['bbox_height'] = bbox_data.apply(lambda x: x[3] if x else None)
    
    # Filter out rows with invalid bbox
    valid_bbox = df[['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].notna().all(axis=1)
    df_valid = df[valid_bbox].copy()
    
    print(f"Inscriptions with valid bbox: {len(df_valid)} ({len(df_valid)/len(df)*100:.1f}%)")
    print(f"Inscriptions without valid bbox: {len(df) - len(df_valid)}")
    
    # Save prepared dataset
    df_valid.to_csv(output_csv, index=False)
    print(f"\nSaved prepared dataset to: {output_csv}")
    
    # Print bbox statistics
    print(f"\nBbox statistics:")
    print(f"  X range: {df_valid['bbox_x'].min():.3f} - {df_valid['bbox_x'].max():.3f}")
    print(f"  Y range: {df_valid['bbox_y'].min():.3f} - {df_valid['bbox_y'].max():.3f}")
    print(f"  Width range: {df_valid['bbox_width'].min():.3f} - {df_valid['bbox_width'].max():.3f}")
    print(f"  Height range: {df_valid['bbox_height'].min():.3f} - {df_valid['bbox_height'].max():.3f}")
    print(f"  Avg width: {df_valid['bbox_width'].mean():.3f}")
    print(f"  Avg height: {df_valid['bbox_height'].mean():.3f}")
    
    return df_valid

def split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split dataset into train/val/test sets.
    
    Args:
        df: DataFrame to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    np.random.seed(random_seed)
    
    # Shuffle the dataset
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    n = len(df_shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df_shuffled[:train_end]
    val_df = df_shuffled[train_end:val_end]
    test_df = df_shuffled[val_end:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    print(f"  Val: {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/n*100:.1f}%)")
    
    return train_df, val_df, test_df

def main():
    """Main function."""
    # Paths
    data_dir = '../data'
    
    # Find the graffiti CSV file
    import glob
    graffiti_files = glob.glob(os.path.join(data_dir, 'inscriptions_graffiti_*.csv'))
    
    if not graffiti_files:
        print("Error: No inscriptions_graffiti_*.csv file found in data directory!")
        return
    
    # Use the most recent file
    input_csv = sorted(graffiti_files)[-1]
    print(f"Using input file: {input_csv}")
    
    # Prepare dataset
    output_csv = os.path.join(data_dir, 'graffiti_prepared.csv')
    df_prepared = prepare_dataset(input_csv, output_csv)
    
    # Split into train/val/test
    print("\n" + "="*60)
    print("Splitting dataset into train/val/test...")
    print("="*60)
    
    train_df, val_df, test_df = split_dataset(df_prepared, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Save splits
    train_df.to_csv(os.path.join(data_dir, 'train_dataset.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'val_dataset.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test_dataset.csv'), index=False)
    
    print(f"\nSaved split datasets:")
    print(f"  - {data_dir}/train_dataset.csv")
    print(f"  - {data_dir}/val_dataset.csv")
    print(f"  - {data_dir}/test_dataset.csv")
    
    print("\n" + "="*60)
    print("✅ Dataset preparation complete!")
    print("="*60)
    print("\nNext step: Run crop_images.py to crop the images")

if __name__ == '__main__':
    main()

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


def get_language_text(lang_str):
    """Extract language text from JSON string."""
    try:
        if pd.isna(lang_str) or lang_str == '':
            return 'Unknown'
        lang_dict = eval(lang_str) if isinstance(lang_str, str) else lang_str
        return lang_dict.get('text', 'Unknown') if isinstance(lang_dict, dict) else str(lang_str)
    except:
        return 'Unknown'


def stratified_split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split dataset into train/val/test sets with STRATIFIED sampling by language.
    This ensures each split has proportional representation of all languages.
    
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
    
    # Extract language text for stratification
    df = df.copy()
    df['_language_text'] = df['language'].apply(get_language_text)
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    print(f"\nStratified split by language:")
    print("-" * 60)
    
    # Split each language group proportionally
    for lang in df['_language_text'].unique():
        lang_df = df[df['_language_text'] == lang].sample(frac=1, random_state=random_seed).reset_index(drop=True)
        n = len(lang_df)
        
        if n < 3:
            # Too few samples - put all in training
            train_dfs.append(lang_df)
            print(f"  {lang}: {n} samples (all in train - too few to split)")
            continue
        
        train_end = max(1, int(n * train_ratio))
        val_end = train_end + max(1, int(n * val_ratio))
        
        train_dfs.append(lang_df[:train_end])
        val_dfs.append(lang_df[train_end:val_end])
        test_dfs.append(lang_df[val_end:])
        
        print(f"  {lang}: {n} total → train={len(lang_df[:train_end])}, val={len(lang_df[train_end:val_end])}, test={len(lang_df[val_end:])}")
    
    # Combine all splits
    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True) if val_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True) if test_dfs else pd.DataFrame()
    
    # Remove temporary column
    train_df = train_df.drop(columns=['_language_text'])
    if len(val_df) > 0:
        val_df = val_df.drop(columns=['_language_text'])
    if len(test_df) > 0:
        test_df = test_df.drop(columns=['_language_text'])
    
    total = len(train_df) + len(val_df) + len(test_df)
    print("-" * 60)
    print(f"\nFinal dataset split:")
    print(f"  Train: {len(train_df)} ({len(train_df)/total*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/total*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/total*100:.1f}%)")
    
    return train_df, val_df, test_df


def split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split dataset into train/val/test sets (simple random split).
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--stratified', action='store_true', default=True,
                       help='Use stratified split by language (recommended)')
    parser.add_argument('--random', action='store_true',
                       help='Use simple random split')
    args = parser.parse_args()
    
    use_stratified = not args.random
    
    # Paths
    data_dir = '../data'
    
    # Use the cleaned dataset (has transcription_clean column)
    input_csv = os.path.join(data_dir, 'inscriptions_graffiti_cleaned.csv')
    
    if not os.path.exists(input_csv):
        print(f"Error: Cleaned dataset not found at {input_csv}")
        print("Run clean_text.py first!")
        return
    
    print(f"Using input file: {input_csv}")
    
    # Prepare dataset
    output_csv = os.path.join(data_dir, 'graffiti_prepared.csv')
    df_prepared = prepare_dataset(input_csv, output_csv)
    
    # Filter to only include rows with transcription
    has_text = df_prepared['transcription_clean'].notna() & (df_prepared['transcription_clean'].astype(str).str.strip() != '')
    df_with_text = df_prepared[has_text].copy()
    print(f"\nFiltered to {len(df_with_text)} inscriptions with transcription")
    
    # Split into train/val/test
    print("\n" + "="*60)
    if use_stratified:
        print("Using STRATIFIED split by language (recommended)")
    else:
        print("Using simple random split")
    print("="*60)
    
    if use_stratified:
        train_df, val_df, test_df = stratified_split_dataset(df_with_text, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    else:
        train_df, val_df, test_df = split_dataset(df_with_text, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Save splits
    train_df.to_csv(os.path.join(data_dir, 'train_dataset.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'val_dataset.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test_dataset.csv'), index=False)
    
    print(f"\nSaved split datasets:")
    print(f"  - {data_dir}/train_dataset.csv ({len(train_df)} samples)")
    print(f"  - {data_dir}/val_dataset.csv ({len(val_df)} samples)")
    print(f"  - {data_dir}/test_dataset.csv ({len(test_df)} samples)")
    
    print("\n" + "="*60)
    print(" Dataset preparation complete!")
    print("="*60)
    print("\nNext step: Run crop_images.py to crop the images")

if __name__ == '__main__':
    main()

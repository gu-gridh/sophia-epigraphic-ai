#!/usr/bin/env python3
"""
Download inscription crops from panel IIIF images.

For inscriptions without dedicated images, we can:
1. Download the full panel image via IIIF
2. Apply position_on_surface (pct:x,y,w,h) to crop the inscription
3. Save as a usable training image

Usage:
    python scripts/download_panel_crops.py --output_dir data/panel_crops
"""

import os
import sys
import argparse
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
import re
import time


def parse_position(position_str):
    """
    Parse position_on_surface string like 'pct:12.07,67.01,2.17,0.62'
    Returns (x_pct, y_pct, w_pct, h_pct) as floats
    """
    if not position_str or pd.isna(position_str):
        return None
    
    # Match pct:x,y,w,h format
    match = re.match(r'pct:([\d.]+),([\d.]+),([\d.]+),([\d.]+)', str(position_str))
    if match:
        return tuple(float(x) for x in match.groups())
    return None


def extract_iiif_uuid(iiif_url):
    """
    Extract UUID from IIIF URL like:
    https://img.dh.gu.se/saintsophia/static/inscriptions/iiif/018142ce-1e3c-452f-b74b-88687417fad7.tif/pct:...
    """
    if not iiif_url or pd.isna(iiif_url):
        return None
    
    match = re.search(r'/iiif/([a-f0-9-]+)\.tif/', str(iiif_url))
    if match:
        return match.group(1)
    return None


def download_and_crop(inscription_id, iiif_uuid, position, output_dir, max_width=3000):
    """
    Download panel image and crop to inscription bbox.
    
    Args:
        inscription_id: ID of the inscription
        iiif_uuid: UUID of the panel IIIF image
        position: Tuple (x_pct, y_pct, w_pct, h_pct)
        output_dir: Directory to save cropped images
        max_width: Max width for downloaded panel image
    
    Returns:
        Path to saved crop, or None if failed
    """
    # Construct IIIF URL for full image
    iiif_base = f"https://img.dh.gu.se/saintsophia/static/inscriptions/iiif/{iiif_uuid}.tif"
    full_url = f"{iiif_base}/full/{max_width},/0/default.jpg"
    
    try:
        # Download panel image
        response = requests.get(full_url, timeout=30)
        response.raise_for_status()
        
        # Open image
        img = Image.open(BytesIO(response.content))
        img_width, img_height = img.size
        
        # Calculate crop box from percentage coordinates
        x_pct, y_pct, w_pct, h_pct = position
        
        left = int(img_width * x_pct / 100)
        top = int(img_height * y_pct / 100)
        right = int(img_width * (x_pct + w_pct) / 100)
        bottom = int(img_height * (y_pct + h_pct) / 100)
        
        # Ensure valid crop box
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)
        
        # Check if crop is valid (not too small)
        crop_width = right - left
        crop_height = bottom - top
        
        if crop_width < 10 or crop_height < 10:
            print(f"  Warning: Crop too small for {inscription_id}: {crop_width}x{crop_height}")
            return None
        
        # Crop image
        cropped = img.crop((left, top, right, bottom))
        
        # Save
        output_path = output_dir / f"{inscription_id}_panel_crop.jpg"
        cropped.save(output_path, "JPEG", quality=95)
        
        return output_path
        
    except Exception as e:
        print(f"  Error for {inscription_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Download inscription crops from panel IIIF images')
    parser.add_argument('--data_csv', type=str, default='data/complete_dataset.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--output_dir', type=str, default='data/panel_crops',
                        help='Output directory for cropped images')
    parser.add_argument('--max_width', type=int, default=3000,
                        help='Max width for panel download')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of downloads (for testing)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between requests (seconds)')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_csv}...")
    df = pd.read_csv(args.data_csv)
    print(f"Total inscriptions: {len(df)}")
    
    # Find inscriptions without images but with IIIF URL and position
    has_text = df['transcription_clean'].notna() & (df['transcription_clean'].astype(str).str.strip() != '')
    has_korniienko = (df['korniienko_photo'].notna() & (df['korniienko_photo'].str.strip() != '')) | \
                     (df['korniienko_drawing'].notna() & (df['korniienko_drawing'].str.strip() != ''))
    has_rti = (df['original_image'].notna() & (df['original_image'].str.strip() != '')) | \
              (df['blended_image'].notna() & (df['blended_image'].str.strip() != ''))
    has_any_image = has_korniienko | has_rti
    
    # Need IIIF URL and position
    has_iiif = df['inscription_iiif_url'].notna() & (df['inscription_iiif_url'].str.strip() != '')
    has_position = df['position_on_surface'].notna() & (df['position_on_surface'].str.strip() != '')
    
    # Target: has text, no image, but has IIIF + position
    target_mask = has_text & ~has_any_image & has_iiif & has_position
    target_df = df[target_mask].copy()
    
    print(f"\nInscriptions without images: {(has_text & ~has_any_image).sum()}")
    print(f"With IIIF URL and position: {len(target_df)}")
    
    if len(target_df) == 0:
        print("No inscriptions to process!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Limit if specified
    if args.limit:
        target_df = target_df.head(args.limit)
        print(f"Limited to {args.limit} inscriptions")
    
    # Process each inscription
    successful = 0
    failed = 0
    results = []
    
    print(f"\nDownloading and cropping {len(target_df)} inscriptions...")
    
    for idx, row in tqdm(target_df.iterrows(), total=len(target_df)):
        inscription_id = row['id']
        iiif_url = row['inscription_iiif_url']
        position_str = row['position_on_surface']
        
        # Parse position
        position = parse_position(position_str)
        if not position:
            failed += 1
            continue
        
        # Extract UUID
        iiif_uuid = extract_iiif_uuid(iiif_url)
        if not iiif_uuid:
            failed += 1
            continue
        
        # Download and crop
        output_path = download_and_crop(
            inscription_id, 
            iiif_uuid, 
            position, 
            output_dir,
            args.max_width
        )
        
        if output_path:
            successful += 1
            results.append({
                'id': inscription_id,
                'panel_crop': str(output_path.relative_to(Path('data').parent) if 'data' in str(output_path) else output_path)
            })
        else:
            failed += 1
        
        # Rate limiting
        time.sleep(args.delay)
    
    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output dir: {output_dir}")
    
    # Save results mapping
    if results:
        results_df = pd.DataFrame(results)
        results_path = output_dir / "panel_crops_mapping.csv"
        results_df.to_csv(results_path, index=False)
        print(f"  Mapping saved to: {results_path}")
    
    print(f"{'='*50}")


if __name__ == '__main__':
    main()

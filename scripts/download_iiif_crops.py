#!/usr/bin/env python3
"""
Download inscription crops from IIIF panel images.

For inscriptions without dedicated Korniienko/RTI images, we can extract
crops from the panel IIIF images using the position_on_surface coordinates.

Usage:
    python scripts/download_iiif_crops.py
"""

import os
import sys
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import re

# API base URL
API_BASE = "https://saintsophia.dh.gu.se/api/inscriptions/inscription/"
IIIF_BASE = "https://img.dh.gu.se/saintsophia/static/inscriptions/iiif/"

# Output directory
OUTPUT_DIR = Path("data/iiif_crops")


def parse_iiif_url(inscription_iiif_url):
    """
    Parse the inscription_iiif_url to extract the IIIF identifier and region.
    
    Example URL: https://img.dh.gu.se/saintsophia/static/inscriptions/iiif/018142ce-1e3c-452f-b74b-88687417fad7.tif/pct:12.07,67.01,2.17,0.62/
    """
    if not inscription_iiif_url:
        return None, None
    
    # Extract the UUID and pct coordinates
    match = re.search(r'/iiif/([a-f0-9-]+\.tif)/pct:([0-9.,]+)/?', inscription_iiif_url)
    if match:
        iiif_id = match.group(1)
        pct_coords = match.group(2)
        return iiif_id, f"pct:{pct_coords}"
    
    return None, None


def download_inscription_crop(inscription_id, iiif_id, region, output_path, width=800):
    """
    Download a cropped inscription image from IIIF.
    
    Args:
        inscription_id: The inscription ID
        iiif_id: The IIIF image identifier (UUID.tif)
        region: The region string (e.g., "pct:12.07,67.01,2.17,0.62")
        output_path: Path to save the image
        width: Target width in pixels
    """
    # Build IIIF URL: {base}/{identifier}/{region}/{size}/{rotation}/{quality}.{format}
    url = f"{IIIF_BASE}{iiif_id}/{region}/{width},/0/default.jpg"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"  Failed to download {inscription_id}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  Error downloading {inscription_id}: {e}")
        return False


def get_inscription_iiif_url(inscription_id):
    """Fetch inscription_iiif_url from API."""
    try:
        url = f"{API_BASE}{inscription_id}/"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('inscription_iiif_url', '')
        return None
    except Exception as e:
        return None


def main():
    print("="*70)
    print("DOWNLOAD IIIF INSCRIPTION CROPS")
    print("="*70)
    
    # Load dataset
    df = pd.read_csv('data/complete_dataset.csv')
    print(f"Total inscriptions: {len(df)}")
    
    # Find inscriptions without images
    has_text = df['transcription_clean'].notna() & (df['transcription_clean'].astype(str).str.strip() != '')
    has_korniienko = (df['korniienko_photo'].notna() & (df['korniienko_photo'].str.strip() != '')) | \
                     (df['korniienko_drawing'].notna() & (df['korniienko_drawing'].str.strip() != ''))
    has_rti = (df['original_image'].notna() & (df['original_image'].str.strip() != '')) | \
              (df['blended_image'].notna() & (df['blended_image'].str.strip() != ''))
    
    has_any_image = has_korniienko | has_rti
    needs_image = has_text & ~has_any_image
    
    missing_df = df[needs_image]
    print(f"Inscriptions needing images: {len(missing_df)}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download images
    downloaded = 0
    failed = 0
    no_iiif = 0
    
    for idx, row in tqdm(missing_df.iterrows(), total=len(missing_df), desc="Downloading"):
        inscription_id = row['id']
        output_path = OUTPUT_DIR / f"{inscription_id}_iiif_crop.jpg"
        
        # Skip if already exists
        if output_path.exists():
            downloaded += 1
            continue
        
        # Get IIIF URL from API
        iiif_url = get_inscription_iiif_url(inscription_id)
        
        if not iiif_url:
            no_iiif += 1
            continue
        
        # Parse IIIF URL
        iiif_id, region = parse_iiif_url(iiif_url)
        
        if not iiif_id or not region:
            no_iiif += 1
            continue
        
        # Download
        if download_inscription_crop(inscription_id, iiif_id, region, output_path):
            downloaded += 1
        else:
            failed += 1
        
        # Rate limiting
        time.sleep(0.1)
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Downloaded: {downloaded}")
    print(f"Failed: {failed}")
    print(f"No IIIF URL: {no_iiif}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Update CSV with new image paths
    if downloaded > 0:
        print(f"\nUpdating complete_dataset.csv with new image paths...")
        
        # Add iiif_crop column
        df['iiif_crop'] = ''
        for f in OUTPUT_DIR.glob('*_iiif_crop.jpg'):
            inscription_id = int(f.stem.replace('_iiif_crop', ''))
            rel_path = f"iiif_crops/{f.name}"
            df.loc[df['id'] == inscription_id, 'iiif_crop'] = rel_path
        
        df.to_csv('data/complete_dataset.csv', index=False)
        
        # Count new usable samples
        has_iiif = df['iiif_crop'].notna() & (df['iiif_crop'].str.strip() != '')
        new_total = (has_any_image | has_iiif).sum()
        print(f"New total with images: {new_total}")


if __name__ == '__main__':
    main()

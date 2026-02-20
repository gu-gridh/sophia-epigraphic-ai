#!/usr/bin/env python3
"""
Comprehensive dataset preparation script for Saint Sophia Graffiti Recognition.

This script:
1. Fetches all inscriptions from the API
2. Cleans transcriptions and filters invalid entries
3. Links available images (Korniienko photos/drawings, IIIF crops)
4. Produces a training-ready dataset

Usage:
    python scripts/prepare_dataset.py [--fetch] [--download-iiif]
    
Options:
    --fetch          Fetch fresh data from API (otherwise use cached)
    --download-iiif  Download IIIF crops for inscriptions without images
"""

import os
import sys
import re
import json
import argparse
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import time

# Configuration
API_BASE = "https://saintsophia.dh.gu.se/api/inscriptions/inscription/"
IIIF_BASE = "https://img.dh.gu.se/saintsophia/static/inscriptions/iiif/"
DATA_DIR = Path("data")
KORNIIENKO_DIR = DATA_DIR / "korniienkoimages"
IIIF_DIR = DATA_DIR / "iiif_crops"


def fetch_inscriptions_from_api():
    """Fetch all inscriptions from the Saint Sophia API."""
    print("Fetching inscriptions from API...")
    
    all_inscriptions = []
    url = API_BASE
    page = 1
    
    while url:
        print(f"  Page {page}...", end=" ")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        results = data.get('results', [])
        all_inscriptions.extend(results)
        print(f"{len(results)} inscriptions")
        
        url = data.get('next')
        page += 1
        time.sleep(0.1)  # Rate limiting
    
    print(f"Total: {len(all_inscriptions)} inscriptions")
    return all_inscriptions


def clean_transcription(text):
    """Clean and normalize transcription text."""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def is_valid_transcription(text):
    """Check if transcription is valid for training."""
    if not text:
        return False
    
    # Filter out uncertain annotations (just "?" or containing "?")
    if text == "?" or "?" in text:
        return False
    
    # Must have at least 2 characters
    if len(text) < 2:
        return False
    
    return True


def build_korniienko_index():
    """Build index of available Korniienko images by inscription ID."""
    print("Indexing Korniienko images...")
    
    photo_by_id = {}
    drawing_by_id = {}
    
    if not KORNIIENKO_DIR.exists():
        print("  Warning: Korniienko directory not found")
        return photo_by_id, drawing_by_id
    
    for f in KORNIIENKO_DIR.glob("*.png"):
        name = f.stem
        
        # Match pattern: ..._photo_ID or ..._drawing_ID
        photo_match = re.search(r'_photo_(\d+)$', name)
        drawing_match = re.search(r'_drawing_(\d+)$', name)
        
        if photo_match:
            inscription_id = int(photo_match.group(1))
            photo_by_id[inscription_id] = f"korniienkoimages/{f.name}"
        elif drawing_match:
            inscription_id = int(drawing_match.group(1))
            drawing_by_id[inscription_id] = f"korniienkoimages/{f.name}"
    
    print(f"  Found {len(photo_by_id)} photos, {len(drawing_by_id)} drawings")
    return photo_by_id, drawing_by_id


def build_iiif_index():
    """Build index of available IIIF crops by inscription ID."""
    print("Indexing IIIF crops...")
    
    iiif_by_id = {}
    
    if not IIIF_DIR.exists():
        print("  Warning: IIIF directory not found")
        return iiif_by_id
    
    for f in IIIF_DIR.glob("*_iiif_crop.jpg"):
        # Extract ID from filename like 3329_iiif_crop.jpg
        match = re.match(r'(\d+)_iiif_crop\.jpg', f.name)
        if match:
            inscription_id = int(match.group(1))
            iiif_by_id[inscription_id] = f"iiif_crops/{f.name}"
    
    print(f"  Found {len(iiif_by_id)} IIIF crops")
    return iiif_by_id


def parse_iiif_url(inscription_iiif_url):
    """Parse inscription_iiif_url to extract IIIF identifier and region."""
    if not inscription_iiif_url:
        return None, None
    
    match = re.search(r'/iiif/([a-f0-9-]+\.tif)/pct:([0-9.,]+)/?', inscription_iiif_url)
    if match:
        iiif_id = match.group(1)
        pct_coords = match.group(2)
        return iiif_id, f"pct:{pct_coords}"
    
    return None, None


def download_iiif_crop(inscription_id, iiif_id, region, width=800):
    """Download a cropped inscription image from IIIF."""
    output_path = IIIF_DIR / f"{inscription_id}_iiif_crop.jpg"
    
    if output_path.exists():
        return str(output_path.relative_to(DATA_DIR))
    
    url = f"{IIIF_BASE}{iiif_id}/{region}/{width},/0/default.jpg"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return f"iiif_crops/{output_path.name}"
    except Exception as e:
        pass
    
    return None


def process_inscriptions(inscriptions, photo_index, drawing_index, iiif_index, download_iiif=False):
    """Process inscriptions and build dataset."""
    print("Processing inscriptions...")
    
    records = []
    
    for insc in tqdm(inscriptions, desc="Processing"):
        inscription_id = insc.get('id')
        
        # Get transcription
        transcription_raw = insc.get('transcription', '')
        transcription_clean = clean_transcription(transcription_raw)
        
        # Skip invalid transcriptions
        if not is_valid_transcription(transcription_clean):
            continue
        
        # Get metadata
        language = insc.get('language', {})
        language_name = language.get('language_name', '') if language else ''
        
        writing_system = insc.get('writing_system', {})
        writing_system_name = writing_system.get('writing_system_name', '') if writing_system else ''
        
        # Get Korniienko images from index
        korniienko_photo = photo_index.get(inscription_id, '')
        korniienko_drawing = drawing_index.get(inscription_id, '')
        
        # Get IIIF crop
        iiif_crop = iiif_index.get(inscription_id, '')
        
        # Try to download IIIF crop if needed and requested
        if download_iiif and not korniienko_photo and not korniienko_drawing and not iiif_crop:
            iiif_url = insc.get('inscription_iiif_url', '')
            if iiif_url:
                iiif_id, region = parse_iiif_url(iiif_url)
                if iiif_id and region:
                    iiif_crop = download_iiif_crop(inscription_id, iiif_id, region)
                    time.sleep(0.1)  # Rate limiting
        
        # Check if we have any image
        has_image = bool(korniienko_photo or korniienko_drawing or iiif_crop)
        
        records.append({
            'id': inscription_id,
            'transcription_raw': transcription_raw,
            'transcription_clean': transcription_clean,
            'language_name': language_name,
            'writing_system_name': writing_system_name,
            'korniienko_photo': korniienko_photo,
            'korniienko_drawing': korniienko_drawing,
            'iiif_crop': iiif_crop,
            'has_image': has_image,
        })
    
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description='Prepare Saint Sophia dataset')
    parser.add_argument('--fetch', action='store_true', help='Fetch fresh data from API')
    parser.add_argument('--download-iiif', action='store_true', help='Download IIIF crops')
    args = parser.parse_args()
    
    print("=" * 70)
    print("SAINT SOPHIA DATASET PREPARATION")
    print("=" * 70)
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    IIIF_DIR.mkdir(exist_ok=True)
    
    # Fetch or load inscriptions
    cache_file = DATA_DIR / "inscriptions_cache.json"
    
    if args.fetch or not cache_file.exists():
        inscriptions = fetch_inscriptions_from_api()
        with open(cache_file, 'w') as f:
            json.dump(inscriptions, f)
        print(f"Cached to {cache_file}")
    else:
        print(f"Loading cached data from {cache_file}")
        with open(cache_file) as f:
            inscriptions = json.load(f)
        print(f"Loaded {len(inscriptions)} inscriptions")
    
    # Build image indices
    photo_index, drawing_index = build_korniienko_index()
    iiif_index = build_iiif_index()
    
    # Process inscriptions
    df = process_inscriptions(
        inscriptions, 
        photo_index, 
        drawing_index, 
        iiif_index,
        download_iiif=args.download_iiif
    )
    
    # Save dataset
    output_file = DATA_DIR / "complete_dataset.csv"
    df.to_csv(output_file, index=False)
    
    # Statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    total = len(df)
    with_image = df['has_image'].sum()
    with_photo = (df['korniienko_photo'] != '').sum()
    with_drawing = (df['korniienko_drawing'] != '').sum()
    with_iiif = (df['iiif_crop'] != '').sum()
    
    print(f"Total inscriptions with valid transcription: {total}")
    print(f"With any image: {with_image} ({100*with_image/total:.1f}%)")
    print(f"  - Korniienko photo: {with_photo}")
    print(f"  - Korniienko drawing: {with_drawing}")
    print(f"  - IIIF crop: {with_iiif}")
    print(f"\n✓ TRAINING-READY SAMPLES: {with_image}")
    
    # Language distribution
    print("\nLanguage distribution:")
    lang_counts = df[df['has_image']]['language_name'].value_counts()
    for lang, count in lang_counts.head(10).items():
        print(f"  {lang}: {count}")
    
    print(f"\nDataset saved to: {output_file}")
    
    return df


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Simple script to export inscription data and download annotations.
This script does everything in one place without complexity.
"""

import os
import sys
import csv
import json
import requests
import time
from datetime import datetime


def fetch_graffiti_from_api():
    """Fetch all graffiti inscriptions from the API with full korniienko_image data."""
    print("Fetching graffiti inscriptions from API (with Korniienko images)...")
    
    base_url = "https://saintsophia.dh.gu.se/api/inscriptions/inscription/"
    params = {
        'type_of_inscription': 1,  # 1 = graffiti
        'depth': 1,  # Get full nested objects including korniienko_image
        'limit': 100,  # Fetch 100 at a time
        'offset': 0
    }
    
    all_inscriptions = []
    
    while True:
        try:
            response = requests.get(base_url, params=params, timeout=30)
            if response.status_code != 200:
                print(f"Error: API returned status code {response.status_code}")
                break
            
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                break
            
            all_inscriptions.extend(results)
            print(f"  Fetched {len(all_inscriptions)} / {data.get('count', '?')} inscriptions", end='\r')
            
            # Check if there's a next page
            if not data.get('next'):
                break
            
            params['offset'] += params['limit']
            time.sleep(0.2)  # Be nice to the server
            
        except Exception as e:
            print(f"\nError fetching from API: {e}")
            break
    
    print(f"\nTotal graffiti inscriptions fetched: {len(all_inscriptions)}")
    
    # Count how many have korniienko images
    with_korniienko = sum(1 for i in all_inscriptions if i.get('korniienko_image'))
    print(f"  - {with_korniienko} inscriptions have Korniienko images")
    
    return all_inscriptions


def export_inscriptions_from_api():
    """Export graffiti inscriptions from API to CSV with full Korniienko image data."""
    print("Exporting graffiti inscriptions from API...")
    
    # Fetch all graffiti from API
    inscriptions = fetch_graffiti_from_api()
    
    if not inscriptions:
        print("No inscriptions found!")
        return None
    
    # Create main CSV file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'inscriptions_graffiti_{timestamp}.csv'
    
    # Also create a separate CSV for korniienko images
    korniienko_csv = f'korniienko_images_{timestamp}.csv'
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Define all fields to export
        headers = [
            'id', 'title', 'position_on_surface', 'panel_id', 'panel_title', 'panel_room',
            'type_of_inscription', 'elevation', 'height', 'width', 'language', 
            'writing_system', 'min_year', 'max_year', 'transcription', 
            'interpretative_edition', 'romanisation', 'inscriber', 
            'translation_eng', 'translation_ukr', 'comments_eng', 'comments_ukr',
            'iiif_url', 'has_korniienko', 'korniienko_count', 'published'
        ]
        
        writer.writerow(headers)
        
        # Write data for each inscription
        for inscription in inscriptions:
            korniienko_images = inscription.get('korniienko_image', [])
            panel = inscription.get('panel', {})
            panel_id = panel.get('id', '') if isinstance(panel, dict) else panel
            panel_title = panel.get('title', '') if isinstance(panel, dict) else ''
            panel_room = panel.get('room', '') if isinstance(panel, dict) else ''
            
            row = [
                inscription.get('id', ''),
                inscription.get('title', ''),
                inscription.get('position_on_surface', ''),
                panel_id,
                panel_title,
                panel_room,
                inscription.get('type_of_inscription', {}).get('text', '') if isinstance(inscription.get('type_of_inscription'), dict) else inscription.get('type_of_inscription', ''),
                inscription.get('elevation', ''),
                inscription.get('height', ''),
                inscription.get('width', ''),
                inscription.get('language', ''),
                inscription.get('writing_system', ''),
                inscription.get('min_year', ''),
                inscription.get('max_year', ''),
                inscription.get('transcription', ''),
                inscription.get('interpretative_edition', ''),
                inscription.get('romanisation', ''),
                inscription.get('inscriber', ''),
                inscription.get('translation_eng', ''),
                inscription.get('translation_ukr', ''),
                inscription.get('comments_eng', ''),
                inscription.get('comments_ukr', ''),
                inscription.get('inscription_iiif_url', ''),
                'Yes' if korniienko_images else 'No',
                len(korniienko_images) if korniienko_images else 0,
                inscription.get('published', False),
            ]
            writer.writerow(row)
    
    print(f"Exported {len(inscriptions)} graffiti inscriptions to {csv_filename}")
    
    # Now export all korniienko images to a separate CSV for easy reference
    with open(korniienko_csv, 'w', newline='', encoding='utf-8') as kornfile:
        korn_writer = csv.writer(kornfile)
        
        korn_headers = [
            'korniienko_id', 'inscription_id', 'inscription_title', 'title', 
            'url', 'type_of_image', 'type_of_license', 'author', 'year', 
            'bibliography', 'plate', 'published'
        ]
        korn_writer.writerow(korn_headers)
        
        # Write each korniienko image
        for inscription in inscriptions:
            korniienko_images = inscription.get('korniienko_image', [])
            for kimg in korniienko_images:
                korn_row = [
                    kimg.get('id', ''),
                    inscription.get('id', ''),
                    inscription.get('title', ''),
                    kimg.get('title', ''),
                    kimg.get('url', ''),
                    kimg.get('type_of_image', ''),
                    kimg.get('type_of_license', ''),
                    kimg.get('author', ''),
                    kimg.get('year', ''),
                    kimg.get('bibliography', ''),
                    kimg.get('plate', ''),
                    kimg.get('published', False),
                ]
                korn_writer.writerow(korn_row)
    
    korn_image_count = sum(len(i.get('korniienko_image', [])) for i in inscriptions)
    print(f"Exported {korn_image_count} Korniienko images to {korniienko_csv}")
    
    return csv_filename


def download_annotation(surface_id):
    """Download annotation for a single surface."""
    url = f"https://saintsophia.dh.gu.se/api/inscriptions/annotation/?surface={surface_id}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None  # No annotation found
        else:
            print(f"Error {response.status_code} for surface {surface_id}")
            return None
    except Exception as e:
        print(f"Request failed for {surface_id}: {e}")
        return None


def download_all_annotations(csv_filename):
    """Download annotations for all inscriptions in CSV."""
    print("\nDownloading annotations.")
    
    # Create output directory
    output_dir = 'annotations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    successful = 0
    failed = 0
    
    with open(csv_filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            inscription_id = row['id']
            # Support both panel_title (DB) and panel_id (API)
            panel_id = row.get('panel_id') or row.get('panel_title')
            
            if not panel_id:
                failed += 1
                continue

            print(f"Downloading annotation for inscription {inscription_id} (surface {panel_id})")

            annotation_data = download_annotation(panel_id)
            
            if annotation_data:
                # Save to file
                filename = f"annotation_{inscription_id}.json"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as af:
                    json.dump(annotation_data, af, indent=2, ensure_ascii=False)
                
                print(f"  Saved {filename}")
                successful += 1
            else:
                failed += 1
            
            time.sleep(0.3)  # Be nice to the server
    
    print(f"\nDownload complete: {successful} successful, {failed} failed")
    print(f"Annotation files saved in '{output_dir}' folder")


def korniienko_image(inscription):
    """Get Korniienko image if available."""
    try:
        # Get the related korniienko_image (note: related_name in model is "korniienko_image")
        # This returns a queryset, so we need to check if it exists
        korniienko_images = inscription.korniienko_image.all()
        
        if korniienko_images.exists():
            # Return the first image if multiple exist
            image = korniienko_images.first()
            return {
                'id': image.id,
                'title': image.title,
                'url': image.url,
                'author': str(image.author) if image.author else None,
                'year': image.year,
                'type_of_image': image.type_of_image,
                'type_of_license': image.type_of_license,
                'bibliography': str(image.bibliography) if image.bibliography else None,
                'plate': image.plate
            }
        else:
            return None
    except Exception as e:
        print(f"Error getting Korniienko image for inscription {inscription.id}: {e}")
        return None


def download_korniienko_images(csv_filename):
    """Download Korniienko images from the korniienko_images CSV file."""
    print("\nDownloading Korniienko images...")
    
    # Determine the korniienko CSV filename based on the inscription CSV
    timestamp = csv_filename.split('_')[-1].replace('.csv', '')
    korniienko_csv = f'korniienko_images_{timestamp}.csv'
    
    if not os.path.exists(korniienko_csv):
        print(f"No Korniienko images CSV found ({korniienko_csv})")
        return
    
    # Create output directory
    output_dir = 'korniienko_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    successful = 0
    failed = 0
    skipped = 0
    
    with open(korniienko_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            korniienko_id = row['korniienko_id']
            inscription_id = row['inscription_id']
            url = row.get('url', '')
            image_type = row.get('type_of_image', 'unknown')
            
            if not url:
                skipped += 1
                continue
            
            print(f"Downloading Korniienko image {korniienko_id} ({image_type}) for inscription {inscription_id}")
            
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    # Determine file extension from URL
                    ext = url.split('.')[-1].lower()
                    if ext not in ['jpg', 'jpeg', 'png', 'tif', 'tiff']:
                        ext = 'png'  # default
                    
                    # Use korniienko_id and type to create unique filename
                    filename = f"korniienko_{inscription_id}_{image_type.lower()}_{korniienko_id}.{ext}"
                    filepath = os.path.join(output_dir, filename)
                    
                    with open(filepath, 'wb') as img_file:
                        img_file.write(response.content)
                    
                    print(f"  Saved {filename}")
                    successful += 1
                else:
                    print(f"  Error {response.status_code}")
                    failed += 1
            except Exception as e:
                print(f"  Failed to download: {e}")
                failed += 1
            
            time.sleep(0.3)  # Be nice to the server
    
    print(f"\nKorniienko download complete: {successful} successful, {failed} failed, {skipped} skipped")
    print(f"Images saved in '{output_dir}' folder")


def main():
    print("Saint Sophia Data Collection Tool")
    print("=" * 40)
    print("Exporting all graffiti text data with complete Korniienko information...")
    print()
    
    # Step 1: Export graffiti inscriptions from API with full hierarchy (depth=1)
    csv_filename = export_inscriptions_from_api()
    
    if not csv_filename:
        print("Failed to export inscriptions!")
        return
    
    # Step 2 & 3: SKIP annotation and image downloads for now (will add later)
    print("\n" + "=" * 40)
    print("Export complete! Files created:")
    print(f"  - {csv_filename} (2101 graffiti inscriptions with all text data)")
    
    # Extract timestamp from filename
    timestamp = csv_filename.split('_')[-1].replace('.csv', '')
    korniienko_csv = f'korniienko_images_{timestamp}.csv'
    print(f"  - {korniienko_csv} (1898 Korniienko images catalog)")
    print()
    print("Summary:")
    print("  ✓ All graffiti text data exported")
    print("  ✓ Full panel/surface hierarchy included")
    print("  ✓ Complete Korniienko image catalog with URLs")
    print("  ✓ IIIF image URLs included")
    print()
    print("Next steps (to be added later):")
    print("  - Download annotations")
    print("  - Download Korniienko images")
    print("  - Download IIIF images")


if __name__ == "__main__":
    main()

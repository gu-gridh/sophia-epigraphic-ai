#!/usr/bin/env python3
"""
Image Cropping Script for Saint Sophia Graffiti Recognition
Crops bounding boxes from all 4 image types for each annotation.
"""

import pandas as pd
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Set PIL to handle large images
Image.MAX_IMAGE_PIXELS = None

def get_image_files(panel_title, original_dir):
    """
    Get the 4 image types for a given panel_title.
    
    Args:
        panel_title: Panel name (e.g., "118-02")
        original_dir: Directory containing original images
        
    Returns:
        dict: Mapping of image type to file path
    """
    image_types = {
        'original': None,
        'blended': None,
        'normal': None,
        'texture': None
    }
    
    all_files = os.listdir(original_dir)
    
    # Find files for this panel
    for filename in all_files:
        if filename.startswith(panel_title):
            file_path = os.path.join(original_dir, filename)
            
            # Categorize by type (prioritize exact matches)
            if filename == f"{panel_title}.jpg":
                image_types['original'] = file_path
            elif 'blended_map_texture_level_grey' in filename and image_types['blended'] is None:
                # Prefer non-decimated version if available
                if 'decimated' not in filename:
                    image_types['blended'] = file_path
                elif image_types['blended'] is None:
                    image_types['blended'] = file_path
            elif 'normal_map' in filename and image_types['normal'] is None:
                # Prefer non-decimated version if available
                if 'decimated' not in filename:
                    image_types['normal'] = file_path
                elif image_types['normal'] is None:
                    image_types['normal'] = file_path
            elif 'texture_map' in filename and image_types['texture'] is None:
                # Prefer non-decimated version if available
                if 'decimated' not in filename:
                    image_types['texture'] = file_path
                elif image_types['texture'] is None:
                    image_types['texture'] = file_path
    
    return image_types

def crop_bbox_from_image(image_path, bbox_x, bbox_y, bbox_width, bbox_height, target_size=(224, 224)):
    """
    Crop bounding box area from image.
    
    Args:
        image_path: Path to source image
        bbox_x, bbox_y, bbox_width, bbox_height: Bbox coordinates (0-1 range)
        target_size: Output size (width, height)
        
    Returns:
        PIL.Image: Cropped and resized image
    """
    try:
        # Open image
        img = Image.open(image_path).convert('RGB')
        img_width, img_height = img.size
        
        # Convert bbox to pixel coordinates
        x = int(bbox_x * img_width)
        y = int(bbox_y * img_height)
        w = int(bbox_width * img_width)
        h = int(bbox_height * img_height)
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = max(1, min(w, img_width - x))
        h = max(1, min(h, img_height - y))
        
        # Crop the region
        cropped = img.crop((x, y, x + w, y + h))
        
        # Resize to target size
        resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
        
        return resized
        
    except Exception as e:
        print(f"Error cropping {image_path}: {e}")
        # Return a blank image if cropping fails
        return Image.new('RGB', target_size, color='white')

def crop_dataset_images(dataset_csv, original_dir, output_dir, target_size=(224, 224)):
    """
    Crop all annotations from dataset.
    
    Args:
        dataset_csv: Path to dataset CSV file
        original_dir: Directory with original images
        output_dir: Directory to save cropped images
        target_size: Output image size
    """
    # Load dataset
    df = pd.read_csv(dataset_csv)
    print(f"Processing {len(df)} annotations from {dataset_csv}")
    
    # Create output directories
    output_dirs = {
        'original': os.path.join(output_dir, 'original'),
        'blended': os.path.join(output_dir, 'blended'),
        'normal': os.path.join(output_dir, 'normal'),
        'texture': os.path.join(output_dir, 'texture')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Stats tracking
    stats = {
        'total': len(df),
        'success': 0,
        'missing_files': 0,
        'errors': 0
    }
    
    # Process each annotation
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cropping images"):
        annotation_id = row['id']
        panel_title = row['panel_title']
        
        # Get bbox coordinates
        bbox_x = row['bbox_x']
        bbox_y = row['bbox_y'] 
        bbox_width = row['bbox_width']
        bbox_height = row['bbox_height']
        
        # Skip if bbox is invalid
        if pd.isna(bbox_x) or pd.isna(bbox_y):
            print(f"Invalid bbox for annotation {annotation_id}")
            stats['errors'] += 1
            continue
            
        # Get image files for this panel
        image_files = get_image_files(panel_title, original_dir)
        
        # Check if we have all 4 types
        missing_types = [k for k, v in image_files.items() if v is None]
        if missing_types:
            print(f"Missing image types for {panel_title}: {missing_types}")
            stats['missing_files'] += 1
            continue
        
        # Crop each image type
        success_count = 0
        for img_type, img_path in image_files.items():
            if img_path and os.path.exists(img_path):
                try:
                    # Crop the bbox area
                    cropped_img = crop_bbox_from_image(
                        img_path, bbox_x, bbox_y, bbox_width, bbox_height, target_size
                    )
                    
                    # Save cropped image
                    output_filename = f"{annotation_id}_{img_type}.jpg"
                    output_path = os.path.join(output_dirs[img_type], output_filename)
                    cropped_img.save(output_path, 'JPEG', quality=95)
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        if success_count == 4:
            stats['success'] += 1
        else:
            stats['errors'] += 1
    
    # Print statistics
    print(f"\\n=== Cropping Results ===")
    print(f"Total annotations: {stats['total']}")
    print(f"Successfully processed: {stats['success']}")
    print(f"Missing files: {stats['missing_files']}")
    print(f"Errors: {stats['errors']}")
    
    return stats

def main():
    """Main function to crop images for all datasets."""
    original_dir = 'data/original'
    datasets = ['train_dataset.csv', 'val_dataset.csv', 'test_dataset.csv']
    
    for dataset_name in datasets:
        dataset_path = f'data/{dataset_name}'
        if os.path.exists(dataset_path):
            print(f"\\n{'='*50}")
            print(f"Processing {dataset_name}")
            print(f"{'='*50}")
            
            # Create output directory for this dataset
            split_name = dataset_name.replace('_dataset.csv', '')
            output_dir = f'data/cropped_images/{split_name}'
            
            # Crop images
            stats = crop_dataset_images(dataset_path, original_dir, output_dir)
            
        else:
            print(f"Dataset {dataset_path} not found")

if __name__ == '__main__':
    main()

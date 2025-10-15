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

def crop_bbox_from_image(image_path, bbox_x, bbox_y, bbox_width, bbox_height, target_size=(224, 224), 
                         save_original_crop=False, min_quality_threshold=300):
    """
    Crop bounding box area from image with high-quality preservation.
    
    Args:
        image_path: Path to source image
        bbox_x, bbox_y, bbox_width, bbox_height: Bbox coordinates (0-1 range)
        target_size: Output size (width, height) - only used if crop is smaller than threshold
        save_original_crop: If True, save at original crop size (no resize)
        min_quality_threshold: Minimum size before resizing (pixels)
        
    Returns:
        PIL.Image: Cropped (and optionally resized) high-quality image
    """
    try:
        # Open image - don't convert to RGB yet to preserve quality
        img = Image.open(image_path)
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
        
        # Convert to RGB after cropping to preserve quality
        cropped = cropped.convert('RGB')
        
        # Decide whether to resize
        if save_original_crop:
            # Keep original crop size for maximum quality
            return cropped
        else:
            # Only resize if crop is too small or if we need uniform sizes
            crop_min_dim = min(w, h)
            
            if crop_min_dim < min_quality_threshold:
                # Crop is small, resize using high-quality resampling
                resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
                return resized
            else:
                # Crop is large enough, keep it at original size for best quality
                # Or resize down with LANCZOS (high quality downsampling)
                if w > target_size[0] or h > target_size[1]:
                    resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
                    return resized
                else:
                    # Keep original crop
                    return cropped
        
        return cropped
        
    except Exception as e:
        print(f"Error cropping {image_path}: {e}")
        # Return a blank image if cropping fails
        return Image.new('RGB', target_size, color='white')

def crop_dataset_images(dataset_csv, original_dir, output_dir, target_size=(224, 224), 
                        save_original_size=False, use_png=True, require_all_types=False):
    """
    Crop all annotations from dataset with high quality preservation.
    
    Args:
        dataset_csv: Path to dataset CSV file
        original_dir: Directory with original images
        output_dir: Directory to save cropped images
        target_size: Output image size (only used if save_original_size=False)
        save_original_size: If True, save crops at their original size (best quality)
        use_png: If True, save as PNG (lossless). If False, use high-quality JPEG.
        require_all_types: If True, skip inscriptions missing any image type. If False, save whatever is available.
    """
    # Load dataset
    df = pd.read_csv(dataset_csv)
    print(f"Processing {len(df)} annotations from {dataset_csv}")
    print(f"Quality settings: save_original_size={save_original_size}, use_png={use_png}")
    print(f"Require all image types: {require_all_types}")
    
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
        'partial_success': 0,
        'missing_files': 0,
        'corrupted_files': 0,
        'errors': 0,
        'total_crop_sizes': [],
        'image_type_counts': {'original': 0, 'blended': 0, 'normal': 0, 'texture': 0}
    }
    
    # Determine file extension
    file_ext = 'png' if use_png else 'jpg'
    
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
            stats['errors'] += 1
            continue
            
        # Get image files for this panel
        image_files = get_image_files(panel_title, original_dir)
        
        # Check if we have all 4 types
        missing_types = [k for k, v in image_files.items() if v is None]
        available_types = [k for k, v in image_files.items() if v is not None]
        
        if missing_types and require_all_types:
            # Skip this inscription if we require all types
            stats['missing_files'] += 1
            continue
        
        if not available_types:
            # No images available at all
            stats['missing_files'] += 1
            continue
        
        # Crop each available image type
        success_count = 0
        error_count = 0
        
        for img_type in available_types:
            img_path = image_files[img_type]
            if img_path and os.path.exists(img_path):
                try:
                    # Crop the bbox area with high quality
                    cropped_img = crop_bbox_from_image(
                        img_path, bbox_x, bbox_y, bbox_width, bbox_height, 
                        target_size=target_size,
                        save_original_crop=save_original_size
                    )
                    
                    # Track crop size (only for successful crops)
                    if img_type == 'original' and cropped_img is not None:
                        stats['total_crop_sizes'].append(cropped_img.size)
                    
                    # Save cropped image with optimal quality
                    output_filename = f"{annotation_id}_{img_type}.{file_ext}"
                    output_path = os.path.join(output_dirs[img_type], output_filename)
                    
                    if use_png:
                        # PNG: lossless compression
                        cropped_img.save(output_path, 'PNG', compress_level=6)
                    else:
                        # JPEG: maximum quality
                        cropped_img.save(output_path, 'JPEG', quality=100, subsampling=0)
                    
                    success_count += 1
                    stats['image_type_counts'][img_type] += 1
                    
                except Exception as e:
                    # File is corrupted or other error
                    error_count += 1
                    if 'broken data stream' in str(e) or 'unrecognized data stream' in str(e):
                        stats['corrupted_files'] += 1
        
        # Update overall stats
        if success_count == 4:
            stats['success'] += 1
        elif success_count > 0:
            stats['partial_success'] += 1
        else:
            stats['errors'] += 1
    
    # Print statistics
    print(f"\n=== Cropping Results ===")
    print(f"Total annotations: {stats['total']}")
    print(f"Fully processed (all 4 types): {stats['success']}")
    print(f"Partially processed (1-3 types): {stats['partial_success']}")
    print(f"Missing all files: {stats['missing_files']}")
    print(f"Corrupted files encountered: {stats['corrupted_files']}")
    print(f"Other errors: {stats['errors']}")
    print(f"\nImages saved per type:")
    for img_type, count in stats['image_type_counts'].items():
        print(f"  {img_type}: {count}")
    
    # Print crop size statistics
    if stats['total_crop_sizes']:
        crop_sizes = np.array(stats['total_crop_sizes'])
        print(f"\nCrop size statistics (original images):")
        print(f"  Average: {crop_sizes.mean(axis=0).astype(int)}")
        print(f"  Min: {crop_sizes.min(axis=0)}")
        print(f"  Max: {crop_sizes.max(axis=0)}")
    
    return stats

def main():
    """Main function to crop images for all datasets with high quality preservation."""
    # Use relative paths from scripts directory
    original_dir = '../data/original'
    # In case if we needed to use iiif images
    # iiif_images_dir = '../data/iiif_images'
    datasets = ['train_dataset.csv', 'val_dataset.csv', 'test_dataset.csv']
    
    # Quality settings
    # Option 1: Save at original crop size (BEST QUALITY, variable sizes)
    # Option 2: Resize to fixed size (UNIFORM SIZE, some quality loss)
    save_original_size = True  # Set to False for uniform 224x224 size
    use_png = True  # Set to False for JPEG (smaller files but lossy)
    
    print("="*60)
    print("HIGH QUALITY IMAGE CROPPING")
    print("="*60)
    print(f"Settings:")
    print(f"  - Save original crop size: {save_original_size}")
    print(f"  - Format: {'PNG (lossless)' if use_png else 'JPEG (quality=100)'}")
    print(f"  - Source: {original_dir}")
    print("="*60)
    
    for dataset_name in datasets:
        dataset_path = f'../data/{dataset_name}'
        if os.path.exists(dataset_path):
            print(f"\n{'='*50}")
            print(f"Processing {dataset_name}")
            print(f"{'='*50}")
            
            # Create output directory for this dataset
            split_name = dataset_name.replace('_dataset.csv', '')
            output_dir = f'../data/cropped_images_hq/{split_name}'  # Use _hq suffix for high-quality
            
            # Crop images with high quality
            stats = crop_dataset_images(
                dataset_path, 
                original_dir, 
                output_dir,
                save_original_size=save_original_size,
                use_png=use_png,
                require_all_types=False  # Save whatever image types are available
            )
            
        else:
            print(f"Dataset {dataset_path} not found")

if __name__ == '__main__':
    main()

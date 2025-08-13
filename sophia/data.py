"""
Data processing and dataset creation for SOPHIA.

This module handles:
1. Loading inscription data and images
2. Processing spatial annotations 
3. Creating training/validation datasets
4. Data augmentation for images and text
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2


class InscriptionDataset(Dataset):
    """
    Dataset for multimodal inscription data.
    
    Combines:
    - Images of inscription surfaces
    - Spatial annotation data (coordinates, shapes)
    - Textual information (transcriptions, metadata)
    """
    
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        annotations_dir: str,
        tokenizer_name: str = "xlm-roberta-base",
        max_text_length: int = 512,
        image_size: tuple = (224, 224),
        augment: bool = True
    ):
        self.data = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_length = max_text_length
        self.image_size = image_size
        
        # Image augmentation pipeline
        if augment:
            self.image_transform = A.Compose([
                A.Resize(*image_size),
                A.HorizontalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.image_transform = A.Compose([
                A.Resize(*image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        image_path = self._get_image_path(row)
        image = self._load_image(image_path)
        
        # Load spatial annotations
        spatial_features = self._load_spatial_annotations(row)
        
        # Process text
        text_features = self._process_text(row)
        
        # Metadata features
        metadata_features = self._process_metadata(row)
        
        return {
            'image': image,
            'spatial_features': spatial_features,
            'text_features': text_features,
            'metadata_features': metadata_features,
            'target_text': row['transcription'],
            'inscription_id': row['id']
        }
    
    def _get_image_path(self, row):
        """Find corresponding image file for inscription."""
        inscription_id = row['id']
        panel_title = row['panel_title']
        
        # Try multiple naming conventions
        possible_names = [
            f"{inscription_id}.jpg",
            f"{inscription_id}.png", 
            f"{panel_title}.jpg",
            f"{panel_title}.png",
            f"inscription_{inscription_id}.jpg"
        ]
        
        for name in possible_names:
            path = os.path.join(self.images_dir, name)
            if os.path.exists(path):
                return path
        
        # Return placeholder if no image found
        return self._create_placeholder_image()
    
    def _load_image(self, image_path):
        """Load and transform image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            transformed = self.image_transform(image=image)
            return transformed['image']
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self._create_placeholder_image()
    
    def _create_placeholder_image(self):
        """Create placeholder image for missing files."""
        placeholder = np.zeros((*self.image_size, 3), dtype=np.uint8)
        transformed = self.image_transform(image=placeholder)
        return transformed['image']
    
    def _load_spatial_annotations(self, row):
        """Load and process spatial annotation data."""
        inscription_id = row['id']
        annotation_file = os.path.join(self.annotations_dir, f"annotation_{inscription_id}.json")
        
        if not os.path.exists(annotation_file):
            return self._empty_spatial_features()
        
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            return self._process_spatial_annotations(annotations)
        except Exception as e:
            print(f"Error loading annotations for {inscription_id}: {e}")
            return self._empty_spatial_features()
    
    def _process_spatial_annotations(self, annotations):
        """Process geometric annotation data into features."""
        if not isinstance(annotations, list):
            annotations = [annotations]
        
        features = {
            'num_annotations': len(annotations),
            'bbox_features': [],
            'geometry_types': [],
            'annotation_properties': []
        }
        
        for ann in annotations:
            # Extract bounding box if available
            if 'geometry' in ann and 'coordinates' in ann['geometry']:
                bbox = self._extract_bbox(ann['geometry'])
                features['bbox_features'].append(bbox)
            
            # Extract geometry type
            geom_type = ann.get('geometry', {}).get('type', 'unknown')
            features['geometry_types'].append(geom_type)
            
            # Extract annotation properties
            props = ann.get('properties', {})
            features['annotation_properties'].append(props)
        
        return self._normalize_spatial_features(features)
    
    def _extract_bbox(self, geometry):
        """Extract bounding box from geometry coordinates."""
        coords = geometry['coordinates']
        if not coords:
            return [0, 0, 0, 0]
        
        # Handle different geometry types
        if geometry['type'] == 'Polygon':
            if coords and len(coords) > 0:
                points = coords[0]  # First ring
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                return [min(xs), min(ys), max(xs), max(ys)]
        elif geometry['type'] == 'Point':
            x, y = coords[:2]
            return [x, y, x, y]
        
        return [0, 0, 0, 0]
    
    def _normalize_spatial_features(self, features):
        """Convert spatial features to tensor format."""
        # Pad and normalize bounding boxes
        max_annotations = 10  # Maximum annotations per inscription
        bbox_tensor = torch.zeros(max_annotations, 4)
        
        for i, bbox in enumerate(features['bbox_features'][:max_annotations]):
            bbox_tensor[i] = torch.tensor(bbox, dtype=torch.float32)
        
        return {
            'num_annotations': torch.tensor(features['num_annotations'], dtype=torch.long),
            'bounding_boxes': bbox_tensor,
            'has_annotations': torch.tensor(len(features['bbox_features']) > 0, dtype=torch.bool)
        }
    
    def _empty_spatial_features(self):
        """Return empty spatial features when no annotations available."""
        return {
            'num_annotations': torch.tensor(0, dtype=torch.long),
            'bounding_boxes': torch.zeros(10, 4),
            'has_annotations': torch.tensor(False, dtype=torch.bool)
        }
    
    def _process_text(self, row):
        """Process textual data using tokenizer."""
        # Combine relevant text fields
        text_parts = []
        
        if pd.notna(row['transcription']):
            text_parts.append(f"Transcription: {row['transcription']}")
        
        if pd.notna(row.get('interpretative_edition', '')):
            text_parts.append(f"Edition: {row['interpretative_edition']}")
        
        if pd.notna(row.get('translation_eng', '')):
            text_parts.append(f"Translation: {row['translation_eng']}")
        
        text = " ".join(text_parts) if text_parts else "No text available"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }
    
    def _process_metadata(self, row):
        """Process metadata features."""
        features = {}
        
        # Categorical features
        categorical_fields = [
            'language', 'writing_system', 'type_of_inscription'
        ]
        
        for field in categorical_fields:
            value = row.get(field, 'unknown')
            features[f'{field}_encoded'] = self._encode_categorical(value, field)
        
        # Numerical features
        numerical_fields = [
            'min_year', 'max_year', 'height', 'width', 'elevation'
        ]
        
        for field in numerical_fields:
            value = row.get(field, 0)
            features[field] = torch.tensor(float(value) if pd.notna(value) else 0.0, dtype=torch.float32)
        
        return features
    
    def _encode_categorical(self, value, field):
        """Simple categorical encoding (can be enhanced with proper vocabularies)."""
        # For now, return hash-based encoding
        if pd.isna(value) or value == 'unknown':
            return torch.tensor(0, dtype=torch.long)
        
        return torch.tensor(hash(str(value)) % 1000, dtype=torch.long)


class ImageAnnotationProcessor:
    """
    Utility class for processing images and annotations.
    """
    
    def __init__(self, images_dir: str, annotations_dir: str):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
    
    def create_image_mapping(self, csv_path: str) -> dict:
        """Create mapping between inscription IDs and image files."""
        data = pd.read_csv(csv_path)
        mapping = {}
        
        for _, row in data.iterrows():
            inscription_id = row['id']
            image_path = self._find_image_file(inscription_id, row.get('panel_title', ''))
            mapping[inscription_id] = image_path
        
        return mapping
    
    def _find_image_file(self, inscription_id: str, panel_title: str) -> str:
        """Find image file for given inscription."""
        possible_names = [
            f"{inscription_id}.jpg",
            f"{inscription_id}.png",
            f"{panel_title}.jpg", 
            f"{panel_title}.png"
        ]
        
        for name in possible_names:
            path = os.path.join(self.images_dir, name)
            if os.path.exists(path):
                return path
        
        return None
    
    def validate_dataset(self, csv_path: str) -> dict:
        """Validate dataset completeness."""
        data = pd.read_csv(csv_path)
        stats = {
            'total_inscriptions': len(data),
            'with_images': 0,
            'with_annotations': 0,
            'with_transcriptions': 0,
            'complete_samples': 0
        }
        
        for _, row in data.iterrows():
            inscription_id = row['id']
            
            # Check image
            has_image = self._find_image_file(inscription_id, row.get('panel_title', '')) is not None
            if has_image:
                stats['with_images'] += 1
            
            # Check annotation
            annotation_file = os.path.join(self.annotations_dir, f"annotation_{inscription_id}.json")
            has_annotation = os.path.exists(annotation_file)
            if has_annotation:
                stats['with_annotations'] += 1
            
            # Check transcription
            has_transcription = pd.notna(row.get('transcription', '')) and row.get('transcription', '') != ''
            if has_transcription:
                stats['with_transcriptions'] += 1
            
            # Complete sample
            if has_image and has_annotation and has_transcription:
                stats['complete_samples'] += 1
        
        return stats


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    images_dir: str,
    annotations_dir: str,
    batch_size: int = 16,
    num_workers: int = 4
) -> tuple:
    """Create training and validation dataloaders."""
    
    train_dataset = InscriptionDataset(
        csv_path=train_csv,
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        augment=True
    )
    
    val_dataset = InscriptionDataset(
        csv_path=val_csv,
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

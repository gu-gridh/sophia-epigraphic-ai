#!/usr/bin/env python3

import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

# Import model architectures
from models.models_multichannel import MultiChannelModel
from models.models_enhanced import EnhancedModel
from models.models_transformer import SophiaTransformerModel

# For tokenization
try:
    from transformers import XLMRobertaTokenizer
except ImportError:
    print(" XLM-RoBERTa tokenizer not available. Using basic character tokenizer.")
    XLMRobertaTokenizer = None

# Increase PIL image size limit
Image.MAX_IMAGE_PIXELS = None


class CharacterTokenizer:
    """Simple character-level tokenizer as fallback."""
    
    def __init__(self, vocab_file=None):
        # Create character vocabulary
        self.char_to_idx = {
            '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3,
        }
        
        # Add common characters (Greek, Latin, Cyrillic, numbers, punctuation)
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        # Greek (including extended)
        chars += "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψωϫ"
        # Standard Cyrillic
        chars += "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        # Ancient/Extended Cyrillic (CRITICAL for Saint Sophia!)
        chars += "ЄІєіѕѣѥѦѧѩѫѯѰѱѲѵѿ҂"  # iotified letters, yus, fita, izhitsa, etc.
        chars += "ꙀꙁꙂꙃꙄꙅꙆꙇꙈꙉꙊꙋꙌꙍꙎꙏꙐꙑꙒꙓꙔꙕꙖꙗꙘꙙꙚꙛꙜꙝꙞꙟꙠꙡꙢꙣ"  # Extended Cyrillic Supplement
        # Armenian
        chars += "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖաբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆ"
        # Numbers and punctuation
        chars += "0123456789०१२३४५६७८९୳၊"  # Latin + Devanagari + Oriya + Myanmar
        chars += ".,;:!?-–—'\"()[]{}/@#$%&*+=<>|\\~`_"
        # Special symbols
        chars += "✠ⰰⰱⰲⰳⰴⰵⰶⰷⰸⰹⰺⰻⰼⰽⰾⰿⱀⱁⱂⱃⱄⱅⱆⱇⱈⱉⱊⱋⱌⱍⱎⱏⱐⱑⱒⱓⱔⱕⱖⱗⱘⱙⱚⱛⱜⱝⱞ"  # Glagolitic + Cross
        # Whitespace
        chars += " \n\t"
        
        for idx, char in enumerate(chars, start=4):
            self.char_to_idx[char] = idx
        
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def __call__(self, text, max_length=128, padding='max_length', truncation=True):
        """Tokenize text into character indices."""
        if isinstance(text, list):
            return {'input_ids': [self._encode_single(t, max_length, padding=='max_length', truncation) for t in text],
                    'attention_mask': [self._create_mask(t, max_length) for t in text]}
        else:
            return {'input_ids': self._encode_single(text, max_length, padding=='max_length', truncation),
                    'attention_mask': self._create_mask(text, max_length)}
    
    def _encode_single(self, text, max_length, padding, truncation):
        """Encode single text."""
        # Add SOS and EOS tokens
        indices = [self.char_to_idx['<SOS>']]
        
        for char in text:
            indices.append(self.char_to_idx.get(char, self.char_to_idx['<UNK>']))
        
        indices.append(self.char_to_idx['<EOS>'])
        
        # Truncate if needed
        if truncation and len(indices) > max_length:
            indices = indices[:max_length-1] + [self.char_to_idx['<EOS>']]
        
        # Pad if needed
        if padding and len(indices) < max_length:
            indices += [self.char_to_idx['<PAD>']] * (max_length - len(indices))
        
        return indices
    
    def _create_mask(self, text, max_length):
        """Create attention mask."""
        length = min(len(text) + 2, max_length)  # +2 for SOS and EOS
        mask = [1] * length + [0] * (max_length - length)
        return mask
    
    def decode(self, indices):
        """Decode indices back to text."""
        chars = []
        for idx in indices:
            if idx in [0, 1, 2]:  # PAD, SOS, EOS
                continue
            chars.append(self.idx_to_char.get(idx, '<UNK>'))
        return ''.join(chars)


class SophiaMultiModalDataset(Dataset):
    """
    Multi-modal dataset for Saint Sophia graffiti recognition.
    
    Supports flexible data loading:
    - RTI images (12 channels: 4 types × 3 RGB)
    - Korniienko photo (3 RGB channels)
    - Korniienko drawing (3 RGB channels)
    - Language and writing system labels
    
    Gracefully handles missing modalities.
    """
    
    def __init__(self, csv_file, data_dir, tokenizer, max_length=128,
                 use_rti=True, use_korniienko=True, model_type='enhanced',
                 image_size=224, augment=False, split='train'):
        """
        Args:
            csv_file: Path to comprehensive dataset CSV
            data_dir: Base directory containing all data
            tokenizer: Text tokenizer
            max_length: Maximum text sequence length
            use_rti: Whether to load RTI images
            use_korniienko: Whether to load Korniienko images
            model_type: 'multichannel', 'enhanced', or 'transformer'
            image_size: Target image size
            augment: Whether to apply data augmentation
            split: Dataset split ('train', 'val', or 'test')
        """
        self.df = pd.read_csv(csv_file)
        self.data_dir = Path(data_dir)
        self.rti_dir = self.data_dir  # RTI images paths are relative to data/ directory
        self.korniienko_dir = self.data_dir  # Korniienko images are in data/ directory
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_rti = use_rti
        self.use_korniienko = use_korniienko
        self.model_type = model_type
        self.image_size = image_size
        self.split = split
        
        # Define transforms
        self.transform = self._create_transforms(augment)
        
        # Create mappings
        self._create_mappings()
        
        # Filter valid samples
        self._filter_valid_samples()
        
        print(f"✓ Dataset loaded: {len(self.df)} samples")
        print(f"  RTI: {'✓' if use_rti else '✗'}, Korniienko: {'✓' if use_korniienko else '✗'}")
        
    def _create_transforms(self, augment):
        """Create image transformations."""
        if augment:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Add blur
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
    
    def _create_mappings(self):
        """Create language and writing system mappings."""
        # Language mapping
        languages = self.df['language_name'].fillna('unknown').unique()
        self.language_to_idx = {lang: idx for idx, lang in enumerate(sorted(languages))}
        self.idx_to_language = {idx: lang for lang, idx in self.language_to_idx.items()}
        
        # Writing system mapping
        writing_systems = self.df['writing_system_name'].fillna('unknown').unique()
        self.ws_to_idx = {ws: idx for idx, ws in enumerate(sorted(writing_systems))}
        self.idx_to_ws = {idx: ws for ws, idx in self.ws_to_idx.items()}
        
        self.num_languages = len(self.language_to_idx)
        self.num_writing_systems = len(self.ws_to_idx)
    
    def _filter_valid_samples(self):
        """Filter for samples with valid transcriptions and available images."""
        valid_indices = []
        
        for idx, row in self.df.iterrows():
            # Check transcription (allow single characters - valid ancient letters)
            if pd.isna(row.get('transcription_clean')) or len(str(row.get('transcription_clean'))) < 1:
                continue
            
            # Check RTI images if required
            if self.use_rti and not self._check_rti_images(row):
                continue
            
            # Check Korniienko images if required
            if self.use_korniienko and not self._check_korniienko_images(row):
                continue
            
            valid_indices.append(idx)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
    
    def _check_rti_images(self, row):
        """Check if RTI images or IIIF crops exist."""
        # First try to use CSV columns (preferred method)
        for img_type in ['original', 'blended', 'normal', 'texture']:
            col_name = f"{img_type}_image"
            if col_name in row.index and pd.notna(row[col_name]):
                img_path = self.data_dir / str(row[col_name])
                if img_path.exists():
                    return True
        
        # Fallback: try old structure with isialy_id
        isialy_id = row.get('isialy_id', row.get('id', ''))
        if not pd.isna(isialy_id):
            rti_dir = self.data_dir / 'cropped_images_hq' / self.split
            for img_type in ['original', 'blended', 'normal', 'texture']:
                img_path = rti_dir / img_type / f"{isialy_id}_{img_type}.png"
                if img_path.exists():
                    return True
        
        # Check IIIF crop as fallback
        iiif_crop_path_str = row.get('iiif_crop', '')
        if not pd.isna(iiif_crop_path_str) and str(iiif_crop_path_str).strip():
            iiif_path = self.data_dir / str(iiif_crop_path_str)
            if iiif_path.exists():
                return True
        
        return False
    
    def _check_korniienko_images(self, row):
        """Check if Korniienko images or IIIF crops exist."""
        # Check if CSV has Korniienko paths
        photo_path_str = row.get('korniienko_photo', '')
        drawing_path_str = row.get('korniienko_drawing', '')
        iiif_crop_path_str = row.get('iiif_crop', '')
        
        # Check if at least one image exists
        if not pd.isna(photo_path_str) and str(photo_path_str).strip():
            photo_path = self.data_dir / str(photo_path_str)
            if photo_path.exists():
                return True
        
        if not pd.isna(drawing_path_str) and str(drawing_path_str).strip():
            drawing_path = self.data_dir / str(drawing_path_str)
            if drawing_path.exists():
                return True
        
        # Check IIIF crop as fallback
        if not pd.isna(iiif_crop_path_str) and str(iiif_crop_path_str).strip():
            iiif_path = self.data_dir / str(iiif_crop_path_str)
            if iiif_path.exists():
                return True
        
        return False
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        isialy_id = row.get('isialy_id', row.get('id', ''))
        
        # Load transcription
        transcription = str(row.get('transcription_clean', ''))
        
        # Tokenize
        encoded = self.tokenizer(
            transcription,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        # Load RTI images (4 types × 3 RGB = 12 channels)
        rti_images = []
        rti_types = ['original', 'blended', 'normal', 'texture']
        
        # Check if we have IIIF crop as fallback
        iiif_crop_path = None
        if 'iiif_crop' in row.index and pd.notna(row['iiif_crop']) and str(row['iiif_crop']).strip():
            potential_iiif = os.path.join(self.data_dir, str(row['iiif_crop']))
            if os.path.exists(potential_iiif):
                iiif_crop_path = potential_iiif
        
        for rti_type in rti_types:
            # Try to get path from CSV column (e.g., 'original_image', 'blended_image')
            col_name = f"{rti_type}_image"
            img_path = None
            
            if col_name in row.index and pd.notna(row[col_name]) and str(row[col_name]).strip():
                # CSV has relative path, prepend data_dir
                img_path = os.path.join(self.data_dir, str(row[col_name]))
                if not os.path.exists(img_path):
                    img_path = None
            
            if img_path is None:
                # Fallback: try old structure with isialy_id
                fallback_path = os.path.join(self.rti_dir, str(isialy_id), f"{rti_type}.jpg")
                if os.path.exists(fallback_path):
                    img_path = fallback_path
            
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                rti_images.append(img)
            elif iiif_crop_path:
                # Use IIIF crop as fallback for missing RTI images
                img = Image.open(iiif_crop_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                rti_images.append(img)
            else:
                # Create blank image if missing and no IIIF fallback
                blank = torch.zeros(3, self.image_size, self.image_size)
                rti_images.append(blank)
        
        # Stack RTI images: [4, 3, H, W] -> [12, H, W]
        rti_tensor = torch.cat(rti_images, dim=0)
        
        # Load Korniienko PHOTO (photograph) or IIIF crop as fallback
        korniienko_photo_tensor = None
        korniienko_photo_path = None
        
        # Try different possible column names and path formats
        possible_photo_cols = ['korniienko_photo', 'korniienko_photo_path', 'korniienko_image']
        for col in possible_photo_cols:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                # Could be full path or just filename
                photo_value = str(row[col])
                
                # Try different path constructions
                possible_paths = [
                    photo_value,  # Full path from CSV
                    os.path.join(self.korniienko_dir, photo_value),  # korniienko_images/filename
                    os.path.join(self.korniienko_dir, f"{isialy_id}_photo.jpg"),  # Standard naming
                    os.path.join(self.korniienko_dir, f"{isialy_id}.jpg"),  # Simple naming
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        korniienko_photo_path = path
                        break
                
                if korniienko_photo_path:
                    break
        
        # Try IIIF crop as fallback if no Korniienko photo found
        if not korniienko_photo_path and 'iiif_crop' in row and pd.notna(row['iiif_crop']) and str(row['iiif_crop']).strip():
            iiif_path = os.path.join(self.data_dir, str(row['iiif_crop']))
            if os.path.exists(iiif_path):
                korniienko_photo_path = iiif_path
        
        if korniienko_photo_path and os.path.exists(korniienko_photo_path):
            img = Image.open(korniienko_photo_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            korniienko_photo_tensor = img
        else:
            # Create blank if missing
            korniienko_photo_tensor = torch.zeros(3, self.image_size, self.image_size)
        
        # Load Korniienko DRAWING (tracing/drawing)
        korniienko_drawing_tensor = None
        korniienko_drawing_path = None
        
        possible_drawing_cols = ['korniienko_drawing', 'korniienko_drawing_path']
        for col in possible_drawing_cols:
            if col in row and pd.notna(row[col]):
                drawing_value = str(row[col])
                
                possible_paths = [
                    drawing_value,
                    os.path.join(self.korniienko_dir, drawing_value),
                    os.path.join(self.korniienko_dir, f"{isialy_id}_drawing.jpg"),
                    os.path.join(self.korniienko_dir, f"{isialy_id}_tracing.jpg"),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        korniienko_drawing_path = path
                        break
                
                if korniienko_drawing_path:
                    break
        
        if korniienko_drawing_path and os.path.exists(korniienko_drawing_path):
            img = Image.open(korniienko_drawing_path).convert('RGB')  # Convert to RGB even if grayscale
            if self.transform:
                img = self.transform(img)
            korniienko_drawing_tensor = img
        else:
            # Create blank if missing
            korniienko_drawing_tensor = torch.zeros(3, self.image_size, self.image_size)
        
        # Prepare output
        return {
            'rti_images': rti_tensor,  # [12, H, W]
            'korniienko_photo': korniienko_photo_tensor,  # [3, H, W]
            'korniienko_drawing': korniienko_drawing_tensor,  # [3, H, W]
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'language': self.language_to_idx.get(row.get('language_name', 'unknown'), 0),
            'writing_system': self.ws_to_idx.get(row.get('writing_system_name', 'unknown'), 0),
            'transcription': transcription,
            'isialy_id': isialy_id
        }
    
    def _load_rti_images(self, isialy_id):
        """Load and stack RTI images (4 types × 3 RGB = 12 channels)."""
        rti_dir = self.data_dir / 'data' / 'cropped_images_hq' / self.split
        rti_types = ['original', 'blended', 'normal', 'texture']
        
        channels = []
        for img_type in rti_types:
            img_path = rti_dir / img_type / f"{isialy_id}_{img_type}.png"
            
            if img_path.exists():
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img)  # [3, H, W]
            else:
                # Create blank image if missing
                img_tensor = torch.zeros(3, self.image_size, self.image_size)
            
            channels.append(img_tensor)
        
        # Stack to create [12, H, W]
        rti_tensor = torch.cat(channels, dim=0)
        return rti_tensor
    
    def _load_korniienko_images(self, row):
        """Load Korniienko photo and drawing from CSV paths."""
        photo_tensor = None
        drawing_tensor = None
        
        # Load photo if path exists in CSV
        photo_path_str = row.get('korniienko_photo', '')
        if not pd.isna(photo_path_str) and photo_path_str:
            photo_path = self.data_dir / 'data' / photo_path_str
            if photo_path.exists():
                try:
                    photo = Image.open(photo_path).convert('RGB')
                    photo_tensor = self.transform(photo)
                except Exception as e:
                    print(f"Warning: Failed to load photo {photo_path}: {e}")
                    photo_tensor = None
        
        # Load drawing if path exists in CSV
        drawing_path_str = row.get('korniienko_drawing', '')
        if not pd.isna(drawing_path_str) and drawing_path_str:
            drawing_path = self.data_dir / 'data' / drawing_path_str
            if drawing_path.exists():
                try:
                    drawing = Image.open(drawing_path).convert('RGB')
                    drawing_tensor = self.transform(drawing)
                except Exception as e:
                    print(f"Warning: Failed to load drawing {drawing_path}: {e}")
                    drawing_tensor = None
        
        return photo_tensor, drawing_tensor


def collate_fn(batch):
    """Custom collate function to handle None values in Korniienko images."""
    # Stack regular tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    languages = torch.tensor([item['language'] for item in batch])
    writing_systems = torch.tensor([item['writing_system'] for item in batch])
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'language': languages,
        'writing_system': writing_systems,
        'transcription': [item['transcription'] for item in batch],
        'isialy_id': [item['isialy_id'] for item in batch]
    }
    
    # Handle RTI images
    if 'rti_images' in batch[0]:
        result['rti_images'] = torch.stack([item['rti_images'] for item in batch])
    
    # Handle Korniienko photo (replace None with zeros to maintain batch size)
    if 'korniienko_photo' in batch[0]:
        photos = []
        for item in batch:
            if item['korniienko_photo'] is not None:
                photos.append(item['korniienko_photo'])
            else:
                # Create zero tensor with same shape as others
                photos.append(torch.zeros(3, 224, 224))  # Assuming 224x224 images
        result['korniienko_photo'] = torch.stack(photos)
    
    # Handle Korniienko drawing (replace None with zeros to maintain batch size)
    if 'korniienko_drawing' in batch[0]:
        drawings = []
        for item in batch:
            if item['korniienko_drawing'] is not None:
                drawings.append(item['korniienko_drawing'])
            else:
                # Create zero tensor with same shape as others
                drawings.append(torch.zeros(3, 224, 224))  # Assuming 224x224 images
        result['korniienko_drawing'] = torch.stack(drawings)
    
    return result


def create_model(model_type, vocab_size, num_languages, num_writing_systems, 
                 use_korniienko=True):
    """
    Create model instance based on type.
    
    Args:
        model_type: 'multichannel', 'enhanced', or 'transformer'
        vocab_size: Size of vocabulary
        num_languages: Number of language classes
        num_writing_systems: Number of writing system classes
        use_korniienko: Whether to enable Korniienko support
    
    Returns:
        model: PyTorch model instance
    """
    if model_type == 'multichannel':
        model = MultiChannelModel(
            vocab_size=vocab_size,
            vision_dim=512,
            hidden_dim=512,
            num_layers=8,
            num_languages=num_languages,
            num_writing_systems=num_writing_systems,
            use_korniienko=use_korniienko
        )
        
    elif model_type == 'enhanced':
        model = EnhancedModel(
            vocab_size=vocab_size,
            vision_dim=512,
            hidden_dim=512,
            num_layers=8,
            num_languages=num_languages,
            num_writing_systems=num_writing_systems,
            use_korniienko=use_korniienko
        )
        
    elif model_type == 'transformer':
        model = SophiaTransformerModel(
            vocab_size=vocab_size,
            embed_dim=512,
            num_torso_layers=6,
            num_decoder_layers=3,
            num_heads=8,
            num_languages=num_languages,
            num_writing_systems=num_writing_systems,
            max_text_length=128
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_epoch(model, dataloader, optimizer, criterion, device, model_type, 
                use_rti=True, use_korniienko=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Move to device
        rti_images = batch['rti_images'].to(device)
        korniienko_photo = batch['korniienko_photo'].to(device)
        korniienko_drawing = batch['korniienko_drawing'].to(device)
        text_indices = batch['input_ids'].to(device)  # Use input_ids from collate_fn
        text_mask = batch['attention_mask'].to(device)  # Use attention_mask from collate_fn
        
        # Prepare images
        kwargs = {}
        if use_rti and 'rti_images' in batch:
            kwargs['images'] = batch['rti_images'].to(device)
        
        if use_korniienko:
            if batch.get('korniienko_photo') is not None:
                kwargs['korniienko_photo'] = batch['korniienko_photo'].to(device)
            
            if batch.get('korniienko_drawing') is not None:
                kwargs['korniienko_drawing'] = batch['korniienko_drawing'].to(device)
        
        # Forward pass - different models have different signatures
        if model_type == 'transformer':
            # Transformer uses different parameter names and returns a dict
            outputs = model(
                rti_images=rti_images,
                korniienko_photo=korniienko_photo,
                korniienko_drawing=korniienko_drawing,
                text_indices=text_indices,
                text_mask=text_mask,
                training=True
            )
            logits = outputs['transcription_logits']
        else:
            # Enhanced and MultiChannel models
            logits = model(
                input_ids=text_indices,
                attention_mask=text_mask,
                images=rti_images,
                languages=batch.get('language'),
                writing_systems=batch.get('writing_system'),
                korniienko_photo=korniienko_photo,
                korniienko_drawing=korniienko_drawing
            )
        
        # Compute loss
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            text_indices.view(-1)
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, model_type,
             use_rti=True, use_korniienko=True):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            languages = batch['language'].to(device)
            writing_systems = batch['writing_system'].to(device)
            
            # Prepare images
            kwargs = {}
            if use_rti and 'rti_images' in batch:
                kwargs['images'] = batch['rti_images'].to(device)
            
            if use_korniienko:
                if batch.get('korniienko_photo') is not None:
                    kwargs['korniienko_photo'] = batch['korniienko_photo'].to(device)
                
                if batch.get('korniienko_drawing') is not None:
                    kwargs['korniienko_drawing'] = batch['korniienko_drawing'].to(device)
            
            # Forward pass
            if model_type == 'transformer':
                kwargs['text_indices'] = input_ids
                kwargs['text_mask'] = attention_mask
                if 'images' in kwargs:
                    kwargs['rti_images'] = kwargs.pop('images')
                outputs = model(**kwargs)
                logits = outputs['transcription_logits']
            else:
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    languages=languages,
                    writing_systems=writing_systems,
                    **kwargs
                )
            
            # Compute loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1)
            )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(
        description='Unified Training for Saint Sophia Graffiti Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-Channel model with full multi-modal training
  python train.py --model multichannel --epochs 15 --batch_size 6
  
  # Enhanced model with Korniienko-only (Phase 1)
  python train.py --model enhanced --use_korniienko --no_rti --epochs 10
  
  # Transformer model with all modalities
  python train.py --model transformer --epochs 20 --batch_size 4 --lr 5e-5
        """
    )
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['multichannel', 'enhanced', 'transformer'],
                        help='Model architecture to train')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, 
                        default='/home/aram/GRIDH/Saint_Sophia/sophia-epigraphic-ai',
                        help='Base data directory')
    parser.add_argument('--train_csv', type=str, default='train_comprehensive.csv',
                        help='Training CSV file (relative to data_dir/data/)')
    parser.add_argument('--val_csv', type=str, default='val_comprehensive.csv',
                        help='Validation CSV file (relative to data_dir/data/)')
    
    # Modality selection
    parser.add_argument('--use_rti', action='store_true', default=False,
                        help='Use RTI images')
    parser.add_argument('--no_rti', action='store_true', default=False,
                        help='Disable RTI images')
    parser.add_argument('--use_korniienko', action='store_true', default=False,
                        help='Use Korniienko images')
    parser.add_argument('--no_korniienko', action='store_true', default=False,
                        help='Disable Korniienko images')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum text sequence length')
    
    # Tokenizer selection
    parser.add_argument('--tokenizer', type=str, default='xlm',
                        choices=['xlm', 'character'],
                        help='Tokenizer type: xlm (XLM-RoBERTa) or character (character-level)')
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Learning rate warmup steps')
    
    # Checkpoint and logging
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (batches)')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Handle modality flags (default to both enabled if neither specified)
    if not args.use_rti and not args.no_rti and not args.use_korniienko and not args.no_korniienko:
        args.use_rti = True
        args.use_korniienko = True
    elif args.use_rti:
        args.use_rti = True
    elif args.no_rti:
        args.use_rti = False
    else:
        args.use_rti = True
    
    if args.use_korniienko:
        args.use_korniienko = True
    elif args.no_korniienko:
        args.use_korniienko = False
    else:
        args.use_korniienko = True
    
    # Validate modality selection
    if not args.use_rti and not args.use_korniienko:
        raise ValueError("At least one modality (RTI or Korniienko) must be enabled!")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Print configuration
    print("=" * 70)
    print("SAINT SOPHIA GRAFFITI RECOGNITION - UNIFIED TRAINING")
    print("=" * 70)
    print(f"\n Model: {args.model.upper()}")
    print(f"  Modalities:")
    print(f"   RTI Images: {'✓' if args.use_rti else '✗'}")
    print(f"   Korniienko: {'✓' if args.use_korniienko else '✗'}")
    print(f"\n  Hyperparameters:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Image Size: {args.image_size}×{args.image_size}")
    print(f"   Max Text Length: {args.max_length}")
    print(f"\n Data:")
    print(f"   Base Dir: {args.data_dir}")
    print(f"   Train CSV: {args.train_csv}")
    print(f"   Val CSV: {args.val_csv}")
    print(f"\n Device: {args.device}")
    print("=" * 70)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.model / datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n Checkpoints: {checkpoint_dir}")
    
    # Save configuration
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ Config saved: {config_path}")
    
    # Initialize tokenizer
    print("\n Initializing tokenizer...")
    if args.tokenizer == 'character':
        tokenizer = CharacterTokenizer()
        vocab_size = tokenizer.vocab_size
        print(f"✓ Character-level tokenizer loaded (vocab size: {vocab_size})")
        print(f"  ✓ Preserves all ancient Cyrillic characters (ꙗ, ꙅ, ѧ, etc.)")
    else:  # xlm
        if XLMRobertaTokenizer is not None:
            try:
                tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
                vocab_size = tokenizer.vocab_size
                print(f"✓ XLM-RoBERTa tokenizer loaded (vocab size: {vocab_size})")
                print(f"  Warning: May corrupt ancient Cyrillic characters")
            except:
                print(" XLM-RoBERTa failed, falling back to character tokenizer")
                tokenizer = CharacterTokenizer()
                vocab_size = tokenizer.vocab_size
        else:
            print(" XLM-RoBERTa not available, using character tokenizer")
            tokenizer = CharacterTokenizer()
            vocab_size = tokenizer.vocab_size
    
    # Create datasets
    print("\n Loading datasets...")
    
    # Handle CSV paths - if they contain 'data/', remove it since data_dir already points to data/
    train_csv_path = args.train_csv.replace('data/', '') if 'data/' in args.train_csv else args.train_csv
    val_csv_path = args.val_csv.replace('data/', '') if 'data/' in args.val_csv else args.val_csv
    
    train_dataset = SophiaMultiModalDataset(
        csv_file=Path(args.data_dir) / train_csv_path,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_rti=args.use_rti,
        use_korniienko=args.use_korniienko,
        model_type=args.model,
        image_size=args.image_size,
        augment=True,
        split='train'
    )
    
    val_dataset = SophiaMultiModalDataset(
        csv_file=Path(args.data_dir) / val_csv_path,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_rti=args.use_rti,
        use_korniienko=args.use_korniienko,
        model_type=args.model,
        image_size=args.image_size,
        augment=False,
        split='val'
    )
    
    print(f"\n✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create model
    print("\n  Creating model...")
    model = create_model(
        model_type=args.model,
        vocab_size=vocab_size,
        num_languages=train_dataset.num_languages,
        num_writing_systems=train_dataset.num_writing_systems,
        use_korniienko=args.use_korniienko
    )
    
    model = model.to(args.device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created: {args.model}")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    
    # Create optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\n Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✓ Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n Epoch {epoch + 1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, args.device,
            args.model, args.use_rti, args.use_korniienko
        )
        
        # Validate
        val_loss = validate(
            model, val_loader, criterion, args.device,
            args.model, args.use_rti, args.use_korniienko
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics
        print(f"\n Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   LR:         {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or val_loss < best_val_loss:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': vars(args),
                'num_languages': train_dataset.num_languages,
                'num_writing_systems': train_dataset.num_writing_systems,
                'language_to_idx': train_dataset.language_to_idx,
                'idx_to_language': train_dataset.idx_to_language,
                'ws_to_idx': train_dataset.ws_to_idx,
                'idx_to_ws': train_dataset.idx_to_ws,
                'vocab_size': vocab_size
            }, checkpoint_path)
            
            print(f" Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = checkpoint_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'config': vars(args),
                    'num_languages': train_dataset.num_languages,
                    'num_writing_systems': train_dataset.num_writing_systems,
                    'language_to_idx': train_dataset.language_to_idx,
                    'idx_to_language': train_dataset.idx_to_language,
                    'ws_to_idx': train_dataset.ws_to_idx,
                    'idx_to_ws': train_dataset.idx_to_ws,
                    'vocab_size': vocab_size
                }, best_model_path)
                print(f" Best model saved: {best_model_path} (val_loss: {val_loss:.4f})")
    
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n Best Validation Loss: {best_val_loss:.4f}")
    print(f" Checkpoints saved in: {checkpoint_dir}")
    
    # Save final model
    final_model_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'config': vars(args),
        'num_languages': train_dataset.num_languages,
        'num_writing_systems': train_dataset.num_writing_systems,
        'language_to_idx': train_dataset.language_to_idx,
        'idx_to_language': train_dataset.idx_to_language,
        'ws_to_idx': train_dataset.ws_to_idx,
        'idx_to_ws': train_dataset.idx_to_ws,
        'vocab_size': vocab_size
    }, final_model_path)
    print(f" Final model saved: {final_model_path}")
    
    print("\n Done!")


if __name__ == '__main__':
    main()

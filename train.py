#!/usr/bin/env python3#!/usr/bin/env python3

""""""

Unified Training Script for Saint Sophia Graffiti RecognitionTraining script for SOPHIA model.

============================================================="""



Supports three model architectures with multi-modal inputs:import os

1. Multi-Channel CNN (70M params) - 12-channel RTI + Korniienkoimport sys

2. Enhanced CNN (58M params) - Deep ResNet + Attention + Korniienko  import argparse

3. Transformer (51M params) - Native multi-modal with task headsimport json

from pathlib import Path

Features:

- Multi-modal training (RTI + Korniienko photo + drawing)# Add project root to path

- Language and writing system conditioningproject_root = Path(__file__).parent.parent

- Flexible data requirements (any subset of modalities)sys.path.append(str(project_root))

- Phase-based training support

- Automatic model selection and configurationfrom sophia.training import SophiaTrainer

from sophia.data import create_dataloaders

Usage:

    # Multi-Channel model

    python train.py --model multichannel --epochs 15 --batch_size 6def main():

        parser = argparse.ArgumentParser(description='Train SOPHIA model')

    # Enhanced model    parser.add_argument('--config', required=True, help='Path to configuration file')

    python train.py --model enhanced --epochs 12 --batch_size 8    parser.add_argument('--data_dir', default='./data', help='Data directory')

        parser.add_argument('--resume', help='Path to checkpoint to resume from')

    # Transformer model    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    python train.py --model transformer --epochs 20 --batch_size 4    

        args = parser.parse_args()

    # Korniienko-only training (Phase 1)    

    python train.py --model enhanced --use_korniienko --no_rti --epochs 10    # Load configuration

        with open(args.config, 'r') as f:

    # RTI-only training (Phase 2)        config = json.load(f)

    python train.py --model enhanced --use_rti --no_korniienko --epochs 10    

        # Override data paths if provided

    # Full multi-modal training (Phase 3)    if args.data_dir:

    python train.py --model enhanced --use_korniienko --use_rti --epochs 15        config['data']['train_csv'] = os.path.join(args.data_dir, 'train_dataset.csv')

"""        config['data']['val_csv'] = os.path.join(args.data_dir, 'val_dataset.csv')

        config['data']['images_dir'] = os.path.join(args.data_dir, 'images')

import os        config['data']['annotations_dir'] = os.path.join(args.data_dir, 'annotations')

import sys    

import argparse    # Debug mode

import pandas as pd    if args.debug:

import numpy as np        config['training']['num_epochs'] = 2

import torch        config['training']['batch_size'] = 2

import torch.nn as nn        config['use_wandb'] = False

import torch.optim as optim        print("DEBUG MODE: Reduced epochs and batch size")

from torch.utils.data import Dataset, DataLoader    

from PIL import Image    print("Configuration:")

import torchvision.transforms as transforms    print(json.dumps(config, indent=2))

from tqdm import tqdm    

import json    # Check data availability

from datetime import datetime    train_csv = config['data']['train_csv']

from pathlib import Path    val_csv = config['data']['val_csv']

import warnings    images_dir = config['data']['images_dir']

warnings.filterwarnings('ignore')    annotations_dir = config['data']['annotations_dir']

    

# Import model architectures    if not all(os.path.exists(p) for p in [train_csv, val_csv, images_dir, annotations_dir]):

from models.models_multichannel import MultiChannelModel        print("ERROR: Some data files are missing!")

from models.models_enhanced import EnhancedModel        print(f"Train CSV: {train_csv} - {'✓' if os.path.exists(train_csv) else '✗'}")

from models.models_transformer import SophiaTransformerModel        print(f"Val CSV: {val_csv} - {'✓' if os.path.exists(val_csv) else '✗'}")

        print(f"Images dir: {images_dir} - {'✓' if os.path.exists(images_dir) else '✗'}")

# For tokenization        print(f"Annotations dir: {annotations_dir} - {'✓' if os.path.exists(annotations_dir) else '✗'}")

try:        return 1

    from transformers import XLMRobertaTokenizer    

except ImportError:    # Create data loaders

    print("⚠️  XLM-RoBERTa tokenizer not available. Using basic character tokenizer.")    print("Creating data loaders...")

    XLMRobertaTokenizer = None    train_loader, val_loader = create_dataloaders(

        train_csv=train_csv,

# Increase PIL image size limit        val_csv=val_csv,

Image.MAX_IMAGE_PIXELS = None        images_dir=images_dir,

        annotations_dir=annotations_dir,

        batch_size=config['training']['batch_size'],

class CharacterTokenizer:        num_workers=config['training']['num_workers']

    """Simple character-level tokenizer as fallback."""    )

        

    def __init__(self, vocab_file=None):    print(f"Train loader: {len(train_loader)} batches")

        # Create character vocabulary    print(f"Validation loader: {len(val_loader)} batches")

        self.char_to_idx = {    

            '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3,    # Create trainer

        }    print("Initializing trainer...")

            trainer = SophiaTrainer(config)

        # Add common characters (Greek, Latin, Cyrillic, numbers, punctuation)    

        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"    # Resume from checkpoint if provided

        chars += "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"    if args.resume:

        chars += "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"        print(f"Resuming from checkpoint: {args.resume}")

        chars += "0123456789.,;:!?-–—'\"()[]{}/@#$%&*+=<>|\\~`"        trainer.load_checkpoint(args.resume)

        chars += " \n\t"    

            # Start training

        for idx, char in enumerate(chars, start=4):    print("Starting training...")

            self.char_to_idx[char] = idx    trainer.train(

                train_loader=train_loader,

        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}        val_loader=val_loader,

        self.vocab_size = len(self.char_to_idx)        num_epochs=config['training']['num_epochs']

            )

    def __call__(self, text, max_length=128, padding='max_length', truncation=True):    

        """Tokenize text into character indices."""    print("Training completed!")

        if isinstance(text, list):    return 0

            return {'input_ids': [self._encode_single(t, max_length, padding=='max_length', truncation) for t in text],

                    'attention_mask': [self._create_mask(t, max_length) for t in text]}

        else:if __name__ == "__main__":

            return {'input_ids': self._encode_single(text, max_length, padding=='max_length', truncation),    sys.exit(main())

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
                 image_size=224, augment=False):
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
        """
        self.df = pd.read_csv(csv_file)
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_rti = use_rti
        self.use_korniienko = use_korniienko
        self.model_type = model_type
        self.image_size = image_size
        
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
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
        languages = self.df['language'].fillna('unknown').unique()
        self.language_to_idx = {lang: idx for idx, lang in enumerate(sorted(languages))}
        self.idx_to_language = {idx: lang for lang, idx in self.language_to_idx.items()}
        
        # Writing system mapping
        writing_systems = self.df['writing_system'].fillna('unknown').unique()
        self.ws_to_idx = {ws: idx for idx, ws in enumerate(sorted(writing_systems))}
        self.idx_to_ws = {idx: ws for ws, idx in self.ws_to_idx.items()}
        
        self.num_languages = len(self.language_to_idx)
        self.num_writing_systems = len(self.ws_to_idx)
    
    def _filter_valid_samples(self):
        """Filter for samples with valid transcriptions and available images."""
        valid_indices = []
        
        for idx, row in self.df.iterrows():
            # Check transcription
            if pd.isna(row.get('clean_transcription')) or len(str(row.get('clean_transcription'))) < 2:
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
        """Check if RTI images exist."""
        isialy_id = row.get('isialy_id', row.get('id', ''))
        if pd.isna(isialy_id):
            return False
        
        # Check for at least one RTI type
        rti_dir = self.data_dir / 'cropped_images_hq' / 'train'
        for img_type in ['original', 'blended', 'normal', 'texture']:
            img_path = rti_dir / img_type / f"{isialy_id}.png"
            if img_path.exists():
                return True
        return False
    
    def _check_korniienko_images(self, row):
        """Check if Korniienko images exist."""
        isialy_id = row.get('isialy_id', row.get('id', ''))
        if pd.isna(isialy_id):
            return False
        
        korniienko_dir = self.data_dir / 'korniienko_images' / isialy_id
        photo_path = korniienko_dir / 'photo.jpg'
        drawing_path = korniienko_dir / 'drawing.jpg'
        
        return photo_path.exists() or drawing_path.exists()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        isialy_id = row.get('isialy_id', row.get('id', ''))
        
        # Load transcription
        transcription = str(row.get('clean_transcription', ''))
        
        # Tokenize
        encoded = self.tokenizer(
            transcription,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        # Load images
        item = {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'transcription': transcription,
            'isialy_id': isialy_id
        }
        
        # Load RTI images (12 channels)
        if self.use_rti:
            rti_images = self._load_rti_images(isialy_id)
            item['rti_images'] = rti_images
        
        # Load Korniienko images
        if self.use_korniienko:
            photo, drawing = self._load_korniienko_images(isialy_id)
            item['korniienko_photo'] = photo
            item['korniienko_drawing'] = drawing
        
        # Load metadata
        language = row.get('language', 'unknown')
        writing_system = row.get('writing_system', 'unknown')
        
        item['language'] = self.language_to_idx.get(language, 0)
        item['writing_system'] = self.ws_to_idx.get(writing_system, 0)
        
        return item
    
    def _load_rti_images(self, isialy_id):
        """Load and stack RTI images (4 types × 3 RGB = 12 channels)."""
        rti_dir = self.data_dir / 'cropped_images_hq' / 'train'
        rti_types = ['original', 'blended', 'normal', 'texture']
        
        channels = []
        for img_type in rti_types:
            img_path = rti_dir / img_type / f"{isialy_id}.png"
            
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
    
    def _load_korniienko_images(self, isialy_id):
        """Load Korniienko photo and drawing."""
        korniienko_dir = self.data_dir / 'korniienko_images' / isialy_id
        
        # Load photo
        photo_path = korniienko_dir / 'photo.jpg'
        if photo_path.exists():
            photo = Image.open(photo_path).convert('RGB')
            photo_tensor = self.transform(photo)
        else:
            photo_tensor = None
        
        # Load drawing
        drawing_path = korniienko_dir / 'drawing.jpg'
        if drawing_path.exists():
            drawing = Image.open(drawing_path).convert('RGB')
            drawing_tensor = self.transform(drawing)
        else:
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
    
    # Handle Korniienko photo (may have None values)
    if 'korniienko_photo' in batch[0]:
        photos = [item['korniienko_photo'] for item in batch if item['korniienko_photo'] is not None]
        result['korniienko_photo'] = torch.stack(photos) if photos else None
    
    # Handle Korniienko drawing (may have None values)
    if 'korniienko_drawing' in batch[0]:
        drawings = [item['korniienko_drawing'] for item in batch if item['korniienko_drawing'] is not None]
        result['korniienko_drawing'] = torch.stack(drawings) if drawings else None
    
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
        
        # Forward pass (different for transformer)
        if model_type == 'transformer':
            kwargs['text_indices'] = input_ids
            kwargs['text_mask'] = attention_mask
            if 'images' in kwargs:
                kwargs['rti_images'] = kwargs.pop('images')
            outputs = model(**kwargs)
            logits = outputs['transcription']
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
                logits = outputs['transcription']
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
    parser.add_argument('--train_csv', type=str, default='data/train_comprehensive.csv',
                        help='Training CSV file (relative to data_dir)')
    parser.add_argument('--val_csv', type=str, default='data/val_comprehensive.csv',
                        help='Validation CSV file (relative to data_dir)')
    
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
    print(f"\n📦 Model: {args.model.upper()}")
    print(f"🖼️  Modalities:")
    print(f"   RTI Images: {'✓' if args.use_rti else '✗'}")
    print(f"   Korniienko: {'✓' if args.use_korniienko else '✗'}")
    print(f"\n⚙️  Hyperparameters:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Image Size: {args.image_size}×{args.image_size}")
    print(f"   Max Text Length: {args.max_length}")
    print(f"\n💾 Data:")
    print(f"   Base Dir: {args.data_dir}")
    print(f"   Train CSV: {args.train_csv}")
    print(f"   Val CSV: {args.val_csv}")
    print(f"\n🔧 Device: {args.device}")
    print("=" * 70)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.model / datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Checkpoints: {checkpoint_dir}")
    
    # Save configuration
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ Config saved: {config_path}")
    
    # Initialize tokenizer
    print("\n🔤 Initializing tokenizer...")
    if XLMRobertaTokenizer is not None:
        try:
            tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
            vocab_size = tokenizer.vocab_size
            print(f"✓ XLM-RoBERTa tokenizer loaded (vocab size: {vocab_size})")
        except:
            print("⚠️  XLM-RoBERTa failed, using character tokenizer")
            tokenizer = CharacterTokenizer()
            vocab_size = tokenizer.vocab_size
    else:
        tokenizer = CharacterTokenizer()
        vocab_size = tokenizer.vocab_size
        print(f"✓ Character tokenizer initialized (vocab size: {vocab_size})")
    
    # Create datasets
    print("\n📊 Loading datasets...")
    train_dataset = SophiaMultiModalDataset(
        csv_file=Path(args.data_dir) / args.train_csv,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_rti=args.use_rti,
        use_korniienko=args.use_korniienko,
        model_type=args.model,
        image_size=args.image_size,
        augment=True
    )
    
    val_dataset = SophiaMultiModalDataset(
        csv_file=Path(args.data_dir) / args.val_csv,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_rti=args.use_rti,
        use_korniienko=args.use_korniienko,
        model_type=args.model,
        image_size=args.image_size,
        augment=False
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
    print("\n🏗️  Creating model...")
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
        print(f"\n📂 Resuming from: {args.resume}")
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
        print(f"\n📅 Epoch {epoch + 1}/{args.epochs}")
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
        print(f"\n📊 Epoch {epoch + 1} Results:")
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
                'config': vars(args)
            }, checkpoint_path)
            
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = checkpoint_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'config': vars(args)
                }, best_model_path)
                print(f"🏆 Best model saved: {best_model_path} (val_loss: {val_loss:.4f})")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n🏆 Best Validation Loss: {best_val_loss:.4f}")
    print(f"📁 Checkpoints saved in: {checkpoint_dir}")
    
    # Save final model
    final_model_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'config': vars(args)
    }, final_model_path)
    print(f"💾 Final model saved: {final_model_path}")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()

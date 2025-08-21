#!/usr/bin/env python3
"""
SOPHIA Training Script
=====================

Unified training script for SOPHIA graffiti recognition models.
Supports tw        # Tokenize with custom tokenizer
        encoded = self.tokenizer(
            conditioned_transcription,
            max_length=self.max_length,
            padding=True,
            truncation=True
        )
        
        return {
            'image': image,
            'input_ids': torch.tensor(encoded['input_ids'        print(        print(" SOPHIA Model Architectures:")
        print("\n Multi-Channel Model:")
        print("   - 12-channel vision encoder (4 image types × 3 RGB)")
        print("   - 8-layer transformer decoder")
        print("   - Language + writing system conditioning")
        print("   - ~207M parameters")
        print("   - Best for: Comprehensive feature extraction")
        
        print("\n⚡ Enhanced Model v2.0:")
        print("   - Deep ResNet vision encoder with attention")
        print("   - 8-layer transformer decoder")
        print("   - Advanced language conditioning")
        print("   - Channel + spatial + cross-modal attention")
        print("   - ~311M parameters")
        print("   - Best for: Maximum performance, language conditioning")
        print(" SOPHIA Model Architectures:")
        print("\n Multi-Channel Model:")
        print("   - 12-channel vision encoder (4 image types × 3 RGB)")
        print("   - 8-layer transformer decoder")
        print("   - Language + writing system conditioning")
        print("   - ~207M parameters")
        print("   - Best for: Comprehensive feature extraction")
        
        print("\n⚡ Enhanced Model v2.0:")
        print("   - Deep ResNet vision encoder with attention")
        print("   - 8-layer transformer decoder")
        print("   - Advanced language conditioning")
        print("   - Channel + spatial + cross-modal attention")
        print("   - ~311M parameters")
        print("   - Best for: Maximum performance, language conditioning").long),
            'attention    if args.model_type == 'multichannel':
        model = MultiChannelModel(vocab_size, vision_dim=512, hidden_dim=512, num_layers=8)
        model_info = model.get_model_info()
        print(f" {model_info['description']}")
        print(f" Parameters: {model_info['total_parameters']/1e6:.1f}M total")
        print(f" Vision: {model_info['vision_encoder']}")
        print(f" Decoder: {model_info['decoder']}")
        print(f" Conditioning: {model_info['language_conditioning']}")
    else:
        # Enhanced model v2.0 with deep learning and language conditioning
        model = EnhancedModel(
            vocab_size=vocab_size, 
            vision_dim=512, 
            hidden_dim=512, 
            num_layers=8,
            num_languages=10,
            num_writing_systems=5
        )
        model_info = model.get_model_info()
        print(f" {model_info['description']}")
        print(f" Parameters: {model_info['total_parameters']/1e6:.1f}M total")
        print(f" Vision: {model_info['vision_encoder']}")
        print(f" Decoder: {model_info['decoder']}")
        print(f" Conditioning: {model_info['language_conditioning']}")
        print(f" Attention: {model_info['attention_mechanisms']}")
        print(f" Regularization: {model_info['regularization']}")nsor(encoded['attention_mask'], dtype=torch.long),
            'transcription': transcription,
            'language': language
        }
1. Multi-channel: Comprehensive 12-channel vision processing
2. Enhanced: Simplified but effective language-conditioned approach

Usage:
    python train_sophia.py --model_type enhanced --epochs 12
    python train_sophia.py --model_type multichannel --epochs 15
"""

import os
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
import warnings
warnings.filterwarnings('ignore')

# Import model architectures
from models_multichannel import MultiChannelModel
from models_enhanced import EnhancedModel
from transformers import XLMRobertaTokenizer

# Increase PIL image size limit to handle large graffiti images
Image.MAX_IMAGE_PIXELS = None

class SophiaDataset(Dataset):
    """Unified dataset class supporting both multi-channel and enhanced models."""
    
    def __init__(self, csv_file, cropped_images_dir, tokenizer, max_length=128, 
                 transform=None, model_type='enhanced'):
        """
        Args:
            csv_file: Path to the annotations CSV
            cropped_images_dir: Directory containing cropped images
            tokenizer: XLM-RoBERTa tokenizer
            max_length: Maximum sequence length
            transform: Image transforms
            model_type: 'multichannel' or 'enhanced'
        """
        self.df = pd.read_csv(csv_file)
        self.cropped_images_dir = cropped_images_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.model_type = model_type
        
        # Create language mappings based on model type
        self._create_mappings()
        
        # Filter for valid annotations
        valid_indices = []
        for idx, row in self.df.iterrows():
            annotation_id = row['id']
            if self._check_all_images_exist(annotation_id) and self._has_valid_targets(row):
                valid_indices.append(idx)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"SOPHIA Dataset ({model_type}): {len(self.df)} valid annotations")
        
    def _create_mappings(self):
        """Create language mappings based on model type."""
        languages = self.df['language'].fillna('unknown').unique()
        
        if self.model_type == 'enhanced':
            # Simplified approach - only language conditioning
            self.language_map = {lang: f"[{lang.upper()}]" for lang in languages}
        else:
            # Multi-channel approach - language + writing system
            writing_systems = self.df['writing_system'].fillna('unknown').unique()
            self.language_map = {lang: f"[{lang.upper()}]" for lang in languages}
            self.writing_system_map = {ws: f"[{ws.upper()}]" for ws in writing_systems}
        
        print(f"Language prefixes: {list(self.language_map.values())}")
        
    def _has_valid_targets(self, row):
        """Check if row has valid transcription."""
        return ('transcription' in self.df.columns and 
                pd.notna(row.get('transcription')) and 
                str(row.get('transcription', '')).strip())
        
    def _check_all_images_exist(self, annotation_id):
        """Check if all 4 image types exist."""
        image_types = ['original', 'blended', 'normal', 'texture']
        for img_type in image_types:
            img_path = os.path.join(self.cropped_images_dir, img_type, f"{annotation_id}_{img_type}.jpg")
            if not os.path.exists(img_path):
                return False
        return True
    
    def _load_multichannel_image(self, annotation_id):
        """Load all 4 image types for multi-channel processing."""
        image_types = ['original', 'blended', 'normal', 'texture']
        channels = []
        
        for img_type in image_types:
            img_path = os.path.join(self.cropped_images_dir, img_type, f"{annotation_id}_{img_type}.jpg")
            
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                channels.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                blank = torch.zeros(3, 224, 224)
                channels.append(blank)
        
        return torch.cat(channels, dim=0)  # Shape: [12, 224, 224]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        annotation_id = row['id']
        transcription = str(row['transcription']) if pd.notna(row['transcription']) else ""
        
        # Get language and writing system information
        language = row['language'] if pd.notna(row['language']) else 'unknown'
        writing_system = row['writing_system'] if pd.notna(row['writing_system']) else 'unknown'
        
        if self.model_type == 'enhanced':
            # Simplified language conditioning
            language_prefix = self.language_map.get(language, "[UNKNOWN]")
            conditioned_transcription = f"{language_prefix} {transcription}"
        else:
            # Multi-channel with language + writing system
            language_prefix = self.language_map.get(language, "[UNKNOWN]")
            ws_prefix = self.writing_system_map.get(writing_system, "[UNKNOWN]")
            conditioned_transcription = f"{language_prefix}{ws_prefix} {transcription}"
        
        # Load image (same multi-channel format for both models)
        image = self._load_multichannel_image(annotation_id)
        
        # Tokenize
        tokens = self.tokenizer(
            conditioned_transcription,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'transcription': transcription,
            'annotation_id': annotation_id,
            'language': language,
            'writing_system': writing_system
        }

def train_model(model, train_loader, val_loader, tokenizer, model_type, num_epochs=15):
    """Enhanced training function with improved learning strategies."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Multi-stage learning rate schedule
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95))
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, T_mult=2, eta_min=1e-6
    )
    
    # Improved loss functions
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
    
    # Additional loss for vision-text alignment
    mse_loss = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7  # Increased patience for better training
    
    # Training history for monitoring
    train_losses = []
    val_losses = []
    learning_rates = []
    
    # Early stopping with loss increase detection
    loss_increase_threshold = 0.1  # Stop if loss increases by more than 10%
    consecutive_increases = 0
    max_consecutive_increases = 3
    
    # Learning curriculum - start with shorter sequences
    def get_curriculum_max_length(epoch):
        if epoch < 3:
            return 32  # Start with shorter sequences
        elif epoch < 6:
            return 64
        elif epoch < 9:
            return 96
        else:
            return 128  # Full length
    
    print(f" Early stopping: patience={patience}, loss increase threshold={loss_increase_threshold:.1%}")
    print(f" Curriculum learning: 32→64→96→128 tokens")
    
    
    for epoch in range(num_epochs):
        # Adjust learning based on epoch
        curr_max_len = get_curriculum_max_length(epoch)
        
        # Training
        model.train()
        total_train_loss = 0
        total_alignment_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train] LR:{optimizer.param_groups[0]["lr"]:.6f}')
        
        for batch_idx, batch in enumerate(train_bar):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Curriculum learning - truncate sequences if needed
            if input_ids.size(1) > curr_max_len:
                input_ids = input_ids[:, :curr_max_len]
                attention_mask = attention_mask[:, :curr_max_len]
            
            # Prepare target (shifted input_ids)
            target_ids = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            attention_mask = attention_mask[:, :-1].contiguous()
            
            optimizer.zero_grad()
            
            # Forward pass with language conditioning for multichannel
            if hasattr(model, 'forward') and 'languages' in batch and 'writing_system' in batch:
                # Multi-channel model with language conditioning
                logits = model(
                    images, input_ids, attention_mask,
                    languages=batch['language'],
                    writing_systems=batch['writing_system']
                )
            else:
                # Enhanced model or fallback
                logits = model(images, input_ids, attention_mask)
            
            # Primary language modeling loss
            lm_loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Vision-text alignment loss (every few batches to avoid overhead)
            alignment_loss = 0
            if batch_idx % 4 == 0:  # Apply alignment loss every 4th batch
                # Get vision features
                vision_features = model.vision_encoder(images)
                if hasattr(model, 'vision_projection'):
                    vision_features = model.vision_projection(vision_features)
                
                # Text features from decoder (average pooling)
                with torch.no_grad():
                    text_embeddings = model.decoder.token_embedding(input_ids)
                    text_features = text_embeddings.mean(dim=1)  # [batch, hidden_dim]
                
                # Alignment loss - encourage vision and text features to be similar
                alignment_loss = mse_loss(vision_features, text_features.detach()) * 0.1
                total_alignment_loss += alignment_loss.item()
            
            # Combined loss
            total_loss = lm_loss + alignment_loss
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_train_loss += lm_loss.item()
            train_bar.set_postfix({
                'lm_loss': f'{lm_loss.item():.4f}',
                'align_loss': f'{alignment_loss.item() if isinstance(alignment_loss, torch.Tensor) else alignment_loss:.4f}',
                'max_len': curr_max_len
            })
        
        # Update learning rate
        scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_alignment_loss = total_alignment_loss / (len(train_loader) // 4) if total_alignment_loss > 0 else 0
        
        # Validation with early stopping
        model.eval()
        total_val_loss = 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for batch in val_bar:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Prepare target
                target_ids = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                attention_mask = attention_mask[:, :-1].contiguous()
                
                # Forward pass with language conditioning
                if hasattr(model, 'forward') and 'languages' in batch and 'writing_system' in batch:
                    # Multi-channel model with language conditioning
                    logits = model(
                        images, input_ids, attention_mask,
                        languages=batch['language'],
                        writing_systems=batch['writing_system']
                    )
                else:
                    # Enhanced model or fallback
                    logits = model(images, input_ids, attention_mask)
                    
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                total_val_loss += loss.item()
                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = total_val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store training history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        learning_rates.append(current_lr)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Alignment: {avg_alignment_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}')
        
        # Check for loss increase (potential overfitting)
        if len(val_losses) > 1:
            prev_val_loss = val_losses[-2]
            loss_increase = (avg_val_loss - prev_val_loss) / prev_val_loss
            
            if loss_increase > loss_increase_threshold:
                consecutive_increases += 1
                print(f"  Validation loss increased by {loss_increase:.1%} (consecutive: {consecutive_increases})")
                
                # Reduce learning rate if loss keeps increasing
                if consecutive_increases >= 2:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print(f" Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")
            else:
                consecutive_increases = 0
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            improvement = (best_val_loss - avg_val_loss) / best_val_loss * 100
            best_val_loss = avg_val_loss
            patience_counter = 0
            consecutive_increases = 0  # Reset since we improved
            
            model_path = f'models/best_{model_type}_model.pth'
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer_vocab': tokenizer.get_vocab(),
                'model_type': model_type,
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'train_history': {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'learning_rates': learning_rates
                }
            }, model_path)
            print(f' Best model saved: {model_path} (Val Loss: {avg_val_loss:.4f}, Improvement: {improvement:.2f}%)')
        else:
            patience_counter += 1
            print(f" No improvement for {patience_counter}/{patience} epochs")
            
            # Check multiple stopping conditions
            stop_training = False
            
            # Condition 1: Patience exceeded
            if patience_counter >= patience:
                print(f" Early stopping: No improvement for {patience} epochs")
                stop_training = True
            
            # Condition 2: Too many consecutive loss increases
            if consecutive_increases >= max_consecutive_increases:
                print(f" Early stopping: Loss increased {consecutive_increases} consecutive times")
                stop_training = True
            
            # Condition 3: Learning rate too small
            if current_lr < 1e-7:
                print(f" Early stopping: Learning rate too small ({current_lr:.2e})")
                stop_training = True
            
            if stop_training:
                print(f" Training stopped at epoch {epoch+1}")
                print(f" Best validation loss: {best_val_loss:.4f}")
                break
    
    # Training summary
    print(f"\n{'='*60}")
    print(f" TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f" Best validation loss: {best_val_loss:.4f}")
    print(f" Total epochs: {len(train_losses)}")
    print(f" Model saved: models/best_{model_type}_model.pth")
    
    # Show training progress
    if len(train_losses) > 1:
        initial_train_loss = train_losses[0]
        final_train_loss = train_losses[-1]
        train_improvement = (initial_train_loss - final_train_loss) / initial_train_loss * 100
        print(f" Training loss improvement: {train_improvement:.1f}% ({initial_train_loss:.4f} → {final_train_loss:.4f})")
        
        initial_val_loss = val_losses[0]
        final_val_loss = val_losses[-1]
        val_improvement = (initial_val_loss - final_val_loss) / initial_val_loss * 100
        print(f" Validation loss improvement: {val_improvement:.1f}% ({initial_val_loss:.4f} → {final_val_loss:.4f})")

def main():
    parser = argparse.ArgumentParser(description='Train SOPHIA Graffiti Recognition Model')
    parser.add_argument('--model_type', choices=['multichannel', 'enhanced'], 
                        default='enhanced', help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--info', action='store_true', help='Show model information and exit')
    
    args = parser.parse_args()
    
    # Show model information if requested
    if args.info:
        print(" SOPHIA Model Architectures:")
        print("\n Multi-Channel Model:")
        print("   - 12-channel vision encoder (4 image types × 3 RGB)")
        print("   - 8-layer transformer decoder")
        print("   - Language + writing system conditioning")
        print("   - ~207M parameters")
        print("   - Best for: Comprehensive feature extraction")
        
        print("\n⚡ Enhanced Model v2.0:")
        print("   - Deep ResNet vision encoder with attention")
        print("   - 8-layer transformer decoder")
        print("   - Advanced language conditioning")
        print("   - Channel + spatial + cross-modal attention")
        print("   - ~311M parameters")
        print("   - Best for: Maximum performance, language conditioning")
        
        print("\n Usage Examples:")
        print("   python train_sophia.py --model_type enhanced --epochs 12")
        print("   python train_sophia.py --model_type multichannel --epochs 15")
        return
    
    print("=" * 60)
    print(f" SOPHIA Training: {args.model_type.upper()} Model")
    print("=" * 60)
    print(f" Configuration: {args.epochs} epochs, batch size {args.batch_size}, lr {args.lr}")
    
    # Setup paths
    data_dir = 'data'
    train_csv = os.path.join(data_dir, 'train_dataset.csv')
    val_csv = os.path.join(data_dir, 'val_dataset.csv')
    cropped_images_dir = os.path.join(data_dir, 'cropped_images')
    
    # Initialize tokenizer
    print(" Loading XLM-RoBERTa tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    print(f" Loaded tokenizer with vocab size: {len(tokenizer)}")
    
    print(f" Expected initial loss: ~{torch.log(torch.tensor(float(len(tokenizer)))):.1f}")
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SophiaDataset(
        train_csv, 
        os.path.join(cropped_images_dir, 'train'),
        tokenizer, 
        model_type=args.model_type,
        transform=transform
    )
    
    val_dataset = SophiaDataset(
        val_csv, 
        os.path.join(cropped_images_dir, 'val'),
        tokenizer, 
        model_type=args.model_type,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model and show detailed information
    vocab_size = len(tokenizer)
    
    if args.model_type == 'multichannel':
        model = MultiChannelModel(vocab_size, vision_dim=512, hidden_dim=512, num_layers=8)
        model_info = model.get_model_info()
        print(f" {model_info['description']}")
        print(f" Parameters: {model_info['total_parameters']/1e6:.1f}M total")
        print(f" Vision: {model_info['vision_encoder']}")
        print(f" Decoder: {model_info['decoder']}")
        print(f" Conditioning: {model_info['language_conditioning']}")
    else:
        model = EnhancedModel(vocab_size, vision_dim=256, hidden_dim=256, num_layers=6)
        model_info = model.get_model_info()
        print(f" {model_info['description']}")
        print(f" Parameters: {model_info['total_parameters']/1e6:.1f}M total")
        print(f" Vision: {model_info['vision_encoder']}")
        print(f" Decoder: {model_info['decoder']}")
        print(f" Conditioning: {model_info['language_conditioning']}")
    
    print("=" * 60)
    
    # Train model
    train_model(model, train_loader, val_loader, tokenizer, args.model_type, args.epochs)
    
    print("=" * 60)
    print(f" Training completed! Best model saved in models/best_{args.model_type}_model.pth")
    print("=" * 60)

if __name__ == '__main__':
    main()

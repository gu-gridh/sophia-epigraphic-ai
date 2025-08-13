"""
Training utilities and main training loop for SOPHIA.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
from typing import Dict, Optional
import json

from .models import SophiaModel, get_default_config
from .data import create_dataloaders


class SophiaTrainer:
    """
    Main trainer class for SOPHIA model.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        config: Dict,
        model: Optional[SophiaModel] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.config = config
        self.device = device
        
        # Initialize model
        if model is None:
            model_config = config.get('model', get_default_config())
            self.model = SophiaModel(model_config, config['vocab_size'])
        else:
            self.model = model
        
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 for padding
        self.restoration_criterion = nn.BCELoss()
        self.dating_criterion = nn.MSELoss()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Logging
        if config.get('use_wandb', True):
            wandb.init(
                project="sophia-epigraphic-ai",
                config=config,
                name=config.get('experiment_name', 'sophia_experiment')
            )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adamw')
        
        if optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 5e-4),
                weight_decay=optimizer_config.get('weight_decay', 0.01),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 5e-4),
                weight_decay=optimizer_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        if not scheduler_config:
            return None
        
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 100),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'linear':
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=scheduler_config.get('end_factor', 0.1),
                total_iters=scheduler_config.get('total_iters', 1000)
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_transcription_loss = 0.0
        total_restoration_loss = 0.0
        total_dating_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                images=batch['image'],
                spatial_features=batch['spatial_features'],
                text_features=batch['text_features'],
                metadata_features=batch['metadata_features'],
                target_ids=batch.get('target_ids')
            )
            
            # Calculate losses
            losses = self._calculate_losses(outputs, batch)
            total_loss_batch = losses['total_loss']
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip_norm', 0) > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip_norm']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            total_transcription_loss += losses['transcription_loss'].item()
            total_restoration_loss += losses['restoration_loss'].item()
            total_dating_loss += losses['dating_loss'].item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss_batch.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log to wandb
            if self.global_step % self.config.get('log_interval', 100) == 0:
                self._log_metrics({
                    'train/total_loss': total_loss_batch.item(),
                    'train/transcription_loss': losses['transcription_loss'].item(),
                    'train/restoration_loss': losses['restoration_loss'].item(),
                    'train/dating_loss': losses['dating_loss'].item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': self.current_epoch,
                    'train/step': self.global_step
                })
        
        # Calculate average losses
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'transcription_loss': total_transcription_loss / num_batches,
            'restoration_loss': total_restoration_loss / num_batches,
            'dating_loss': total_dating_loss / num_batches
        }
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_transcription_loss = 0.0
        total_restoration_loss = 0.0
        total_dating_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(
                    images=batch['image'],
                    spatial_features=batch['spatial_features'],
                    text_features=batch['text_features'],
                    metadata_features=batch['metadata_features'],
                    target_ids=batch.get('target_ids')
                )
                
                # Calculate losses
                losses = self._calculate_losses(outputs, batch)
                
                # Update metrics
                total_loss += losses['total_loss'].item()
                total_transcription_loss += losses['transcription_loss'].item()
                total_restoration_loss += losses['restoration_loss'].item()
                total_dating_loss += losses['dating_loss'].item()
                num_batches += 1
        
        # Calculate average losses
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'transcription_loss': total_transcription_loss / num_batches,
            'restoration_loss': total_restoration_loss / num_batches,
            'dating_loss': total_dating_loss / num_batches
        }
        
        return avg_losses
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log epoch metrics
            epoch_metrics = {
                f'epoch/train_{k}': v for k, v in train_losses.items()
            }
            epoch_metrics.update({
                f'epoch/val_{k}': v for k, v in val_losses.items()
            })
            self._log_metrics(epoch_metrics)
            
            # Print epoch summary
            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            print(f"  Val Loss: {val_losses['total_loss']:.4f}")
            
            # Save checkpoint
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_checkpoint(is_best=True)
                print(f"  New best model saved!")
            
            # Regular checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(is_best=False)
        
        print("Training completed!")
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _calculate_losses(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """Calculate all loss components."""
        losses = {}
        
        # Transcription loss
        if 'target_text' in batch and 'transcription_logits' in outputs:
            # Simplified transcription loss (would need proper tokenization)
            transcription_loss = torch.tensor(0.0, device=self.device)
        else:
            transcription_loss = torch.tensor(0.0, device=self.device)
        
        # Restoration loss (confidence in text restoration)
        restoration_loss = torch.tensor(0.0, device=self.device)
        
        # Dating loss
        dating_loss = torch.tensor(0.0, device=self.device)
        
        # Total loss (weighted combination)
        total_loss = (
            self.config.get('transcription_weight', 1.0) * transcription_loss +
            self.config.get('restoration_weight', 0.1) * restoration_loss +
            self.config.get('dating_weight', 0.1) * dating_loss
        )
        
        losses.update({
            'transcription_loss': transcription_loss,
            'restoration_loss': restoration_loss,
            'dating_loss': dating_loss,
            'total_loss': total_loss
        })
        
        return losses
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb."""
        if self.config.get('use_wandb', True):
            wandb.log(metrics, step=self.global_step)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{self.current_epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


def create_trainer_from_config(config_path: str) -> SophiaTrainer:
    """Create trainer from configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return SophiaTrainer(config)

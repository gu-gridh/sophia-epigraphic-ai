#!/usr/bin/env python3
"""
Transformer-Based Graffiti Recognition Model for Saint Sophia
==============================================================

Inspired by Google DeepMind's "Predicting the Past" (Aeneas) project:
https://github.com/google-deepmind/predictingthepast

Key Features:
- Transformer encoder ("torso") for processing multi-modal inputs
- Multi-head attention mechanisms for focusing on important inscription regions
- Multi-modal fusion: RTI images (4 channels) + Korniienko images + text
- Task-specific heads for transcription, dating, and attribution
- Supports variable-length inputs and missing modalities

Architecture Components:
1. Vision Encoder: Processes RTI multi-channel and Korniienko images
2. Text Encoder: Embeds existing transcriptions for text-guided recognition
3. Spatial Encoder: Encodes bounding box and location information
4. Transformer Torso: Multi-head attention for feature fusion
5. Decoder Heads: Transcription (character-level), dating, language classification

Customizations for Saint Sophia:
- Cyrillic/Greek character vocabulary support
- Byzantine/Church Slavonic text modeling
- Multi-channel RTI image processing (original, blended, normal, texture)
- Korniienko reference image integration
- Spatial-aware attention for cathedral wall locations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create sinusoidal position encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiModalVisionEncoder(nn.Module):
    """
    Vision encoder for RTI multi-channel and Korniienko images.
    
    Processes:
    - RTI images: 4 types × 3 RGB channels = 12 channels total
    - Korniienko photo: 3 RGB channels
    - Korniienko drawing: 3 RGB channels (or 1 grayscale)
    
    Uses ResNet-style architecture with attention for multi-channel fusion.
    """
    
    def __init__(self, embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # RTI image encoder (12 channels: 4 types × 3 RGB)
        self.rti_encoder = nn.Sequential(
            # Initial conv to reduce 12 channels
            nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet-style blocks
            self._make_layer(64, 128, stride=1),
            self._make_layer(128, 256, stride=2),
            self._make_layer(256, 512, stride=2),
            
            # Global pooling and projection
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # Korniienko photo encoder (3 RGB channels)
        self.korniienko_photo_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            self._make_layer(64, 128, stride=1),
            self._make_layer(128, 256, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # Korniienko drawing encoder (3 channels or 1 grayscale)
        self.korniienko_drawing_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            self._make_layer(64, 128, stride=1),
            self._make_layer(128, 256, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # Multi-modal fusion with attention
        self.modality_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Learned modality embeddings
        self.rti_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.korniienko_photo_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.korniienko_drawing_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def _make_layer(self, in_channels, out_channels, stride=1):
        """Create a ResNet-style convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        rti_images: Optional[torch.Tensor] = None,
        korniienko_photo: Optional[torch.Tensor] = None,
        korniienko_drawing: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rti_images: [batch, 12, H, W] - 4 RTI types concatenated
            korniienko_photo: [batch, 3, H, W]
            korniienko_drawing: [batch, 3, H, W]
            
        Returns:
            vision_features: [batch, num_modalities, embed_dim]
            modality_mask: [batch, num_modalities] - indicates available modalities
        """
        batch_size = (rti_images.size(0) if rti_images is not None 
                     else korniienko_photo.size(0) if korniienko_photo is not None
                     else korniienko_drawing.size(0))
        
        vision_features = []
        modality_mask = []
        
        # Encode RTI images
        if rti_images is not None:
            rti_feat = self.rti_encoder(rti_images)  # [batch, embed_dim]
            rti_feat = rti_feat.unsqueeze(1) + self.rti_token  # [batch, 1, embed_dim]
            vision_features.append(rti_feat)
            modality_mask.append(torch.ones(batch_size, 1, device=rti_images.device))
        
        # Encode Korniienko photo
        if korniienko_photo is not None:
            photo_feat = self.korniienko_photo_encoder(korniienko_photo)
            photo_feat = photo_feat.unsqueeze(1) + self.korniienko_photo_token
            vision_features.append(photo_feat)
            modality_mask.append(torch.ones(batch_size, 1, device=korniienko_photo.device))
        
        # Encode Korniienko drawing
        if korniienko_drawing is not None:
            drawing_feat = self.korniienko_drawing_encoder(korniienko_drawing)
            drawing_feat = drawing_feat.unsqueeze(1) + self.korniienko_drawing_token
            vision_features.append(drawing_feat)
            modality_mask.append(torch.ones(batch_size, 1, device=korniienko_drawing.device))
        
        if not vision_features:
            # No images available - return zeros
            return torch.zeros(batch_size, 1, self.embed_dim), torch.zeros(batch_size, 1)
        
        # Concatenate all available modalities
        vision_features = torch.cat(vision_features, dim=1)  # [batch, num_modalities, embed_dim]
        modality_mask = torch.cat(modality_mask, dim=1)  # [batch, num_modalities]
        
        # Apply cross-attention for modality fusion
        fused_features, _ = self.modality_attention(
            vision_features, vision_features, vision_features,
            key_padding_mask=(modality_mask == 0)
        )
        
        return fused_features, modality_mask


class SpatialEncoder(nn.Module):
    """
    Encodes spatial information: bounding box coordinates, panel location, elevation.
    
    Helps the model understand WHERE on the cathedral wall the inscription is located.
    """
    
    def __init__(self, embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # Bounding box encoder (x, y, width, height in 0-1 range)
        self.bbox_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2)
        )
        
        # Location encoder (panel_id, room, elevation)
        self.location_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        bbox: torch.Tensor,  # [batch, 4] (x, y, w, h)
        location: torch.Tensor  # [batch, 3] (panel_id, room, elevation)
    ) -> torch.Tensor:
        """
        Returns:
            spatial_features: [batch, embed_dim]
        """
        bbox_feat = self.bbox_encoder(bbox)  # [batch, embed_dim/2]
        loc_feat = self.location_encoder(location)  # [batch, embed_dim/2]
        
        # Concatenate and project
        spatial_feat = torch.cat([bbox_feat, loc_feat], dim=-1)  # [batch, embed_dim]
        spatial_feat = self.projection(spatial_feat)
        
        return spatial_feat


class TextEncoder(nn.Module):
    """
    Encodes existing transcriptions, translations, or partial text for text-guided recognition.
    
    Uses character-level embeddings for Cyrillic/Greek/Latin scripts.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.char_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        
        # Transformer encoder for text
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(
        self,
        text_indices: torch.Tensor,  # [batch, seq_len]
        text_mask: Optional[torch.Tensor] = None  # [batch, seq_len]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            text_features: [batch, seq_len, embed_dim]
            pooled_features: [batch, embed_dim] - for fusion with vision
        """
        # Embed characters
        x = self.char_embedding(text_indices)  # [batch, seq_len, embed_dim]
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch, embed_dim]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch, seq_len, embed_dim]
        
        # Create attention mask for padding
        # text_mask is 1 for valid tokens, 0 for padding
        # PyTorch transformer needs True for positions to mask (padding)
        if text_mask is None:
            padding_mask = (text_indices == 0)  # Padding positions
        else:
            # Convert attention_mask (1=valid, 0=padding) to padding_mask (True=padding, False=valid)
            padding_mask = (text_mask == 0)
        
        # Apply transformer
        text_features = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask
        )
        
        # Pool for fusion (mean pooling over non-padding tokens)
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # [batch, seq_len, 1]
        pooled_features = (text_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        return text_features, pooled_features


class TransformerTorso(nn.Module):
    """
    Main transformer "torso" that fuses multi-modal features.
    
    Inspired by Aeneas architecture - creates historically-enriched embeddings
    by attending across vision, text, and spatial modalities.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        features: torch.Tensor,  # [batch, seq_len, embed_dim]
        mask: Optional[torch.Tensor] = None  # [batch, seq_len]
    ) -> torch.Tensor:
        """
        Returns:
            enriched_features: [batch, seq_len, embed_dim]
        """
        # Apply transformer with attention masking
        x = self.transformer(features, src_key_padding_mask=mask)
        x = self.norm(x)
        
        return x


class TranscriptionHead(nn.Module):
    """
    Character-level decoder for inscription transcription.
    
    Outputs probability distribution over character vocabulary for each position.
    Supports Cyrillic, Greek, Latin, and special characters.
    """
    
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        max_length: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Token embedding for target sequence
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_length, dropout)
        
        # Decoder transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Learnable start token embedding
        self.start_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(
        self,
        memory: torch.Tensor,  # [batch, memory_len, embed_dim] - from torso
        memory_mask: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,  # [batch, tgt_len] - token indices for training
        target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            memory: Encoded vision/context features [batch, memory_len, embed_dim]
            memory_mask: Mask for memory (True for padding)
            target_indices: Target token indices [batch, tgt_len] for teacher forcing
            target_mask: Mask for target sequence
            
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size = memory.size(0)
        
        if target_indices is None:
            # Inference mode - use start token only (will need autoregressive generation)
            target_seq = self.start_token.expand(batch_size, 1, -1)
        else:
            # Training mode - embed target tokens and add positional encoding
            target_seq = self.token_embedding(target_indices)  # [batch, tgt_len, embed_dim]
            target_seq = self.pos_encoder(target_seq)
        
        # Convert memory_mask to proper format (True = padding)
        if memory_mask is not None and memory_mask.dtype != torch.bool:
            memory_key_padding_mask = (memory_mask != 0)
        else:
            memory_key_padding_mask = memory_mask
        
        # Decode
        output = self.decoder(
            target_seq,
            memory,
            tgt_key_padding_mask=target_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)  # [batch, seq_len, vocab_size]
        
        return logits


class AttributionHead(nn.Module):
    """
    Predicts metadata: date range, language, writing system, etc.
    
    Similar to Aeneas' date and geographical attribution heads.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_languages: int = 10,
        num_writing_systems: int = 5,
        date_bins: int = 160,  # Similar to Aeneas: 1010-1715 CE in ~5 year bins
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Language classifier
        self.language_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_languages)
        )
        
        # Writing system classifier
        self.writing_system_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_writing_systems)
        )
        
        # Date attribution (as distribution over time bins)
        self.date_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, date_bins)
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch, embed_dim] - pooled features from torso
            
        Returns:
            predictions: dict with 'language', 'writing_system', 'date' logits
        """
        return {
            'language': self.language_head(features),
            'writing_system': self.writing_system_head(features),
            'date': self.date_head(features)
        }


class SophiaTransformerModel(nn.Module):
    """
    Complete transformer-based model for Saint Sophia graffiti recognition.
    
    Inspired by Google DeepMind's Aeneas (Predicting the Past) architecture,
    customized for Byzantine graffiti with multi-modal inputs:
    - RTI multi-channel images (original, blended, normal, texture)
    - Korniienko reference images (photo + drawing)
    - Existing transcriptions for text-guided recognition
    - Spatial information (location on cathedral walls)
    
    Architecture:
    1. Vision Encoder → Multi-modal image features
    2. Text Encoder → Character embeddings (optional, for guided recognition)
    3. Spatial Encoder → Location and bounding box features
    4. Transformer Torso → Fuses all modalities with attention
    5. Task-specific Heads:
       - Transcription: Character-level sequence generation
       - Attribution: Date, language, writing system prediction
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_torso_layers: int = 3,
        num_decoder_layers: int = 2,
        num_heads: int = 4,
        num_languages: int = 10,
        num_writing_systems: int = 5,
        date_bins: int = 160,
        max_text_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Encoders
        self.vision_encoder = MultiModalVisionEncoder(embed_dim, dropout)
        self.text_encoder = TextEncoder(vocab_size, embed_dim, num_layers=3, num_heads=num_heads, dropout=dropout)
        self.spatial_encoder = SpatialEncoder(embed_dim, dropout)
        
        # Transformer torso for multi-modal fusion
        self.torso = TransformerTorso(embed_dim, num_torso_layers, num_heads, dropout)
        
        # Task-specific heads
        self.transcription_head = TranscriptionHead(
            embed_dim, vocab_size, max_text_length, num_decoder_layers, num_heads, dropout
        )
        self.attribution_head = AttributionHead(
            embed_dim, num_languages, num_writing_systems, date_bins, dropout
        )
        
        # Modality dropout for robustness (randomly drop modalities during training)
        self.modality_dropout = dropout
        
    def forward(
        self,
        rti_images: Optional[torch.Tensor] = None,
        korniienko_photo: Optional[torch.Tensor] = None,
        korniienko_drawing: Optional[torch.Tensor] = None,
        spatial_info: Optional[Dict[str, torch.Tensor]] = None,
        text_indices: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        target_text: Optional[torch.Tensor] = None,  # For training transcription
        return_embeddings: bool = False,
        training: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with flexible multi-modal inputs.
        
        Args:
            rti_images: [batch, 12, H, W] - RTI images (4 types × 3 RGB)
            korniienko_photo: [batch, 3, H, W] - Korniienko photograph
            korniienko_drawing: [batch, 3, H, W] - Korniienko drawing
            spatial_info: dict with 'bbox' [batch, 4], 'location' [batch, 3]
            text_indices: [batch, seq_len] - existing transcription (optional)
            text_mask: [batch, seq_len] - padding mask for text
            target_text: [batch, seq_len, embed_dim] - for teacher forcing during training
            return_embeddings: whether to return torso embeddings for retrieval
            training: whether in training mode (for modality dropout)
            
        Returns:
            outputs: dict with 'transcription_logits', 'attribution', 'embeddings' (optional)
        """
        batch_size = (rti_images.size(0) if rti_images is not None
                     else korniienko_photo.size(0) if korniienko_photo is not None
                     else text_indices.size(0))
        
        # Encode vision modalities
        vision_features, vision_mask = self.vision_encoder(
            rti_images, korniienko_photo, korniienko_drawing
        )  # [batch, num_vision_modalities, embed_dim]
        
        # Encode text (if available - for text-guided recognition)
        text_features, text_pooled = None, None
        if text_indices is not None:
            text_features, text_pooled = self.text_encoder(text_indices, text_mask)
            # Add text as another modality
            text_features = text_features.mean(dim=1, keepdim=True)  # [batch, 1, embed_dim]
        
        # Encode spatial information
        spatial_features = None
        if spatial_info is not None:
            spatial_features = self.spatial_encoder(
                spatial_info['bbox'],
                spatial_info['location']
            )  # [batch, embed_dim]
            spatial_features = spatial_features.unsqueeze(1)  # [batch, 1, embed_dim]
        
        # Concatenate all modalities for transformer torso
        modalities = [vision_features]
        modality_lengths = [vision_features.size(1)]
        
        if text_features is not None:
            modalities.append(text_features)
            modality_lengths.append(1)
        if spatial_features is not None:
            modalities.append(spatial_features)
            modality_lengths.append(1)
        
        # Fuse modalities
        fused_features = torch.cat(modalities, dim=1)  # [batch, total_modalities, embed_dim]
        
        # Create mask for padding
        total_len = sum(modality_lengths)
        fused_mask = torch.zeros(batch_size, total_len, device=fused_features.device)
        # Mark vision padding
        fused_mask[:, :vision_features.size(1)] = (vision_mask == 0).float()
        
        # Apply transformer torso for multi-modal fusion
        enriched_features = self.torso(fused_features, fused_mask)  # [batch, total_len, embed_dim]
        
        # Pool features for attribution (use mean pooling)
        pooled_features = enriched_features.mean(dim=1)  # [batch, embed_dim]
        
        # Task-specific heads
        outputs = {}
        
        # Transcription head
        transcription_logits = self.transcription_head(
            memory=enriched_features,
            memory_mask=fused_mask,
            target_indices=text_indices  # Pass token indices for teacher forcing during training
        )
        outputs['transcription_logits'] = transcription_logits
        
        # Attribution heads
        attribution_preds = self.attribution_head(pooled_features)
        outputs['attribution'] = attribution_preds
        
        # Return embeddings for retrieval/similarity tasks
        if return_embeddings:
            outputs['embeddings'] = pooled_features
        
        return outputs


def create_sophia_transformer(
    vocab_size: int,
    num_languages: int = 10,
    num_writing_systems: int = 5,
    embed_dim: int = 512,
    num_torso_layers: int = 6,
    **kwargs
) -> SophiaTransformerModel:
    """
    Factory function to create SophiaTransformerModel with default settings.
    
    Args:
        vocab_size: Size of character vocabulary (Cyrillic + Greek + Latin + special)
        num_languages: Number of language classes
        num_writing_systems: Number of writing system classes
        embed_dim: Embedding dimension (512 recommended)
        num_torso_layers: Number of transformer layers (6-12 recommended)
        **kwargs: Additional arguments passed to SophiaTransformerModel
        
    Returns:
        model: SophiaTransformerModel instance
    """
    model = SophiaTransformerModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_torso_layers=num_torso_layers,
        num_languages=num_languages,
        num_writing_systems=num_writing_systems,
        **kwargs
    )
    return model


# Example usage and configuration
if __name__ == "__main__":
    # Example: Create model for Saint Sophia graffiti
    
    # Vocabulary size: Cyrillic (33) + Greek (24) + Latin (26) + special chars + padding
    vocab_size = 150
    
    # Language classes: Church Slavonic, Ukrainian, Polish, Greek, etc.
    num_languages = 10
    
    # Writing systems: Cyrillic, Latin, Greek, Mixed, Glagolitic
    num_writing_systems = 5
    
    model = create_sophia_transformer(
        vocab_size=vocab_size,
        num_languages=num_languages,
        num_writing_systems=num_writing_systems,
        embed_dim=512,
        num_torso_layers=6,
        num_decoder_layers=3,
        num_heads=8,
        dropout=0.1
    )
    
    # Example forward pass
    batch_size = 4
    
    # Dummy inputs (replace with real data loaders)
    rti_images = torch.randn(batch_size, 12, 224, 224)  # RTI 4-channel
    korniienko_photo = torch.randn(batch_size, 3, 224, 224)
    spatial_info = {
        'bbox': torch.rand(batch_size, 4),  # (x, y, w, h)
        'location': torch.randn(batch_size, 3)  # (panel_id, room, elevation)
    }
    
    outputs = model(
        rti_images=rti_images,
        korniienko_photo=korniienko_photo,
        spatial_info=spatial_info,
        return_embeddings=True
    )
    
    print("Model outputs:")
    print(f"  Transcription logits: {outputs['transcription_logits'].shape}")
    print(f"  Language logits: {outputs['attribution']['language'].shape}")
    print(f"  Date logits: {outputs['attribution']['date'].shape}")
    print(f"  Embeddings: {outputs['embeddings'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel size:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

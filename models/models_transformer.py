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

from .visual_tokens import FeatureTokens, build_memory


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

    Each modality becomes a grid of visual tokens (H/16 x W/16) with 2D position and
    modality embeddings, so the torso and decoder can attend to individual regions:
    - RTI images: 4 types x 3 RGB = 12 channels
    - Korniienko photo: 3 channels
    - Korniienko drawing: 3 channels
    """
    
    def __init__(self, embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # RTI image encoder (12 channels) -> 512 x H/16 x W/16
        self.rti_encoder = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, 128, stride=1),
            self._make_layer(128, 256, stride=2),
            self._make_layer(256, 512, stride=2),
        )
        
        # Korniienko photo / drawing encoders (3 channels) -> 256 x H/16 x W/16
        self.korniienko_photo_encoder = self._make_korniienko_encoder()
        self.korniienko_drawing_encoder = self._make_korniienko_encoder()
        
        self.rti_tokens = FeatureTokens(512, embed_dim, dropout)
        self.photo_tokens = FeatureTokens(256, embed_dim, dropout)
        self.drawing_tokens = FeatureTokens(256, embed_dim, dropout)
    
    def _make_korniienko_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, 128, stride=2),
            self._make_layer(128, 256, stride=2),
        )
        
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
        korniienko_drawing: Optional[torch.Tensor] = None,
        modality_mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            vision_tokens: [batch, num_tokens, embed_dim]
            padding_mask: [batch, num_tokens] bool, True for images missing for that sample
        """
        modality_mask = modality_mask or {}
        groups = []
        if rti_images is not None:
            groups.append((self.rti_tokens(self.rti_encoder(rti_images)), modality_mask.get('rti')))
        if korniienko_photo is not None:
            groups.append((self.photo_tokens(self.korniienko_photo_encoder(korniienko_photo)),
                           modality_mask.get('photo')))
        if korniienko_drawing is not None:
            groups.append((self.drawing_tokens(self.korniienko_drawing_encoder(korniienko_drawing)),
                           modality_mask.get('drawing')))
        if not groups:
            raise ValueError("At least one image modality must be provided")
        return build_memory(groups)


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
            # PositionalEncoding expects [seq_len, batch, dim]
            target_seq = self.pos_encoder(target_seq.transpose(0, 1)).transpose(0, 1)

        # Convert memory_mask to proper format (True = padding)
        if memory_mask is not None and memory_mask.dtype != torch.bool:
            memory_key_padding_mask = (memory_mask != 0)
        else:
            memory_key_padding_mask = memory_mask

        # Causal mask: position t may only attend to positions <= t
        tgt_len = target_seq.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=target_seq.device), diagonal=1
        )

        # Decode
        output = self.decoder(
            target_seq,
            memory,
            tgt_mask=causal_mask,
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
        # Compatible interface with Enhanced/MultiChannel models
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,  # RTI images (alias for rti_images)
        languages: Optional[torch.Tensor] = None,
        writing_systems: Optional[torch.Tensor] = None,
        korniienko_photo: Optional[torch.Tensor] = None,
        korniienko_drawing: Optional[torch.Tensor] = None,
        # Original transformer-specific parameters
        rti_images: Optional[torch.Tensor] = None,
        spatial_info: Optional[Dict[str, torch.Tensor]] = None,
        text_indices: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        target_text: Optional[torch.Tensor] = None,  # For training transcription
        return_embeddings: bool = False,
        training: bool = False,
        modality_mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass with flexible multi-modal inputs.
        
        Compatible with both:
        1. Enhanced/MultiChannel interface: input_ids, attention_mask, images, languages, writing_systems
        2. Original Transformer interface: rti_images, text_indices, text_mask, spatial_info
        
        Args:
            input_ids: [batch, seq_len] - target transcription (alias for text_indices)
            attention_mask: [batch, seq_len] - padding mask (alias for text_mask)
            images: [batch, 12, H, W] or [batch, 3, H, W] - images (alias for rti_images)
            languages: [batch] - language IDs (unused, for compatibility)
            writing_systems: [batch] - writing system IDs (unused, for compatibility)
            korniienko_photo: [batch, 3, H, W] - Korniienko photograph
            korniienko_drawing: [batch, 3, H, W] - Korniienko drawing
            rti_images: [batch, 12, H, W] - RTI images (4 types × 3 RGB)
            spatial_info: dict with 'bbox' [batch, 4], 'location' [batch, 3]
            text_indices: [batch, seq_len] - existing transcription (optional)
            text_mask: [batch, seq_len] - padding mask for text
            target_text: [batch, seq_len, embed_dim] - for teacher forcing during training
            return_embeddings: whether to return torso embeddings for retrieval
            training: whether in training mode (for modality dropout)
            
        Returns:
            transcription_logits: [batch, seq_len, vocab_size] - compatible with other models
        """
        # The decoder tokens (input_ids / text_indices) are the (shifted) target sequence.
        # They are ONLY given to the causal decoder, never to the text encoder: feeding the
        # target transcription into the encoder memory would leak the answer.
        decoder_ids = input_ids if input_ids is not None else text_indices
        decoder_mask = attention_mask if attention_mask is not None else text_mask
        if rti_images is None and images is not None:
            rti_images = images
        if decoder_ids is None:
            raise ValueError("Decoder input ids are required; use greedy_generate() for inference")

        memory = self.encode(rti_images, korniienko_photo, korniienko_drawing,
                             spatial_info=spatial_info, modality_mask=modality_mask)
        # Return just the logits for compatibility with other models
        return self.decode(decoder_ids, memory, attention_mask=decoder_mask)

    def encode(self, images=None, korniienko_photo=None, korniienko_drawing=None,
               spatial_info=None, context_text_indices=None, context_text_mask=None,
               modality_mask=None, **_):
        """
        Encode images (+ optional spatial info and optional *context* text that is not the
        target, e.g. a partial reading) into the torso memory. Returns (memory, padding_mask).
        """
        # Encode vision modalities as tokens
        vision_features, vision_padding = self.vision_encoder(
            images, korniienko_photo, korniienko_drawing, modality_mask
        )  # [batch, num_tokens, embed_dim], [batch, num_tokens]
        batch_size = vision_features.size(0)

        # Encode optional context text (must never be the target transcription)
        text_features = None
        if context_text_indices is not None:
            text_features, _ = self.text_encoder(context_text_indices, context_text_mask)
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
        
        # Create mask for padding (True = padding)
        total_len = sum(modality_lengths)
        fused_mask = torch.zeros(batch_size, total_len, dtype=torch.bool, device=fused_features.device)
        # Mark vision padding
        fused_mask[:, :vision_features.size(1)] = vision_padding

        # Apply transformer torso for multi-modal fusion
        enriched_features = self.torso(fused_features, fused_mask)  # [batch, total_len, embed_dim]
        return enriched_features, fused_mask

    def decode(self, input_ids, memory, attention_mask=None, languages=None, writing_systems=None):
        """Causal transcription decoding over torso memory from encode()."""
        enriched_features, fused_mask = memory
        target_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        return self.transcription_head(
            memory=enriched_features,
            memory_mask=fused_mask,
            target_indices=input_ids,
            target_mask=target_padding_mask
        )

    def attribute(self, memory):
        """Attribution heads (language / writing system / date) from torso memory."""
        enriched_features, _ = memory
        return self.attribution_head(enriched_features.mean(dim=1))


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
    
    decoder_input = torch.randint(4, vocab_size, (batch_size, 20))
    logits = model(
        input_ids=decoder_input,
        rti_images=rti_images,
        korniienko_photo=korniienko_photo,
        spatial_info=spatial_info,
    )
    memory = model.encode(rti_images, korniienko_photo, spatial_info=spatial_info)
    attribution = model.attribute(memory)

    print("Model outputs:")
    print(f"  Transcription logits: {logits.shape}")
    print(f"  Language logits: {attribution['language'].shape}")
    print(f"  Date logits: {attribution['date'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel size:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

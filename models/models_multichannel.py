#!/usr/bin/env python3
"""
Multi-Channel Graffiti Recognition Model - v2.0
================================================

- Separate stems for the four RTI image types (original, blended, normal, texture),
  weighted by a learned per-type attention.
- Hierarchical CNN with multi-head spatial attention and spatial self-attention,
  producing a grid of visual tokens (H/16 x W/16).
- Optional Korniienko photo / drawing encoders, also as visual tokens.
- Transformer decoder with optional language conditioning, attending over all tokens.
"""

import torch
import torch.nn as nn

from .visual_tokens import FeatureTokens, build_memory


def _conv_bn_relu(cin, cout, k=3, s=1, p=1):
    return [nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


def _korniienko_encoder():
    """3-channel image -> 256 x H/16 x W/16 feature map."""
    return nn.Sequential(
        *_conv_bn_relu(3, 64, k=7, s=2, p=3),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        *_conv_bn_relu(64, 128, s=2),
        *_conv_bn_relu(128, 256, s=2),
    )


class MultiChannelVisionEncoder(nn.Module):
    """12-channel RTI encoder (+ optional Korniienko) producing visual tokens."""

    def __init__(self, input_channels=12, hidden_dim=512, use_korniienko=True):
        super().__init__()
        self.use_korniienko = use_korniienko

        # One stem per RTI image type (3 RGB channels each)
        self.channel_processors = nn.ModuleList([
            nn.Sequential(*_conv_bn_relu(3, 64)) for _ in range(4)
        ])

        # Attention weights for the 4 image types
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 1),
            nn.Sigmoid()
        )

        self.feature_extractor = nn.Sequential(
            *_conv_bn_relu(256, 128, k=7, s=2, p=3),               # H/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),      # H/4
            *_conv_bn_relu(128, 128), *_conv_bn_relu(128, 128),    # fine details
            *_conv_bn_relu(128, 256, s=2), *_conv_bn_relu(256, 256),  # H/8
            *_conv_bn_relu(256, 512, s=2), *_conv_bn_relu(512, 512),  # H/16
            *_conv_bn_relu(512, hidden_dim, s=1),                  # stay at H/16 for token resolution
        )

        # Multiple spatial attention heads
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 8, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 8, 1, 1),
                nn.Sigmoid()
            ) for _ in range(4)
        ])

        self.rti_tokens = FeatureTokens(hidden_dim, hidden_dim)

        # Self-attention across spatial tokens (with residual)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.self_attention_norm = nn.LayerNorm(hidden_dim)

        if self.use_korniienko:
            self.korniienko_photo_encoder = _korniienko_encoder()
            self.korniienko_drawing_encoder = _korniienko_encoder()
            self.photo_tokens = FeatureTokens(256, hidden_dim)
            self.drawing_tokens = FeatureTokens(256, hidden_dim)

        self.output_dim = hidden_dim

    def encode_rti(self, x):
        processed = [self.channel_processors[i](x[:, 3 * i:3 * i + 3]) for i in range(4)]
        combined = torch.cat(processed, dim=1)                       # [B, 256, H, W]
        weights = self.channel_attention(combined)                   # [B, 4, 1, 1]
        weighted = torch.cat([processed[i] * weights[:, i:i + 1] for i in range(4)], dim=1)

        features = self.feature_extractor(weighted)                  # [B, hidden, H/16, W/16]
        features = torch.stack([features * head(features) for head in self.spatial_attention],
                               dim=1).mean(dim=1)

        tokens = self.rti_tokens(features)                           # [B, N, hidden] (+2D position)
        attended, _ = self.self_attention(tokens, tokens, tokens)
        return self.self_attention_norm(tokens + attended)

    def forward(self, x=None, korniienko_photo=None, korniienko_drawing=None, modality_mask=None):
        """Returns (memory [B, N, hidden], padding_mask [B, N])."""
        modality_mask = modality_mask or {}
        groups = []
        if x is not None:
            groups.append((self.encode_rti(x), modality_mask.get('rti')))
        if self.use_korniienko and korniienko_photo is not None:
            groups.append((self.photo_tokens(self.korniienko_photo_encoder(korniienko_photo)),
                           modality_mask.get('photo')))
        if self.use_korniienko and korniienko_drawing is not None:
            groups.append((self.drawing_tokens(self.korniienko_drawing_encoder(korniienko_drawing)),
                           modality_mask.get('drawing')))
        if not groups:
            raise ValueError("At least one image modality must be provided")
        return build_memory(groups)


class LanguageConditionedDecoder(nn.Module):
    """Transformer decoder with optional language / writing-system conditioning."""

    def __init__(self, vocab_size, hidden_dim=512, num_layers=8, num_heads=8,
                 ff_dim=2048, max_length=256, num_languages=10, num_writing_systems=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

        self.language_embedding = nn.Embedding(num_languages, hidden_dim // 2)
        self.writing_system_embedding = nn.Embedding(num_writing_systems, hidden_dim // 2)
        self.language_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.language_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, memory, memory_mask=None, attention_mask=None,
                language_ids=None, writing_system_ids=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.token_embedding(input_ids) + self.position_embedding(positions)

        if language_ids is not None and writing_system_ids is not None:
            lang = torch.cat([self.language_embedding(language_ids),
                              self.writing_system_embedding(writing_system_ids)], dim=-1)
            lang = self.language_projection(lang + self.language_fusion(lang))  # [B, hidden]
            embeddings = embeddings + lang.unsqueeze(1)
            memory = memory + lang.unsqueeze(1)

        embeddings = self.layer_norm(self.dropout(embeddings))

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        tgt_padding = (attention_mask == 0) if attention_mask is not None else None

        output = self.transformer(
            tgt=embeddings,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding,
            memory_key_padding_mask=memory_mask
        )
        return self.output_projection(output)


class MultiChannelModel(nn.Module):
    """Multi-channel graffiti recognition model with visual tokens and optional language conditioning."""

    def __init__(self, vocab_size, vision_dim=512, hidden_dim=512, num_layers=8,
                 num_languages=10, num_writing_systems=5, use_korniienko=True, max_length=256):
        super().__init__()
        self.use_korniienko = use_korniienko
        self.vision_encoder = MultiChannelVisionEncoder(
            input_channels=12,
            hidden_dim=vision_dim,
            use_korniienko=use_korniienko
        )

        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.decoder = LanguageConditionedDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_length=max_length,
            num_languages=num_languages,
            num_writing_systems=num_writing_systems
        )

        self.model_type = "multichannel_v2"
        self.description = "Per-type RTI stems + visual tokens + transformer decoder"

    def forward(self, input_ids, attention_mask=None, images=None, languages=None, writing_systems=None,
                korniienko_photo=None, korniienko_drawing=None, modality_mask=None):
        """Teacher-forced forward pass: input_ids is the (shifted) decoder input."""
        memory = self.encode(images, korniienko_photo, korniienko_drawing, modality_mask=modality_mask)
        return self.decode(input_ids, memory, attention_mask,
                           languages=languages, writing_systems=writing_systems)

    def encode(self, images=None, korniienko_photo=None, korniienko_drawing=None,
               modality_mask=None, **_):
        """Encode images once. Returns (tokens [B, N, H], padding_mask [B, N])."""
        tokens, mask = self.vision_encoder(images, korniienko_photo, korniienko_drawing, modality_mask)
        return self.vision_projection(tokens), mask

    def decode(self, input_ids, memory, attention_mask=None, languages=None, writing_systems=None):
        """`languages` / `writing_systems` are index tensors from the dataset (or None)."""
        tokens, mask = memory
        return self.decoder(input_ids, tokens, memory_mask=mask, attention_mask=attention_mask,
                            language_ids=languages, writing_system_ids=writing_systems)

    def get_model_info(self):
        return {
            'type': self.model_type,
            'description': self.description,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'korniienko_support': self.use_korniienko,
        }

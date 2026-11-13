#!/usr/bin/env python3
"""
Enhanced Graffiti Recognition Model - v3.0
===========================================

- RTI encoder: channel attention over the 12 input channels, ResNet stages,
  spatial attention, and a small FPN that merges stride-16 and stride-32 features.
- Korniienko photo / drawing encoders (optional).
- Every modality is turned into a grid of visual tokens (H/16 x W/16), so the decoder
  can attend to *where* each character is instead of reading from one pooled vector.
- 8-layer pre-norm transformer decoder with optional language conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_tokens import FeatureTokens, build_memory


class ResidualBlock(nn.Module):
    """Residual block for deep feature learning."""

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


def _korniienko_encoder():
    """3-channel image -> 256 x H/16 x W/16 feature map."""
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ResidualBlock(64, 128, stride=2),
        ResidualBlock(128, 256, stride=2),
    )


class EnhancedVisionEncoder(nn.Module):
    """Encodes RTI (+ optional Korniienko) images into visual tokens."""

    def __init__(self, input_channels=12, hidden_dim=512, use_korniienko=True):
        super().__init__()
        self.use_korniienko = use_korniienko

        # Channel attention for different image types (original, blended, normal, texture)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, input_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 4, input_channels, 1),
            nn.Sigmoid()
        )

        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 128, 2, stride=1)    # H/4
        self.layer2 = self._make_layer(128, 256, 2, stride=2)   # H/8
        self.layer3 = self._make_layer(256, 384, 2, stride=2)   # H/16
        self.layer4 = self._make_layer(384, 512, 2, stride=2)   # H/32

        # Spatial attention for graffiti focus (on the deepest features)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # FPN-style merge: stride-32 context upsampled onto the stride-16 grid
        self.lateral3 = nn.Conv2d(384, 256, kernel_size=1)
        self.lateral4 = nn.Conv2d(512, 256, kernel_size=1)
        self.rti_tokens = FeatureTokens(256, hidden_dim)

        if self.use_korniienko:
            self.korniienko_photo_encoder = _korniienko_encoder()
            self.korniienko_drawing_encoder = _korniienko_encoder()
            self.photo_tokens = FeatureTokens(256, hidden_dim)
            self.drawing_tokens = FeatureTokens(256, hidden_dim)

        self.output_dim = hidden_dim

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = [ResidualBlock(inplanes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(planes, planes))
        return nn.Sequential(*layers)

    def encode_rti(self, x):
        x = x * self.channel_attention(x)
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        x4 = x4 * self.spatial_attention(x4)
        fused = self.lateral3(x3) + F.interpolate(self.lateral4(x4), size=x3.shape[-2:],
                                                  mode='bilinear', align_corners=False)
        return self.rti_tokens(fused)

    def forward(self, x=None, korniienko_photo=None, korniienko_drawing=None, modality_mask=None):
        """
        Returns:
            memory: [B, N, hidden_dim] visual tokens of all available modalities
            padding_mask: [B, N] bool, True for tokens of images missing for that sample
        """
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
    """Transformer decoder with optional language conditioning, attending over visual tokens."""

    def __init__(self, vocab_size, hidden_dim=512, num_layers=8, num_heads=8,
                 ff_dim=2048, max_length=256, num_languages=10, num_writing_systems=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Language and writing system embeddings
        self.language_embedding = nn.Embedding(num_languages, hidden_dim // 4)
        self.writing_system_embedding = nn.Embedding(num_writing_systems, hidden_dim // 4)
        self.language_projection = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.15,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.15)

    def get_language_conditioning(self, languages, writing_systems, batch_size, device):
        """[B, hidden] conditioning vector, or zeros when no language info is given."""
        if languages is None or writing_systems is None:
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        lang_emb = self.language_embedding(languages.to(device))
        ws_emb = self.writing_system_embedding(writing_systems.to(device))
        return self.language_projection(torch.cat([lang_emb, ws_emb], dim=-1))

    def forward(self, input_ids, memory, memory_mask=None, attention_mask=None,
                languages=None, writing_systems=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        lang = self.get_language_conditioning(languages, writing_systems, batch_size, device)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.token_embedding(input_ids) + self.position_embedding(positions) + lang.unsqueeze(1)
        embeddings = self.dropout(self.layer_norm(embeddings))

        memory = memory + lang.unsqueeze(1)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        tgt_padding = (attention_mask == 0) if attention_mask is not None else None

        output = self.transformer(
            tgt=embeddings,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding,
            memory_key_padding_mask=memory_mask
        )
        return self.output_projection(self.final_norm(output))


class EnhancedModel(nn.Module):
    """Enhanced graffiti recognition model: RTI/Korniienko visual tokens + transformer decoder."""

    def __init__(self, vocab_size, vision_dim=512, hidden_dim=512, num_layers=8,
                 num_languages=10, num_writing_systems=5, use_korniienko=True, max_length=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_korniienko = use_korniienko
        self.vision_encoder = EnhancedVisionEncoder(
            input_channels=12,
            hidden_dim=vision_dim,
            use_korniienko=use_korniienko
        )

        # Per-token projection into the decoder width
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.decoder = LanguageConditionedDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=8,
            ff_dim=hidden_dim * 4,
            max_length=max_length,
            num_languages=num_languages,
            num_writing_systems=num_writing_systems
        )

        self.model_type = "enhanced_v3"
        self.description = f"ResNet+FPN visual tokens + 8-layer transformer decoder (vocab:{vocab_size})"

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
        tokens, mask = memory
        return self.decoder(input_ids, tokens, memory_mask=mask, attention_mask=attention_mask,
                            languages=languages, writing_systems=writing_systems)

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'type': self.model_type,
            'description': self.description,
            'total_parameters': total_params,
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'vision_encoder_params': sum(p.numel() for p in self.vision_encoder.parameters()),
            'decoder_params': sum(p.numel() for p in self.decoder.parameters()),
            'korniienko_support': self.use_korniienko,
        }

#!/usr/bin/env python3
"""
Enhanced Graffiti Recognition Model - v2.0
===========================================

Advanced language-conditioned architecture with deep attention mechanisms.
Features:
- Multi-scale vision encoder with channel attention
- Language and writing system conditioning
- 8-layer transformer decoder with cross-modal attention
- Spatial attention for graffiti localization
- Advanced feature fusion and residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedVisionEncoder(nn.Module):
    """Enhanced vision encoder with channel attention and deep feature extraction."""
    
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
        
        # Initial feature extraction with residual connection
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks for deep feature extraction
        self.layer1 = self._make_layer(64, 128, 2, stride=1)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 384, 2, stride=2)
        self.layer4 = self._make_layer(384, 512, 2, stride=2)
        
        # Multi-scale feature fusion
        self.multiscale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2)
            ),
            nn.Sequential(
                nn.Conv2d(384, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=4)
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=8)
            )
        ])
        
        # Enhanced spatial attention for graffiti focus
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
        
        # Feature projection with residual connection
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Korniienko image encoders (optional for multi-modal fusion)
        if self.use_korniienko:
            # Encoder for Korniienko photo (3 RGB channels)
            self.korniienko_photo_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            
            # Encoder for Korniienko drawing (3 RGB channels)
            self.korniienko_drawing_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            
            # Fusion layer for combining RTI + photo + drawing features
            self.multimodal_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        
        self.output_dim = hidden_dim
    
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        """Create residual layer."""
        layers = []
        layers.append(ResidualBlock(inplanes, planes, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x, korniienko_photo=None, korniienko_drawing=None):
        """
        Forward pass with optional multi-modal fusion.
        
        Args:
            x: RTI images [batch_size, 12, height, width]
            korniienko_photo: Optional Korniienko photo [batch_size, 3, height, width]
            korniienko_drawing: Optional Korniienko drawing [batch_size, 3, height, width]
            
        Returns:
            features: Fused visual features [batch_size, hidden_dim]
        """
        # Apply channel attention to emphasize important image types
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        
        # Initial feature extraction
        x = self.initial_conv(x)  # [B, 64, H/4, W/4]
        
        # Deep feature extraction with residual connections
        x1 = self.layer1(x)   # [B, 128, H/4, W/4]
        x2 = self.layer2(x1)  # [B, 256, H/8, W/8]
        x3 = self.layer3(x2)  # [B, 384, H/16, W/16]
        x4 = self.layer4(x3)  # [B, 512, H/32, W/32]
        
        # Apply spatial attention to focus on graffiti regions
        attention = self.spatial_attention(x4)
        x4_attended = x4 * attention
        
        # Global feature extraction from RTI
        rti_features = self.feature_projection(x4_attended)
        
        # Multi-modal fusion if Korniienko images are available
        if self.use_korniienko and (korniienko_photo is not None or korniienko_drawing is not None):
            # Encode Korniienko photo (or use zeros if not available)
            if korniienko_photo is not None:
                photo_features = self.korniienko_photo_encoder(korniienko_photo)
            else:
                photo_features = torch.zeros_like(rti_features)
            
            # Encode Korniienko drawing (or use zeros if not available)
            if korniienko_drawing is not None:
                drawing_features = self.korniienko_drawing_encoder(korniienko_drawing)
            else:
                drawing_features = torch.zeros_like(rti_features)
            
            # Fuse all modalities: RTI + photo + drawing
            combined = torch.cat([rti_features, photo_features, drawing_features], dim=-1)
            features = self.multimodal_fusion(combined)
        else:
            features = rti_features
        
        return features

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

class LanguageConditionedDecoder(nn.Module):
    """Enhanced transformer decoder with language conditioning and cross-modal attention."""
    
    def __init__(self, vocab_size, hidden_dim=512, num_layers=8, num_heads=8, 
                 ff_dim=2048, max_length=128, num_languages=10, num_writing_systems=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Language and writing system embeddings
        self.language_embedding = nn.Embedding(num_languages, hidden_dim // 4)
        self.writing_system_embedding = nn.Embedding(num_writing_systems, hidden_dim // 4)
        
        # Language conditioning projection
        self.language_projection = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Enhanced token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        
        # Enhanced transformer layers with language conditioning
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
        
        # Cross-modal attention for vision-language fusion
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Enhanced output projection with residual connection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.15)
        
        # Language and writing system mappings
        self.language_map = {
            'church_slavonic': 0, 'ukrainian': 1, 'russian': 2, 'polish': 3,
            'ancient_greek': 4, 'armenian': 5, 'latin': 6, 'low_german': 7,
            'mixed': 8, 'unknown': 9
        }
        
        self.writing_system_map = {
            'cyrillic': 0, 'latin': 1, 'greek': 2, 'armenian': 3, 'unknown': 4
        }
    
    def get_language_conditioning(self, languages, writing_systems, device):
        """Generate language conditioning vectors.
        
        Args:
            languages: Tensor of language indices [batch_size] or None
            writing_systems: Tensor of writing system indices [batch_size] or None
            device: Device to create tensors on
        """
        # Handle None or empty inputs
        if languages is None or (isinstance(languages, torch.Tensor) and languages.numel() == 0):
            batch_size = 1
            conditioning = torch.zeros(batch_size, self.hidden_dim, device=device)
            return conditioning
        
        # Get batch size from tensor
        batch_size = languages.size(0) if isinstance(languages, torch.Tensor) else len(languages)
        
        # If languages and writing_systems are already tensors (indices), use them directly
        if isinstance(languages, torch.Tensor) and isinstance(writing_systems, torch.Tensor):
            lang_tensor = languages.to(device)
            ws_tensor = writing_systems.to(device)
            
            # Get embeddings
            lang_emb = self.language_embedding(lang_tensor)
            ws_emb = self.writing_system_embedding(ws_tensor)
            
            # Combine language and writing system embeddings
            combined = torch.cat([lang_emb, ws_emb], dim=-1)
            conditioning = self.language_projection(combined)
            
        else:
            # Legacy path: if they're string names, map them to indices
            lang_indices = []
            ws_indices = []
            
            for lang, ws in zip(languages, writing_systems):
                lang_norm = lang.lower().replace(' ', '_')
                ws_norm = ws.lower()
                
                lang_idx = self.language_map.get(lang_norm, self.language_map['unknown'])
                ws_idx = self.writing_system_map.get(ws_norm, self.writing_system_map['unknown'])
                
                lang_indices.append(lang_idx)
                ws_indices.append(ws_idx)
            
            lang_tensor = torch.tensor(lang_indices, device=device)
            ws_tensor = torch.tensor(ws_indices, device=device)
            
            # Get embeddings
            lang_emb = self.language_embedding(lang_tensor)
            ws_emb = self.writing_system_embedding(ws_tensor)
            
            # Combine language and writing system embeddings
            combined = torch.cat([lang_emb, ws_emb], dim=-1)
            conditioning = self.language_projection(combined)
        
        return conditioning
    
    def forward(self, input_ids, vision_features, attention_mask=None, 
                languages=None, writing_systems=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get language conditioning
        lang_conditioning = self.get_language_conditioning(languages, writing_systems, device)
        
        # Enhanced embeddings with language conditioning
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        # Add language conditioning to token embeddings
        embeddings = token_emb + pos_emb + lang_conditioning.unsqueeze(1)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Vision features as memory with language conditioning
        vision_memory = vision_features.unsqueeze(1) + lang_conditioning.unsqueeze(1)
        
        # Cross-modal attention for enhanced vision-text fusion
        enhanced_embeddings, _ = self.cross_modal_attention(
            query=embeddings,
            key=vision_memory,
            value=vision_memory
        )
        embeddings = embeddings + enhanced_embeddings
        
        # Generate causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        
        # Transformer forward with enhanced memory
        output = self.transformer(
            tgt=embeddings,
            memory=vision_memory,
            tgt_mask=causal_mask
        )
        
        return self.output_projection(output)

class EnhancedModel(nn.Module):
    """Enhanced graffiti recognition model with deep learning and language conditioning."""
    
    def __init__(self, vocab_size, vision_dim=512, hidden_dim=512, num_layers=8, 
                 num_languages=10, num_writing_systems=5, use_korniienko=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_korniienko = use_korniienko
        self.vision_encoder = EnhancedVisionEncoder(
            input_channels=12, 
            hidden_dim=vision_dim,
            use_korniienko=use_korniienko
        )
        
        # Enhanced vision projection with residual connection and layer norm
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Language-conditioned decoder with enhanced architecture
        self.decoder = LanguageConditionedDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=8,
            ff_dim=hidden_dim * 4,
            max_length=128,
            num_languages=num_languages,
            num_writing_systems=num_writing_systems
        )
        
        # Cross-modal fusion layer for vision-language integration
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # Model info
        self.model_type = "enhanced_v2"
        self.description = f"Deep vision + 8-layer transformer + language conditioning (vocab:{vocab_size})"
        
    def forward(self, images, input_ids, attention_mask=None, languages=None, writing_systems=None,
                korniienko_photo=None, korniienko_drawing=None):
        """
        Enhanced forward pass with language conditioning and multi-modal fusion.
        
        Args:
            images: RTI images [batch_size, 12, height, width]
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            languages: Language IDs [batch_size]
            writing_systems: Writing system IDs [batch_size]
            korniienko_photo: Optional Korniienko photo [batch_size, 3, height, width]
            korniienko_drawing: Optional Korniienko drawing [batch_size, 3, height, width]
            
        Returns:
            logits: Character predictions [batch_size, seq_length, vocab_size]
        """
        # Deep vision encoding with attention and multi-modal fusion
        vision_features = self.vision_encoder(
            images, 
            korniienko_photo=korniienko_photo,
            korniienko_drawing=korniienko_drawing
        )
        vision_features = self.vision_projection(vision_features)
        
        # Language-conditioned text generation
        logits = self.decoder(
            input_ids, 
            vision_features, 
            attention_mask,
            languages=languages,
            writing_systems=writing_systems
        )
        return logits
    
    def get_model_info(self):
        """Get comprehensive model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        features = [
            'Deep ResNet + Channel/Spatial Attention',
            'Korniienko reference images (photo + drawing)' if self.use_korniienko else None,
            '8-layer Transformer + Language Conditioning + Cross-Modal Attention',
            'Channel Attention + Spatial Attention + Cross-Modal Attention',
            'Dropout + LayerNorm + Residual Connections'
        ]
        
        return {
            'type': self.model_type,
            'description': self.description,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vision_encoder_params': vision_params,
            'decoder_params': decoder_params,
            'korniienko_support': self.use_korniienko,
            'vision_encoder': 'Deep ResNet + Channel/Spatial Attention',
            'decoder': '8-layer Transformer + Language Conditioning + Cross-Modal Attention',
            'language_conditioning': 'Language + Writing System Embeddings with Cross-Modal Fusion',
            'features': [f for f in features if f is not None],
            'attention_mechanisms': 'Channel Attention + Spatial Attention + Cross-Modal Attention',
            'regularization': 'Dropout + LayerNorm + Residual Connections'
        }

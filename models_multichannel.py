#!/usr/bin/env python3
"""
Multi-Channel Graffiti Recognition Model
========================================

Enhanced 12-channel vision processing with language conditioning for Saint Sophia graffiti.
Features:
- 12-channel input (original, blended, normal, texture × 3 RGB channels)
- Deep vision encoder with multi-scale features and attention
- Language and writing system conditioning
- 6-layer transformer decoder with cross-attention
- Adaptive image processing for variable sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiChannelVisionEncoder(nn.Module):
    """Enhanced 12-channel vision encoder with attention and multi-scale features."""
    
    def __init__(self, input_channels=12, hidden_dim=512):
        super().__init__()
        
        # Channel-specific processors for different image types
        # Each processes 3 RGB channels from one image type
        self.channel_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ) for _ in range(4)  # original, blended, normal, texture
        ])
        
        # Channel attention to weight different image types
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1),  # 4 types × 64 channels = 256
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 1),  # Attention weights for 4 image types
            nn.Sigmoid()
        )
        
        # Enhanced feature extractor with residual connections
        self.feature_extractor = nn.Sequential(
            # Initial conv
            nn.Conv2d(256, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 1 - Focus on fine details (important for graffiti)
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 2 - Medium scale features
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 3 - High-level features
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Block 4 - Abstract features
            nn.Conv2d(512, hidden_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Spatial attention for focusing on important regions
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # Multi-scale feature projection
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),  # Keep some spatial info
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.output_dim = hidden_dim
    
    def forward(self, x):
        """Forward pass through enhanced multi-channel vision encoder."""
        batch_size = x.size(0)
        
        # Process each image type separately (original, blended, normal, texture)
        processed_channels = []
        for i in range(4):
            # Extract 3 RGB channels for this image type
            start_idx = i * 3
            end_idx = start_idx + 3
            channel_input = x[:, start_idx:end_idx, :, :]
            
            # Process through dedicated processor
            processed = self.channel_processors[i](channel_input)
            processed_channels.append(processed)
        
        # Concatenate processed channels
        combined = torch.cat(processed_channels, dim=1)  # [batch, 256, H, W]
        
        # Apply channel attention
        channel_weights = self.channel_attention(combined)  # [batch, 4, 1, 1]
        
        # Apply attention weights to each image type
        weighted_channels = []
        for i in range(4):
            start_idx = i * 64
            end_idx = start_idx + 64
            channel_features = combined[:, start_idx:end_idx, :, :]
            weight = channel_weights[:, i:i+1, :, :]
            weighted = channel_features * weight
            weighted_channels.append(weighted)
        
        # Combine weighted features
        attended_features = torch.cat(weighted_channels, dim=1)
        
        # Extract hierarchical features
        features = self.feature_extractor(attended_features)
        
        # Apply spatial attention
        spatial_weights = self.spatial_attention(features)
        attended_features = features * spatial_weights
        
        # Project to final representation
        output = self.feature_projection(attended_features)
        
        return output

class LanguageConditionedDecoder(nn.Module):
    """Advanced transformer decoder with language and writing system conditioning."""
    
    def __init__(self, vocab_size, hidden_dim=512, num_layers=8, num_heads=8, 
                 ff_dim=2048, max_length=128, num_languages=10, num_writing_systems=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        
        # Language conditioning embeddings
        self.language_embedding = nn.Embedding(num_languages, hidden_dim // 2)
        self.writing_system_embedding = nn.Embedding(num_writing_systems, hidden_dim // 2)
        
        # Language conditioning projection
        self.language_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Vision-text cross-attention
        self.vision_text_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection with multiple stages for better learning
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, vocab_size)
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, vision_features, language_ids=None, writing_system_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        # Language conditioning
        if language_ids is not None and writing_system_ids is not None:
            lang_emb = self.language_embedding(language_ids).unsqueeze(1)  # [batch, 1, hidden_dim//2]
            ws_emb = self.writing_system_embedding(writing_system_ids).unsqueeze(1)  # [batch, 1, hidden_dim//2]
            
            # Combine language and writing system embeddings
            lang_condition = torch.cat([lang_emb, ws_emb], dim=-1)  # [batch, 1, hidden_dim]
            lang_condition = self.language_projection(lang_condition)
            
            # Broadcast language conditioning to all positions
            lang_condition = lang_condition.expand(-1, seq_len, -1)
        else:
            lang_condition = torch.zeros_like(token_emb)
        
        # Combine all embeddings
        embeddings = self.dropout(token_emb + pos_emb + lang_condition)
        
        # Vision features as memory for cross-attention
        vision_memory = vision_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Cross-attention between text and vision
        attended_embeddings, _ = self.vision_text_attention(
            query=embeddings,
            key=vision_memory,
            value=vision_memory
        )
        
        # Combine original embeddings with attended features
        embeddings = self.layer_norm(embeddings + attended_embeddings)
        
        # Generate causal mask for autoregressive generation
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        
        # Transformer forward pass
        output = self.transformer(
            tgt=embeddings,
            memory=vision_memory,
            tgt_mask=causal_mask
        )
        
        # Final output projection
        logits = self.output_projection(output)
        
        return logits

class MultiChannelModel(nn.Module):
    """Enhanced multi-channel graffiti recognition model with language conditioning."""
    
    def __init__(self, vocab_size, vision_dim=512, hidden_dim=512, num_layers=8, 
                 num_languages=10, num_writing_systems=5):
        super().__init__()
        
        # Enhanced vision encoder
        self.vision_encoder = MultiChannelVisionEncoder(input_channels=12, hidden_dim=vision_dim)
        
        # Vision projection
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Language-conditioned decoder
        self.decoder = LanguageConditionedDecoder(
            vocab_size=vocab_size, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            num_languages=num_languages,
            num_writing_systems=num_writing_systems
        )
        
        # Model info
        self.model_type = "multichannel"
        self.description = "Enhanced 12-channel vision + language-conditioned transformer"
        
        # Language and writing system mappings
        self.language_map = {
            'Church Slavonic': 0, 'Ukrainian': 1, 'Russian': 2, 'Polish': 3,
            'Ancient Greek': 4, 'Greek': 5, 'Armenian': 6, 'Latin': 7,
            'Low German': 8, 'unknown': 9
        }
        
        self.writing_system_map = {
            'Cyrillic': 0, 'Latin': 1, 'Greek': 2, 'Armenian': 3, 'unknown': 4
        }
        
    def encode_language_info(self, languages, writing_systems):
        """Convert language and writing system strings to IDs."""
        device = next(self.parameters()).device
        
        # Convert to IDs
        language_ids = torch.tensor([
            self.language_map.get(lang, self.language_map['unknown']) 
            for lang in languages
        ], device=device)
        
        writing_system_ids = torch.tensor([
            self.writing_system_map.get(ws, self.writing_system_map['unknown']) 
            for ws in writing_systems
        ], device=device)
        
        return language_ids, writing_system_ids
        
    def forward(self, images, input_ids, attention_mask=None, languages=None, writing_systems=None):
        """Forward pass with optional language conditioning."""
        
        # Vision encoding with multi-channel processing
        vision_features = self.vision_encoder(images)
        vision_features = self.vision_projection(vision_features)
        
        # Encode language information if provided
        language_ids = None
        writing_system_ids = None
        if languages is not None and writing_systems is not None:
            language_ids, writing_system_ids = self.encode_language_info(languages, writing_systems)
        
        # Text generation with language conditioning
        logits = self.decoder(
            input_ids=input_ids,
            vision_features=vision_features,
            language_ids=language_ids,
            writing_system_ids=writing_system_ids,
            attention_mask=attention_mask
        )
        
        return logits
    
    def generate(self, images, languages=None, writing_systems=None, max_length=50, 
                 bos_token_id=1, eos_token_id=2, temperature=1.0):
        """Generate text from images with optional language conditioning."""
        
        self.eval()
        device = images.device
        batch_size = images.size(0)
        
        # Vision encoding
        with torch.no_grad():
            vision_features = self.vision_encoder(images)
            vision_features = self.vision_projection(vision_features)
            
            # Encode language information
            language_ids = None
            writing_system_ids = None
            if languages is not None and writing_systems is not None:
                language_ids, writing_system_ids = self.encode_language_info(languages, writing_systems)
            
            # Initialize generation
            generated = torch.full((batch_size, 1), bos_token_id, device=device)
            
            for _ in range(max_length - 1):
                # Forward pass
                logits = self.decoder(
                    input_ids=generated,
                    vision_features=vision_features,
                    language_ids=language_ids,
                    writing_system_ids=writing_system_ids
                )
                
                # Get next token
                next_token_logits = logits[:, -1, :] / temperature
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check if all sequences ended
                if (next_token == eos_token_id).all():
                    break
            
            return generated
    
    def get_model_info(self):
        """Get enhanced model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'type': self.model_type,
            'description': self.description,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vision_encoder': 'Enhanced 12-channel CNN with attention',
            'decoder': 'Language-conditioned 6-layer Transformer',
            'language_conditioning': 'Language + Writing System embeddings',
            'supported_languages': list(self.language_map.keys()),
            'supported_writing_systems': list(self.writing_system_map.keys()),
            'features': [
                'Multi-channel image processing',
                'Channel-specific attention',
                'Spatial attention mechanism',
                'Language conditioning',
                'Cross-modal attention',
                'Multi-scale feature extraction'
            ]
        }

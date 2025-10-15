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
    """
    Enhanced 12-channel vision encoder with attention and multi-scale features.
    Supports optional Korniienko reference images (photo + drawing).
    """
    
    def __init__(self, input_channels=12, hidden_dim=512, use_korniienko=True):
        super().__init__()
        self.use_korniienko = use_korniienko
        
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
        
        # Enhanced spatial attention with multiple heads for better feature focus
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 8, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 8, 1, 1),
                nn.Sigmoid()
            ) for _ in range(4)  # Multiple attention heads
        ])
        
        # Self-attention for spatial feature refinement
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Multi-scale feature projection with residual connections
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # Keep more spatial info for graffiti details
            nn.Flatten(),
            nn.Linear(hidden_dim * 16, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Korniienko image encoders (if enabled)
        if self.use_korniienko:
            # Korniienko photo encoder (3 RGB channels)
            self.korniienko_photo_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, hidden_dim),
                nn.ReLU(inplace=True)
            )
            
            # Korniienko drawing encoder (3 channels or 1 grayscale)
            self.korniienko_drawing_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, hidden_dim),
                nn.ReLU(inplace=True)
            )
            
            # Fusion layer for combining RTI + Korniienko features
            self.multimodal_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),  # RTI + photo + drawing
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        
        self.output_dim = hidden_dim
    
    def forward(self, x, korniienko_photo=None, korniienko_drawing=None):
        """
        Forward pass through enhanced multi-channel vision encoder.
        
        Args:
            x: [batch, 12, H, W] - RTI images (4 types × 3 RGB)
            korniienko_photo: [batch, 3, H, W] - Optional Korniienko photograph
            korniienko_drawing: [batch, 3, H, W] - Optional Korniienko drawing
            
        Returns:
            output: [batch, hidden_dim] - Fused visual features
        """
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
        
        # Apply multiple spatial attention heads and combine
        spatial_attentions = []
        for attention_head in self.spatial_attention:
            attention_weights = attention_head(features)
            attended = features * attention_weights
            spatial_attentions.append(attended)
        
        # Combine multiple attention outputs
        multi_attended = torch.stack(spatial_attentions, dim=1).mean(dim=1)
        
        # Reshape for self-attention (spatial locations as sequence)
        batch_size, channels, height, width = multi_attended.shape
        spatial_tokens = multi_attended.view(batch_size, channels, height * width).permute(0, 2, 1)
        
        # Apply self-attention across spatial locations
        attended_spatial, _ = self.self_attention(spatial_tokens, spatial_tokens, spatial_tokens)
        
        # Reshape back and add residual connection
        attended_spatial = attended_spatial.permute(0, 2, 1).view(batch_size, channels, height, width)
        enhanced_features = multi_attended + attended_spatial
        
        # Project to final representation
        output = self.feature_projection(enhanced_features)
        
        # Integrate Korniienko features if available
        if self.use_korniienko and (korniienko_photo is not None or korniienko_drawing is not None):
            features_list = [output]
            
            # Encode Korniienko photo
            if korniienko_photo is not None:
                photo_features = self.korniienko_photo_encoder(korniienko_photo)
                features_list.append(photo_features)
            else:
                # Use zeros if photo not available
                features_list.append(torch.zeros_like(output))
            
            # Encode Korniienko drawing
            if korniienko_drawing is not None:
                drawing_features = self.korniienko_drawing_encoder(korniienko_drawing)
                features_list.append(drawing_features)
            else:
                # Use zeros if drawing not available
                features_list.append(torch.zeros_like(output))
            
            # Fuse all modalities
            combined_features = torch.cat(features_list, dim=-1)  # [batch, hidden_dim * 3]
            output = self.multimodal_fusion(combined_features)
        
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
        
        # Enhanced language conditioning embeddings with more capacity
        self.language_embedding = nn.Embedding(num_languages, hidden_dim // 2)
        self.writing_system_embedding = nn.Embedding(num_writing_systems, hidden_dim // 2)
        
        # Multi-layer language conditioning with cross-attention
        self.language_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-modal attention between vision and language
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Language conditioning projection with residual
        self.language_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
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
        
        # Enhanced output projection with skip connections and better regularization
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
        
        # Additional regularization layers
        self.feature_dropout = nn.Dropout(0.2)
        self.gradient_clipping = True
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, vision_features, language_ids=None, writing_system_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        # Language conditioning with enhanced fusion
        if language_ids is not None and writing_system_ids is not None:
            lang_emb = self.language_embedding(language_ids).unsqueeze(1)  # [batch, 1, hidden_dim//2]
            ws_emb = self.writing_system_embedding(writing_system_ids).unsqueeze(1)  # [batch, 1, hidden_dim//2]
            
            # Combine language and writing system embeddings
            lang_condition = torch.cat([lang_emb, ws_emb], dim=-1)  # [batch, 1, hidden_dim]
            
            # Enhanced language fusion with residual connection
            lang_enhanced = self.language_fusion(lang_condition)
            lang_condition = lang_condition + lang_enhanced
            
            # Project and broadcast to all positions
            lang_condition = self.language_projection(lang_condition)
            lang_condition = lang_condition.expand(-1, seq_len, -1)
        else:
            lang_condition = torch.zeros_like(token_emb)
        
        # Combine all embeddings with feature dropout
        embeddings = self.dropout(token_emb + pos_emb + lang_condition)
        embeddings = self.feature_dropout(embeddings)
        
        # Vision features as memory for cross-attention (expand for better context)
        vision_memory = vision_features.unsqueeze(1).expand(-1, 3, -1)  # [batch, 3, hidden_dim]
        
        # Enhanced cross-modal attention
        cross_attended, cross_attention_weights = self.cross_modal_attention(
            query=embeddings,
            key=vision_memory,
            value=vision_memory
        )
        
        # Vision-text cross-attention
        vision_attended, _ = self.vision_text_attention(
            query=embeddings + cross_attended,  # Combine both attention types
            key=vision_memory,
            value=vision_memory
        )
        
        # Multi-level attention fusion with residual connections
        embeddings = self.layer_norm(embeddings + cross_attended + vision_attended)
        
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
    """Enhanced multi-channel graffiti recognition model with language conditioning and Korniienko support."""
    
    def __init__(self, vocab_size, vision_dim=512, hidden_dim=512, num_layers=8, 
                 num_languages=10, num_writing_systems=5, use_korniienko=True):
        super().__init__()
        
        # Enhanced vision encoder with Korniienko support
        self.vision_encoder = MultiChannelVisionEncoder(
            input_channels=12, 
            hidden_dim=vision_dim,
            use_korniienko=use_korniienko
        )
        self.use_korniienko = use_korniienko
        
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
        
    def forward(self, images, input_ids, attention_mask=None, languages=None, writing_systems=None,
                korniienko_photo=None, korniienko_drawing=None):
        """
        Forward pass with optional language conditioning and Korniienko images.
        
        Args:
            images: [batch, 12, H, W] - RTI images (4 types × 3 RGB)
            input_ids: [batch, seq_len] - Token IDs for decoding
            attention_mask: [batch, seq_len] - Attention mask
            languages: List of language strings
            writing_systems: List of writing system strings
            korniienko_photo: [batch, 3, H, W] - Optional Korniienko photograph
            korniienko_drawing: [batch, 3, H, W] - Optional Korniienko drawing
            
        Returns:
            logits: [batch, seq_len, vocab_size] - Output logits
        """
        
        # Vision encoding with multi-channel processing + Korniienko
        vision_features = self.vision_encoder(
            images, 
            korniienko_photo=korniienko_photo,
            korniienko_drawing=korniienko_drawing
        )
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
                 bos_token_id=1, eos_token_id=2, temperature=1.0,
                 korniienko_photo=None, korniienko_drawing=None):
        """
        Generate text from images with optional language conditioning and Korniienko images.
        
        Args:
            images: [batch, 12, H, W] - RTI images
            languages: List of language strings
            writing_systems: List of writing system strings
            max_length: Maximum generation length
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            temperature: Sampling temperature
            korniienko_photo: [batch, 3, H, W] - Optional Korniienko photograph
            korniienko_drawing: [batch, 3, H, W] - Optional Korniienko drawing
            
        Returns:
            generated: [batch, seq_len] - Generated token IDs
        """
        
        self.eval()
        device = images.device
        batch_size = images.size(0)
        
        # Vision encoding
        with torch.no_grad():
            vision_features = self.vision_encoder(
                images,
                korniienko_photo=korniienko_photo,
                korniienko_drawing=korniienko_drawing
            )
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
            'decoder': 'Language-conditioned 8-layer Transformer',
            'language_conditioning': 'Language + Writing System embeddings',
            'korniienko_support': self.use_korniienko,
            'supported_languages': list(self.language_map.keys()),
            'supported_writing_systems': list(self.writing_system_map.keys()),
            'features': [
                'Multi-channel RTI image processing',
                'Korniienko reference images (photo + drawing)' if self.use_korniienko else None,
                'Channel-specific attention',
                'Spatial attention mechanism',
                'Language conditioning',
                'Cross-modal attention',
                'Multi-scale feature extraction',
                'Multi-modal fusion'
            ]
        }
        
        # Remove None values from features
        return {k: [f for f in v if f is not None] if isinstance(v, list) else v 
                for k, v in info.items()}

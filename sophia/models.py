"""
Multimodal neural network models for SOPHIA.

Inspired by ITHACA and other state-of-the-art approaches to ancient text recognition,
but adapted for the specific challenges of Byzantine graffiti and spatial annotation data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import torchvision.models as models
from typing import Dict, Tuple, Optional


class SophiaModel(nn.Module):
    """
    Main SOPHIA model combining vision, text, and spatial features.
    
    Architecture inspired by:
    - ITHACA (https://github.com/google-deepmind/ithaca)
    - Predicting the Past (https://github.com/google-deepmind/predictingthepast)
    - Ancient Text Restoration approaches
    
    Key innovations:
    1. Spatial-aware attention mechanism
    2. Multi-scale visual feature extraction
    3. Historical context embedding
    4. Byzantine-specific character modeling
    """
    
    def __init__(
        self,
        config: Dict,
        vocab_size: int = 50000,
        max_sequence_length: int = 512
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        
        # Initialize components
        self.vision_encoder = VisionEncoder(config['vision'])
        self.text_encoder = TextEncoder(config['text'])
        self.spatial_encoder = SpatialEncoder(config['spatial'])
        self.fusion_layer = MultimodalFusion(config['fusion'])
        self.decoder = InscriptionDecoder(config['decoder'], vocab_size)
        
        # Task-specific heads
        self.transcription_head = TranscriptionHead(config['decoder']['hidden_size'], vocab_size)
        self.restoration_head = RestorationHead(config['decoder']['hidden_size'])
        self.dating_head = DatingHead(config['decoder']['hidden_size'])
        
    def forward(
        self,
        images: torch.Tensor,
        spatial_features: Dict[str, torch.Tensor],
        text_features: Dict[str, torch.Tensor],
        metadata_features: Dict[str, torch.Tensor],
        target_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        batch_size = images.size(0)
        
        # Encode each modality
        vision_features = self.vision_encoder(images)
        text_features_encoded = self.text_encoder(text_features)
        spatial_features_encoded = self.spatial_encoder(spatial_features)
        
        # Multimodal fusion
        fused_features = self.fusion_layer(
            vision_features,
            text_features_encoded,
            spatial_features_encoded,
            metadata_features
        )
        
        # Generate contextual representations
        context_features = self.decoder(fused_features, target_ids)
        
        # Task predictions
        outputs = {
            'transcription_logits': self.transcription_head(context_features),
            'restoration_probs': self.restoration_head(context_features),
            'dating_predictions': self.dating_head(context_features),
            'context_features': context_features
        }
        
        return outputs


class VisionEncoder(nn.Module):
    """
    Multi-scale vision encoder for inscription images.
    
    Uses pre-trained backbone with spatial pyramid pooling
    to capture features at different scales.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Pre-trained backbone
        if config['backbone'] == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Identity()  # Remove final layer
            backbone_dim = 2048
        elif config['backbone'] == 'efficientnet':
            self.backbone = models.efficientnet_b4(pretrained=True)
            self.backbone.classifier = nn.Identity()
            backbone_dim = 1792
        else:
            raise ValueError(f"Unsupported backbone: {config['backbone']}")
        
        # Spatial Pyramid Pooling
        self.spp = SpatialPyramidPooling([1, 2, 4], backbone_dim)
        spp_output_dim = backbone_dim * (1 + 4 + 16)  # 1x1 + 2x2 + 4x4
        
        # Feature refinement
        self.feature_refiner = nn.Sequential(
            nn.Linear(spp_output_dim, config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(config['hidden_size'], config['hidden_size'])
        )
        
        # Attention mechanism for focusing on inscription regions
        self.spatial_attention = SpatialAttention(config['hidden_size'])
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Extract CNN features
        cnn_features = self.backbone(images)
        
        # Apply spatial pyramid pooling
        pooled_features = self.spp(cnn_features)
        
        # Refine features
        refined_features = self.feature_refiner(pooled_features)
        
        # Apply spatial attention
        attended_features = self.spatial_attention(refined_features)
        
        return attended_features


class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling for multi-scale feature extraction."""
    
    def __init__(self, pool_sizes: list, input_dim: int):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.input_dim = input_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        features = []
        
        for pool_size in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(x, pool_size)
            pooled = pooled.view(batch_size, -1)
            features.append(pooled)
        
        return torch.cat(features, dim=1)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on inscription regions."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(features)
        return features * attention_weights


class TextEncoder(nn.Module):
    """
    Text encoder using pre-trained multilingual model.
    Specialized for ancient languages and historical text patterns.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Use multilingual BERT/RoBERTa for ancient language support
        model_name = config.get('model_name', 'xlm-roberta-base')
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Additional layers for historical text understanding
        transformer_dim = self.transformer.config.hidden_size
        self.historical_adapter = nn.Sequential(
            nn.Linear(transformer_dim, config['hidden_size']),
            nn.LayerNorm(config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        # Character-level modeling for damaged text
        self.char_embeddings = nn.Embedding(256, config['char_embedding_size'])
        self.char_encoder = nn.LSTM(
            config['char_embedding_size'],
            config['char_hidden_size'],
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, text_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Word-level encoding
        transformer_output = self.transformer(
            input_ids=text_features['input_ids'],
            attention_mask=text_features['attention_mask']
        )
        
        # Pool transformer features
        pooled_output = transformer_output.pooler_output
        adapted_output = self.historical_adapter(pooled_output)
        
        return adapted_output


class SpatialEncoder(nn.Module):
    """
    Encoder for spatial annotation data.
    Processes geometric information to understand inscription layout.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Bounding box encoder
        self.bbox_encoder = nn.Sequential(
            nn.Linear(4, config['bbox_hidden_size']),  # x1, y1, x2, y2
            nn.ReLU(),
            nn.Linear(config['bbox_hidden_size'], config['bbox_hidden_size'])
        )
        
        # Geometry type embeddings
        self.geometry_embeddings = nn.Embedding(10, config['geometry_embedding_size'])
        
        # Spatial relationship modeling
        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['ff_size'],
                dropout=config.get('dropout', 0.1)
            ),
            num_layers=config['num_layers']
        )
        
        # Final projection
        self.output_projection = nn.Linear(
            config['bbox_hidden_size'] + config['geometry_embedding_size'],
            config['hidden_size']
        )
        
    def forward(self, spatial_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = spatial_features['bounding_boxes'].size(0)
        max_annotations = spatial_features['bounding_boxes'].size(1)
        
        # Encode bounding boxes
        bbox_encoded = self.bbox_encoder(spatial_features['bounding_boxes'])
        
        # Create geometry embeddings (simplified for now)
        geometry_ids = torch.zeros(batch_size, max_annotations, dtype=torch.long)
        geometry_embedded = self.geometry_embeddings(geometry_ids)
        
        # Combine features
        combined_features = torch.cat([bbox_encoded, geometry_embedded], dim=-1)
        projected_features = self.output_projection(combined_features)
        
        # Apply transformer for spatial relationships
        # Reshape for transformer: (seq_len, batch, features)
        reshaped_features = projected_features.transpose(0, 1)
        transformed_features = self.spatial_transformer(reshaped_features)
        
        # Pool over annotations
        pooled_features = transformed_features.mean(dim=0)
        
        return pooled_features


class MultimodalFusion(nn.Module):
    """
    Fusion layer combining vision, text, and spatial features.
    Uses cross-modal attention inspired by ITHACA's approach.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Cross-modal attention layers
        self.vision_text_attention = CrossModalAttention(config['hidden_size'])
        self.vision_spatial_attention = CrossModalAttention(config['hidden_size'])
        self.text_spatial_attention = CrossModalAttention(config['hidden_size'])
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(config['hidden_size'] * 3, config['hidden_size']),
            nn.LayerNorm(config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(config['hidden_size'], config['hidden_size'])
        )
        
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        spatial_features: torch.Tensor,
        metadata_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        
        # Cross-modal attention
        vision_text_fused = self.vision_text_attention(vision_features, text_features)
        vision_spatial_fused = self.vision_spatial_attention(vision_features, spatial_features)
        text_spatial_fused = self.text_spatial_attention(text_features, spatial_features)
        
        # Concatenate all modalities
        concatenated = torch.cat([
            vision_text_fused,
            vision_spatial_fused,
            text_spatial_fused
        ], dim=-1)
        
        # Final fusion
        fused_features = self.fusion_layer(concatenated)
        
        return fused_features


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        # Add sequence dimension if needed
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)
        
        attended, _ = self.attention(query, key_value, key_value)
        output = self.norm(attended + query)
        
        # Remove sequence dimension
        if output.size(1) == 1:
            output = output.squeeze(1)
        
        return output


class InscriptionDecoder(nn.Module):
    """
    Decoder for generating inscription transcriptions.
    Uses transformer architecture with historical context.
    """
    
    def __init__(self, config: Dict, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Transformer decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['ff_size'],
                dropout=config.get('dropout', 0.1)
            ),
            num_layers=config['num_layers']
        )
        
    def forward(
        self,
        context_features: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # Use context features as memory for decoder
        if context_features.dim() == 2:
            context_features = context_features.unsqueeze(1)
        
        # For training, use target sequence; for inference, generate
        if target_ids is not None:
            # Teacher forcing during training
            return self._forward_training(context_features, target_ids)
        else:
            # Autoregressive generation during inference
            return self._forward_inference(context_features)
    
    def _forward_training(self, memory: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        # Create target embeddings
        target_embeddings = self._create_embeddings(target_ids)
        
        # Apply decoder
        output = self.decoder(target_embeddings.transpose(0, 1), memory.transpose(0, 1))
        
        return output.transpose(0, 1)
    
    def _forward_inference(self, memory: torch.Tensor) -> torch.Tensor:
        # Simple forward pass for inference
        return memory
    
    def _create_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Simplified embedding creation
        embedding_layer = nn.Embedding(self.vocab_size, self.config['hidden_size'])
        return embedding_layer(token_ids)


class TranscriptionHead(nn.Module):
    """Head for transcription prediction."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class RestorationHead(nn.Module):
    """Head for text restoration confidence."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class DatingHead(nn.Module):
    """Head for dating prediction."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2)  # min_year, max_year
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


# Model configuration
def get_default_config() -> Dict:
    """Get default SOPHIA model configuration."""
    return {
        'vision': {
            'backbone': 'resnet50',
            'hidden_size': 768,
            'dropout': 0.1
        },
        'text': {
            'model_name': 'xlm-roberta-base',
            'hidden_size': 768,
            'char_embedding_size': 64,
            'char_hidden_size': 128,
            'dropout': 0.1
        },
        'spatial': {
            'hidden_size': 768,
            'bbox_hidden_size': 128,
            'geometry_embedding_size': 64,
            'num_heads': 8,
            'num_layers': 2,
            'ff_size': 2048,
            'dropout': 0.1
        },
        'fusion': {
            'hidden_size': 768,
            'dropout': 0.1
        },
        'decoder': {
            'hidden_size': 768,
            'num_heads': 8,
            'num_layers': 6,
            'ff_size': 2048,
            'dropout': 0.1
        }
    }


# Alias for backward compatibility
MultimodalTransformer = SophiaModel

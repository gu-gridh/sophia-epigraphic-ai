"""
Inference utilities for SOPHIA model.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple
import json

from .models import SophiaModel
from .data import ImageAnnotationProcessor


class InscriptionReader:
    """
    Main inference class for reading inscriptions with SOPHIA.
    """
    
    def __init__(
        self,
        model_path: str,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.config = config
        
        # Load model
        self.model = SophiaModel(config['model'], config['vocab_size'])
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Initialize tokenizer if needed
        self.tokenizer = None  # Would initialize based on config
        
    def predict(
        self,
        image_path: str,
        annotations: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Predict transcription for an inscription.
        
        Args:
            image_path: Path to inscription image
            annotations: Path to annotation JSON file or annotation data
            metadata: Additional metadata about the inscription
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        
        # Load and preprocess image
        image_tensor = self._preprocess_image(image_path)
        
        # Load spatial features
        spatial_features = self._preprocess_annotations(annotations)
        
        # Create dummy text features (for inference)
        text_features = self._create_dummy_text_features()
        
        # Process metadata
        metadata_features = self._preprocess_metadata(metadata or {})
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(
                images=image_tensor.unsqueeze(0).to(self.device),
                spatial_features=self._to_device(spatial_features),
                text_features=self._to_device(text_features),
                metadata_features=self._to_device(metadata_features)
            )
        
        # Process outputs
        results = self._process_outputs(outputs)
        
        return results
    
    def predict_batch(
        self,
        image_paths: List[str],
        annotations_paths: List[Optional[str]] = None,
        metadata_list: List[Optional[Dict]] = None,
        batch_size: int = 8
    ) -> List[Dict]:
        """Predict transcriptions for multiple inscriptions."""
        
        if annotations_paths is None:
            annotations_paths = [None] * len(image_paths)
        if metadata_list is None:
            metadata_list = [None] * len(image_paths)
        
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_images = image_paths[i:i+batch_size]
            batch_annotations = annotations_paths[i:i+batch_size]
            batch_metadata = metadata_list[i:i+batch_size]
            
            batch_results = self._predict_batch_internal(
                batch_images, batch_annotations, batch_metadata
            )
            results.extend(batch_results)
        
        return results
    
    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input."""
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            # Apply same transforms as training (without augmentation)
            # This would use the same transform pipeline as the dataset
            # For now, simple resize and normalize
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            return transform(image)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Return placeholder
            return torch.zeros(3, 224, 224)
    
    def _preprocess_annotations(self, annotations) -> Dict[str, torch.Tensor]:
        """Preprocess spatial annotation data."""
        
        if annotations is None:
            return self._empty_spatial_features()
        
        if isinstance(annotations, str):
            # Load from file
            try:
                with open(annotations, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
            except Exception as e:
                print(f"Error loading annotations: {e}")
                return self._empty_spatial_features()
        else:
            annotation_data = annotations
        
        # Process similar to dataset
        return self._process_spatial_annotations(annotation_data)
    
    def _process_spatial_annotations(self, annotations) -> Dict[str, torch.Tensor]:
        """Process annotation data into tensor format."""
        if not isinstance(annotations, list):
            annotations = [annotations]
        
        # Extract bounding boxes
        bboxes = []
        for ann in annotations:
            if 'geometry' in ann and 'coordinates' in ann['geometry']:
                bbox = self._extract_bbox(ann['geometry'])
                bboxes.append(bbox)
        
        # Create tensor
        max_annotations = 10
        bbox_tensor = torch.zeros(max_annotations, 4)
        
        for i, bbox in enumerate(bboxes[:max_annotations]):
            bbox_tensor[i] = torch.tensor(bbox, dtype=torch.float32)
        
        return {
            'num_annotations': torch.tensor(len(bboxes), dtype=torch.long),
            'bounding_boxes': bbox_tensor,
            'has_annotations': torch.tensor(len(bboxes) > 0, dtype=torch.bool)
        }
    
    def _extract_bbox(self, geometry) -> List[float]:
        """Extract bounding box from geometry."""
        coords = geometry['coordinates']
        if not coords:
            return [0, 0, 0, 0]
        
        if geometry['type'] == 'Polygon':
            if coords and len(coords) > 0:
                points = coords[0]
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                return [min(xs), min(ys), max(xs), max(ys)]
        elif geometry['type'] == 'Point':
            x, y = coords[:2]
            return [x, y, x, y]
        
        return [0, 0, 0, 0]
    
    def _empty_spatial_features(self) -> Dict[str, torch.Tensor]:
        """Return empty spatial features."""
        return {
            'num_annotations': torch.tensor(0, dtype=torch.long),
            'bounding_boxes': torch.zeros(10, 4),
            'has_annotations': torch.tensor(False, dtype=torch.bool)
        }
    
    def _create_dummy_text_features(self) -> Dict[str, torch.Tensor]:
        """Create dummy text features for inference."""
        max_length = 512
        return {
            'input_ids': torch.zeros(max_length, dtype=torch.long),
            'attention_mask': torch.zeros(max_length, dtype=torch.long)
        }
    
    def _preprocess_metadata(self, metadata: Dict) -> Dict[str, torch.Tensor]:
        """Preprocess metadata features."""
        features = {}
        
        # Simple encoding for now
        for key, value in metadata.items():
            if isinstance(value, (int, float)):
                features[key] = torch.tensor(float(value), dtype=torch.float32)
            else:
                # Simple hash encoding for categorical
                features[key] = torch.tensor(hash(str(value)) % 1000, dtype=torch.long)
        
        return features
    
    def _to_device(self, features) -> Dict:
        """Move features to device."""
        if isinstance(features, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in features.items()}
        elif isinstance(features, torch.Tensor):
            return features.to(self.device)
        else:
            return features
    
    def _predict_batch_internal(
        self,
        image_paths: List[str],
        annotations_paths: List[Optional[str]],
        metadata_list: List[Optional[Dict]]
    ) -> List[Dict]:
        """Internal batch prediction."""
        
        batch_images = []
        batch_spatial = []
        batch_text = []
        batch_metadata = []
        
        # Preprocess batch
        for i, image_path in enumerate(image_paths):
            image_tensor = self._preprocess_image(image_path)
            spatial_features = self._preprocess_annotations(annotations_paths[i])
            text_features = self._create_dummy_text_features()
            metadata_features = self._preprocess_metadata(metadata_list[i] or {})
            
            batch_images.append(image_tensor)
            batch_spatial.append(spatial_features)
            batch_text.append(text_features)
            batch_metadata.append(metadata_features)
        
        # Stack tensors
        batch_images = torch.stack(batch_images).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(
                images=batch_images,
                spatial_features=self._batch_spatial_features(batch_spatial),
                text_features=self._batch_text_features(batch_text),
                metadata_features=self._batch_metadata_features(batch_metadata)
            )
        
        # Process outputs for each item in batch
        results = []
        batch_size = len(image_paths)
        
        for i in range(batch_size):
            item_outputs = {k: v[i] if v.dim() > 0 else v for k, v in outputs.items()}
            result = self._process_outputs(item_outputs)
            results.append(result)
        
        return results
    
    def _batch_spatial_features(self, batch_spatial: List[Dict]) -> Dict:
        """Batch spatial features."""
        return {
            'num_annotations': torch.stack([f['num_annotations'] for f in batch_spatial]).to(self.device),
            'bounding_boxes': torch.stack([f['bounding_boxes'] for f in batch_spatial]).to(self.device),
            'has_annotations': torch.stack([f['has_annotations'] for f in batch_spatial]).to(self.device)
        }
    
    def _batch_text_features(self, batch_text: List[Dict]) -> Dict:
        """Batch text features."""
        return {
            'input_ids': torch.stack([f['input_ids'] for f in batch_text]).to(self.device),
            'attention_mask': torch.stack([f['attention_mask'] for f in batch_text]).to(self.device)
        }
    
    def _batch_metadata_features(self, batch_metadata: List[Dict]) -> Dict:
        """Batch metadata features."""
        # Simple implementation - would need proper handling
        return {}
    
    def _process_outputs(self, outputs: Dict) -> Dict:
        """Process model outputs into readable results."""
        
        results = {
            'transcription': '',
            'confidence': 0.0,
            'restoration_confidence': 0.0,
            'predicted_dating': None,
            'features': None
        }
        
        # Process transcription logits
        if 'transcription_logits' in outputs:
            transcription_probs = F.softmax(outputs['transcription_logits'], dim=-1)
            # Convert to text (would need proper tokenizer)
            results['transcription'] = self._decode_transcription(transcription_probs)
            results['confidence'] = transcription_probs.max().item()
        
        # Process restoration confidence
        if 'restoration_probs' in outputs:
            results['restoration_confidence'] = outputs['restoration_probs'].item()
        
        # Process dating predictions
        if 'dating_predictions' in outputs:
            dating = outputs['dating_predictions']
            if dating.numel() >= 2:
                results['predicted_dating'] = {
                    'min_year': dating[0].item(),
                    'max_year': dating[1].item()
                }
        
        # Store features for analysis
        if 'context_features' in outputs:
            results['features'] = outputs['context_features'].cpu().numpy()
        
        return results
    
    def _decode_transcription(self, probs: torch.Tensor) -> str:
        """Decode transcription probabilities to text."""
        # Simplified decoding - would need proper tokenizer and vocabulary
        predicted_ids = probs.argmax(dim=-1)
        
        # Convert to text (placeholder)
        # In real implementation, would use tokenizer.decode()
        text = f"Predicted transcription (simplified): {predicted_ids.tolist()[:10]}"
        
        return text


def load_model_for_inference(model_path: str, config_path: str) -> InscriptionReader:
    """Load SOPHIA model for inference."""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return InscriptionReader(model_path, config)

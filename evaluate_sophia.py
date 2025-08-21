#!/usr/bin/env python3
"""
Enhanced Evaluation Suite for Saint Sophia Graffiti Recognition Models
Evaluates trained models with language conditioning and saves detailed CSV results.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import XLMRobertaTokenizer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from datetime import datetime
from tqdm import tqdm
import re

from models_multichannel import MultiChannelModel
from models_enhanced import EnhancedModel
from train_sophia import SophiaDataset

class SophiaEvaluator:
    def __init__(self, device='cuda'):
        """Initialize the enhanced evaluator with language conditioning support."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔍 Evaluation device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        print(f"📝 Tokenizer loaded with vocab size: {len(self.tokenizer)}")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Model configurations
        self.models = {
            'enhanced': {
                'path': 'models/best_enhanced_model.pth',
                'class': EnhancedModel,
                'description': 'Enhanced model v2.0 with deep learning and language conditioning'
            },
            'multichannel': {
                'path': 'models/best_multichannel_model.pth', 
                'class': MultiChannelModel,
                'description': 'Multi-channel model with language conditioning and 12-channel vision'
            }
        }
        
        # Dataset configurations
        self.datasets = {
            'test': {
                'name': 'test',
                'csv_path': 'data/test_dataset.csv',
                'images_path': 'data/cropped_images/test'
            },
            'validation': {
                'name': 'validation',
                'csv_path': 'data/val_dataset.csv',
                'images_path': 'data/cropped_images/val'
            }
        }
    
    def load_model(self, model_name):
        """Load a specific model with proper configuration."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_config = self.models[model_name]
        model_path = model_config['path']
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None
            
        print(f"📦 Loading {model_name} from {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            vocab_size = len(self.tokenizer)
            
            # Initialize model with correct parameters for our enhanced architectures
            if model_name == 'enhanced':
                # Enhanced model with the parameters it was actually trained with
                model = model_config['class'](
                    vocab_size=vocab_size,
                    vision_dim=256,
                    hidden_dim=256,
                    num_layers=6,
                    num_languages=10,
                    num_writing_systems=5
                )
            elif model_name == 'multichannel':
                # Initialize with language conditioning parameters (8 layers as per trained model)
                model = model_config['class'](
                    vocab_size=vocab_size, 
                    vision_dim=512, 
                    hidden_dim=512, 
                    num_layers=8,
                    num_languages=10,
                    num_writing_systems=5
                )
            else:
                print(f"⏭️ Skipping {model_name} (unknown architecture)")
                return None
                
            model = model.to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Get model info
            model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
            params = model_info.get('total_parameters', 'unknown')
            
            print(f"✅ {model_name} loaded successfully")
            print(f"   Parameters: {params:,}" if isinstance(params, int) else f"   Parameters: {params}")
            print(f"   Description: {model_config['description']}")
            
            return model, checkpoint
            
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            return None
    
    def create_dataset(self, model_name, dataset_path, images_path):
        """Create dataset for a specific model with proper configuration."""
        try:
            print(f"📊 Creating dataset for {model_name}:")
            print(f"   CSV: {dataset_path}")
            print(f"   Images: {images_path}")
            print(f"   Model type: {model_name}")
            
            # Determine model type for dataset creation
            model_type = 'multichannel' if model_name == 'multichannel' else 'enhanced'
            
            # Create image transforms
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = SophiaDataset(
                csv_file=dataset_path,
                cropped_images_dir=images_path,
                tokenizer=self.tokenizer,
                max_length=128,
                transform=transform,
                model_type=model_type
            )
            
            print(f"✅ Dataset created: {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            print(f"❌ Error creating dataset for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_text_with_language_conditioning(self, model, images, languages=None, writing_systems=None, max_length=50):
        """Generate text using autoregressive decoding with temperature and language conditioning."""
        batch_size = images.shape[0]
        device = images.device
        
        # Generation parameters for more diverse output
        temperature = 0.7
        top_p = 0.9
        
        generated_sequences = []
        
        for i in range(batch_size):
            image = images[i:i+1]  # Single image
            
            # Get language info for this sample
            sample_language = languages[i] if languages else 'unknown'
            sample_writing_system = writing_systems[i] if writing_systems else 'unknown'
            
            # Start with language prefix to better condition the model
            if sample_language != 'unknown':
                lang_prefix = f"[{sample_language.upper()}]"
                prefix_tokens = self.tokenizer(lang_prefix, return_tensors='pt')['input_ids'][0]
                generated_ids = [self.tokenizer.bos_token_id] + prefix_tokens.tolist()
            else:
                generated_ids = [self.tokenizer.bos_token_id]
            
            for step in range(max_length):
                # Prepare input
                input_tensor = torch.tensor([generated_ids], device=device)
                
                # Forward pass with language conditioning
                with torch.no_grad():
                    try:
                        if hasattr(model, 'forward') and languages and writing_systems:
                            # Enhanced v2.0 or Multichannel model with language conditioning
                            logits = model(
                                image, input_tensor, None,
                                languages=[sample_language],
                                writing_systems=[sample_writing_system]
                            )
                        else:
                            # Legacy model or fallback
                            logits = model(image, input_tensor)
                        
                        # Apply temperature and top-p sampling for diversity
                        last_logits = logits[0, -1, :] / temperature
                        
                        # Top-p (nucleus) sampling
                        sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                        last_logits[indices_to_remove] = -float('Inf')
                        
                        # Sample from the filtered distribution
                        probs = F.softmax(last_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                        
                    except Exception as e:
                        print(f"  ⚠️ Generation error at step {step}: {e}")
                        # Fallback to greedy if there's an error
                        try:
                            logits = model(image, input_tensor)
                            next_token = torch.argmax(logits[0, -1, :]).item()
                        except:
                            break
                
                # Stop at EOS or special tokens
                if next_token == self.tokenizer.eos_token_id:
                    break
                    
                # Prevent infinite loops of the same token
                if len(generated_ids) > 5 and generated_ids[-5:] == [next_token] * 5:
                    break
                    
                generated_ids.append(next_token)
            
            generated_sequences.append(generated_ids)
        
        return generated_sequences
    
    def predict_batch(self, model, dataloader, model_name, dataset):
        """Generate predictions with enhanced language conditioning support."""
        predictions = []
        ground_truths = []
        metadata = []
        
        print(f"🔮 Generating graffiti transcriptions for {model_name}...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
                try:
                    images = batch['image'].to(self.device)
                    
                    # Get language information if available
                    languages = batch.get('language', None)
                    writing_systems = batch.get('writing_system', None)
                    
                    # Generate predictions using autoregressive decoding with language conditioning
                    generated_sequences = self.generate_text_with_language_conditioning(
                        model, images, languages, writing_systems, max_length=50
                    )
                    
                    # Process each sample in batch
                    for i in range(len(batch['transcription']) if 'transcription' in batch else len(generated_sequences)):
                        # Decode generated text
                        if i < len(generated_sequences):
                            pred_text = self.tokenizer.decode(generated_sequences[i], skip_special_tokens=True)
                        else:
                            pred_text = ""
                        
                        # Clean prediction (remove language prefixes)
                        clean_pred = self.clean_prediction(pred_text)
                        predictions.append(clean_pred)
                        
                        # Get ground truth transcription
                        if 'transcription' in batch and i < len(batch['transcription']):
                            gt = batch['transcription'][i]
                        else:
                            gt = ""
                        ground_truths.append(gt)
                        
                        # Get metadata
                        if 'annotation_id' in batch and i < len(batch['annotation_id']):
                            ann_id = batch['annotation_id'][i].item() if torch.is_tensor(batch['annotation_id'][i]) else batch['annotation_id'][i]
                            language = languages[i] if languages else 'unknown'
                            writing_system = writing_systems[i] if writing_systems else 'unknown'
                        else:
                            ann_id = f"batch_{batch_idx}_sample_{i}"
                            language = 'unknown'
                            writing_system = 'unknown'
                        
                        meta = {
                            'annotation_id': str(ann_id),
                            'language': language,
                            'writing_system': writing_system,
                            'model_type': model_name,
                            'image_channels': images.shape[1],
                            'image_size': f"{images.shape[2]}x{images.shape[3]}"
                        }
                        metadata.append(meta)
                        
                except Exception as e:
                    print(f"⚠️ Error in batch {batch_idx}: {e}")
                    # Add empty predictions for failed batch
                    batch_size = len(batch.get('annotation_id', [1]))
                    for i in range(batch_size):
                        predictions.append("")
                        ground_truths.append("")
                        metadata.append({
                            'annotation_id': f"error_{batch_idx}_{i}",
                            'language': 'error',
                            'writing_system': 'error',
                            'model_type': model_name,
                            'image_channels': 'error',
                            'image_size': 'error'
                        })
        
        return predictions, ground_truths, metadata
    
    def clean_prediction(self, prediction):
        """Clean prediction text by removing language prefixes and conditioning tokens."""
        if not prediction:
            return ""
        
        import re
        clean_text = prediction.strip()
        
        # Remove bracket-style language prefixes like [CHURCH SLAVONIC], [UKRAINIAN], etc.
        # This pattern handles variations like ][CYRILLIC] or incomplete brackets
        bracket_pattern = r'[\[\]]*[A-Z\s_]*[\[\]]+'
        clean_text = re.sub(bracket_pattern, '', clean_text)
        
        # Remove language names that might appear without brackets
        language_names = [
            'CHURCH SLAVONIC', 'UKRAINIAN', 'RUSSIAN', 'POLISH', 'ANCIENT GREEK', 
            'ARMENIAN', 'LATIN', 'LOW GERMAN', 'GREEK', 'UNKNOWN', 'MIXED',
            'CYRILLIC', 'AVONIC'
        ]
        
        for lang in language_names:
            clean_text = re.sub(rf'\b{re.escape(lang)}\b\s*', '', clean_text, flags=re.IGNORECASE)
        
        # Remove simple language prefixes
        prefixes_to_remove = [
            'cs:', 'uk:', 'ru:', 'pl:', 'grc:', 'hy:', 'la:', 'nds:',
            'church_slavonic:', 'ukrainian:', 'russian:', 'polish:', 
            'ancient_greek:', 'armenian:', 'latin:', 'low_german:'
        ]
        
        for prefix in prefixes_to_remove:
            if clean_text.lower().startswith(prefix):
                clean_text = clean_text[len(prefix):].strip()
                break
        
        # Clean up multiple spaces and leading/trailing whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def calculate_metrics(self, predictions, ground_truths):
        """Calculate various evaluation metrics."""
        if len(predictions) != len(ground_truths):
            print(f"⚠️ Mismatch in lengths: {len(predictions)} vs {len(ground_truths)}")
            min_len = min(len(predictions), len(ground_truths))
            predictions = predictions[:min_len]
            ground_truths = ground_truths[:min_len]
        
        total_samples = len(predictions)
        exact_match_count = 0
        non_empty_predictions = 0
        total_chars = 0
        correct_chars = 0
        
        for pred, gt in zip(predictions, ground_truths):
            pred_clean = pred.strip()
            gt_clean = str(gt).strip()
            
            if pred_clean:
                non_empty_predictions += 1
            
            if pred_clean == gt_clean:
                exact_match_count += 1
            
            if gt_clean:
                total_chars += len(gt_clean)
                correct_chars += sum(1 for i, char in enumerate(gt_clean) 
                                   if i < len(pred_clean) and pred_clean[i] == char)
        
        # Word overlap score
        word_overlaps = []
        for pred, gt in zip(predictions, ground_truths):
            pred_words = set(pred.strip().lower().split())
            gt_words = set(str(gt).strip().lower().split())
            
            if gt_words:
                overlap = len(pred_words & gt_words) / len(gt_words)
                word_overlaps.append(overlap)
        
        metrics = {
            'total_samples': total_samples,
            'exact_match_count': exact_match_count,
            'exact_match_rate': exact_match_count / total_samples if total_samples > 0 else 0,
            'non_empty_predictions': non_empty_predictions,
            'non_empty_rate': non_empty_predictions / total_samples if total_samples > 0 else 0,
            'character_accuracy': correct_chars / total_chars if total_chars > 0 else 0,
            'average_word_overlap': np.mean(word_overlaps) if word_overlaps else 0
        }
        
        return metrics
    
    def evaluate_model(self, model_name, dataset_path, images_path):
        """Evaluate a single model."""
        print(f"\n{'='*60}")
        print(f"🔍 Evaluating {model_name}")
        print(f"📝 Description: {self.models[model_name]['description']}")
        print(f"{'='*60}")
        
        # Load model
        model_result = self.load_model(model_name)
        if model_result is None:
            return None
            
        model, checkpoint = model_result
        
        # Create dataset
        dataset = self.create_dataset(model_name, dataset_path, images_path)
        if dataset is None:
            return None
            
        print(f"📊 Dataset size: {len(dataset)}")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # Generate predictions
        predictions, ground_truths, metadata = self.predict_batch(model, dataloader, model_name, dataset)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, ground_truths)
        
        # Compile results
        results = {
            'model_name': model_name,
            'model_description': self.models[model_name]['description'],
            'dataset_path': dataset_path,
            'evaluation_time': datetime.now().isoformat(),
            'model_epoch': checkpoint.get('epoch', 'unknown'),
            'model_train_loss': checkpoint.get('train_loss', 'unknown'),
            'model_val_loss': checkpoint.get('val_loss', 'unknown'),
            'metrics': metrics,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'metadata': metadata
        }
        
        # Print summary
        print(f"\n📊 {model_name.upper()} RESULTS:")
        print(f"  Non-empty predictions: {metrics['non_empty_predictions']}/{metrics['total_samples']} ({metrics['non_empty_rate']*100:.1f}%)")
        print(f"  Exact matches: {metrics['exact_match_count']}/{metrics['total_samples']} ({metrics['exact_match_rate']*100:.1f}%)")
        print(f"  Character accuracy: {metrics['character_accuracy']*100:.1f}%")
        print(f"  Word overlap: {metrics['average_word_overlap']*100:.1f}%")
        
        return results
    
    def save_results_to_csv(self, results, output_dir='evaluation_results'):
        """Save evaluation results to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save detailed predictions
        predictions_data = []
        for model_name, model_results in results.items():
            if model_results is None:
                continue
                
            for i, (pred, gt, meta) in enumerate(zip(
                model_results['predictions'], 
                model_results['ground_truths'], 
                model_results['metadata']
            )):
                predictions_data.append({
                    'model_name': model_name,
                    'annotation_id': meta['annotation_id'],
                    'prediction': pred,
                    'ground_truth': gt,
                    'language': meta['language'],
                    'writing_system': meta['writing_system'],
                    'prediction_length': len(str(pred)),
                    'ground_truth_length': len(str(gt)),
                    'exact_match': pred.strip() == str(gt).strip(),
                    'is_empty_prediction': len(pred.strip()) == 0
                })
        
        predictions_file = f"{output_dir}/detailed_predictions_{timestamp}.csv"
        pd.DataFrame(predictions_data).to_csv(predictions_file, index=False)
        print(f"💾 Detailed predictions saved to: {predictions_file}")
        
        # 2. Save model comparison
        comparison_data = []
        for model_name, model_results in results.items():
            if model_results is None:
                continue
            metrics = model_results['metrics']
            comparison_data.append({
                'model_name': model_name,
                'description': model_results['model_description'],
                'total_samples': metrics['total_samples'],
                'exact_match_rate': metrics['exact_match_rate'],
                'non_empty_rate': metrics['non_empty_rate'],
                'character_accuracy': metrics['character_accuracy'],
                'word_overlap': metrics['average_word_overlap'],
                'evaluation_time': model_results['evaluation_time']
            })
        
        comparison_file = f"{output_dir}/model_comparison_{timestamp}.csv"
        pd.DataFrame(comparison_data).to_csv(comparison_file, index=False)
        print(f"💾 Model comparison saved to: {comparison_file}")
        
        # 3. Save language-specific performance
        language_data = []
        for model_name, model_results in results.items():
            if model_results is None:
                continue
            
            # Group by language
            df = pd.DataFrame({
                'prediction': model_results['predictions'],
                'ground_truth': model_results['ground_truths'],
                'language': [meta['language'] for meta in model_results['metadata']]
            })
            
            for language in df['language'].unique():
                lang_df = df[df['language'] == language]
                lang_metrics = self.calculate_metrics(
                    lang_df['prediction'].tolist(),
                    lang_df['ground_truth'].tolist()
                )
                language_data.append({
                    'model_name': model_name,
                    'language': language,
                    'samples': len(lang_df),
                    'exact_match_rate': lang_metrics['exact_match_rate'],
                    'non_empty_rate': lang_metrics['non_empty_rate'],
                    'character_accuracy': lang_metrics['character_accuracy']
                })
        
        language_file = f"{output_dir}/language_performance_{timestamp}.csv"
        pd.DataFrame(language_data).to_csv(language_file, index=False)
        print(f"💾 Language performance saved to: {language_file}")
        
        return {
            'predictions': predictions_file,
            'comparison': comparison_file,
            'language': language_file
        }
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation on all models."""
        print("🚀 Starting Comprehensive Saint Sophia Model Evaluation")
        print("="*70)
        
        all_results = {}
        
        for dataset_info in self.datasets.values():
            print(f"\n🔍 Evaluating on {dataset_info['name']} dataset")
            
            dataset_results = {}
            for model_name in self.models.keys():
                try:
                    results = self.evaluate_model(
                        model_name,
                        dataset_info['csv_path'],
                        dataset_info['images_path']
                    )
                    dataset_results[model_name] = results
                except Exception as e:
                    print(f"❌ Failed to evaluate {model_name}: {e}")
                    dataset_results[model_name] = None
            
            # Save results for this dataset
            print(f"\n💾 Saving {dataset_info['name']} results...")
            files = self.save_results_to_csv(dataset_results, f"evaluation_results/{dataset_info['name']}")
            all_results[dataset_info['name']] = {
                'results': dataset_results,
                'files': files
            }
        
        return all_results

def main():
    """Main evaluation function."""
    print("🏛️ Saint Sophia Graffiti Recognition - Comprehensive Evaluation")
    print("="*70)
    
    # Initialize evaluator
    evaluator = SophiaEvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    print("\n🎉 Evaluation completed!")
    print("📁 Results saved in evaluation_results/ directory")
    
    # Print summary
    print("\n📊 EVALUATION SUMMARY:")
    print("="*50)
    
    for dataset_name, dataset_data in results.items():
        print(f"\n📋 {dataset_name.upper()} DATASET:")
        for model_name, model_results in dataset_data['results'].items():
            if model_results and model_results['metrics']:
                metrics = model_results['metrics']
                print(f"  {model_name:20s}: {metrics['exact_match_rate']*100:5.1f}% exact, "
                      f"{metrics['non_empty_rate']*100:5.1f}% non-empty, "
                      f"{metrics['character_accuracy']*100:5.1f}% char accuracy")
            else:
                print(f"  {model_name:20s}: FAILED")

if __name__ == '__main__':
    main()

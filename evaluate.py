#!/usr/bin/env python3
"""
Evaluation Script for Saint Sophia Graffiti Recognition
=======================================================

Comprehensive evaluation supporting all three model architectures:
- Multi-Channel CNN (70M params)
- Enhanced CNN (58M params)
- Transformer (51M params)

Features:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Sequence accuracy
- Per-language performance
- Per-writing-system performance
- Confusion matrices
- Detailed error analysis
- Visual comparison (predicted vs ground truth)

Usage:
    # Evaluate Enhanced model
    python evaluate.py --model enhanced \
        --checkpoint checkpoints/enhanced/20251015_143000/best_model.pt \
        --test_csv data/test_comprehensive.csv

    # Evaluate with specific modalities
    python evaluate.py --model transformer \
        --checkpoint checkpoints/transformer/best_model.pt \
        --use_korniienko --no_rti

    # Save detailed results
    python evaluate.py --model enhanced \
        --checkpoint checkpoints/enhanced/best_model.pt \
        --output_dir evaluation_results/enhanced_phase3
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from training script
from train import (
    SophiaMultiModalDataset,
    CharacterTokenizer,
    collate_fn,
    create_model
)

try:
    from transformers import XLMRobertaTokenizer
except ImportError:
    XLMRobertaTokenizer = None

# For CER/WER calculation
try:
    from jiwer import cer, wer
    JIWER_AVAILABLE = True
except ImportError:
    print("  jiwer not available. Installing basic CER/WER calculation...")
    JIWER_AVAILABLE = False


def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate."""
    if JIWER_AVAILABLE:
        try:
            return cer(reference, hypothesis)
        except:
            pass
    
    # Fallback: Levenshtein distance
    if len(reference) == 0:
        return len(hypothesis)
    if len(hypothesis) == 0:
        return len(reference)
    
    # Create distance matrix
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1))
    
    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(hypothesis) + 1):
        d[0][j] = j
    
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i-1] == hypothesis[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion
                    d[i-1][j-1] + 1     # substitution
                )
    
    return d[len(reference)][len(hypothesis)] / len(reference)


def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate."""
    if JIWER_AVAILABLE:
        try:
            return wer(reference, hypothesis)
        except:
            pass
    
    # Fallback: word-level Levenshtein
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return len(hyp_words)
    if len(hyp_words) == 0:
        return len(ref_words)
    
    # Create distance matrix
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion
                    d[i-1][j-1] + 1     # substitution
                )
    
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def decode_prediction(logits, tokenizer):
    """Decode model output to text with proper EOS handling and cleaning."""
    # Get predicted tokens (greedy decoding)
    predicted_ids = torch.argmax(logits, dim=-1)  # [batch, seq_len]
    
    predictions = []
    for ids in predicted_ids:
        # Remove padding and special tokens
        ids = ids.cpu().numpy()
        
        # Check if it's XLM-RoBERTa tokenizer (has vocab_file attribute) or Character tokenizer
        if hasattr(tokenizer, 'vocab_file') or tokenizer.__class__.__name__ == 'XLMRobertaTokenizer':
            # XLM-RoBERTa tokenizer
            text = tokenizer.decode(ids, skip_special_tokens=True)
        else:
            # Character tokenizer - improved decoding
            chars = []
            for idx in ids:
                # Skip special tokens
                if idx in [0, 1]:  # PAD, SOS
                    continue
                if idx == 2:  # EOS - stop here!
                    break
                    
                # Get character (with fallback for unknown indices)
                if idx in tokenizer.idx_to_char:
                    char = tokenizer.idx_to_char[idx]
                    # Skip <UNK> token (index 3)
                    if idx == 3:
                        continue
                    chars.append(char)
                # If index out of vocab range, skip it
                
            text = ''.join(chars)
        
        # Clean up the prediction
        text = text.strip()
        
        # Remove excessive repetition at the end (common issue)
        # If last 10+ characters are all the same, trim them
        if len(text) > 10:
            last_char = text[-1]
            # Count how many times it repeats at the end
            repeat_count = 0
            for i in range(len(text) - 1, -1, -1):
                if text[i] == last_char:
                    repeat_count += 1
                else:
                    break
            
            # If more than 10 repetitions, it's likely garbage - remove them
            if repeat_count > 10:
                text = text[:-repeat_count].rstrip()
        
        predictions.append(text)
    
    return predictions


def generate_text_transformer(model, tokenizer, max_length=512, **kwargs):
    """
    Autoregressive text generation for transformer model.
    
    Args:
        model: The transformer model
        tokenizer: XLM-RoBERTa tokenizer
        max_length: Maximum sequence length to generate
        **kwargs: Model inputs (rti_images, korniienko_photo, etc.)
    
    Returns:
        texts: List of generated text strings
    """
    batch_size = kwargs.get('rti_images', kwargs.get('korniienko_photo')).size(0)
    device = kwargs.get('rti_images', kwargs.get('korniienko_photo')).device
    
    # Start with BOS token
    generated_ids = torch.full((batch_size, 1), tokenizer.bos_token_id, 
                              dtype=torch.long, device=device)
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Generate tokens one at a time
    for step in range(max_length - 1):
        # Pass generated tokens so far
        outputs = model(
            **kwargs,
            text_indices=generated_ids,
            text_mask=None
        )
        
        logits = outputs['transcription_logits']  # [batch, seq_len, vocab]
        
        # Get next token (greedy)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [batch, 1]
        
        # Check for EOS or repetition
        is_eos = (next_token.squeeze(-1) == tokenizer.eos_token_id)
        finished = finished | is_eos
        
        # Stop if all sequences finished
        if finished.all():
            break
        
        # Check for repetition (if last 5 tokens are the same, stop)
        if step >= 5 and generated_ids.size(1) >= 6:
            last_5 = generated_ids[:, -5:]
            is_repeating = (last_5 == last_5[:, :1]).all(dim=1)
            finished = finished | is_repeating
            
            if finished.all():
                break
        
        # Append to sequence (don't append if finished)
        next_token = torch.where(finished.unsqueeze(-1), 
                                torch.full_like(next_token, tokenizer.pad_token_id),
                                next_token)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
    
    # Decode to text
    texts = []
    for ids in generated_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        texts.append(text.strip())
    
    return texts


def evaluate_model(model, dataloader, tokenizer, device, model_type,
                   use_rti=True, use_korniienko=True):
    """
    Evaluate model on test set.
    
    Returns:
        results: dict with predictions, ground truth, and metadata
    """
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    all_isialy_ids = []
    all_languages = []
    all_writing_systems = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            languages = batch['language'].to(device)
            writing_systems = batch['writing_system'].to(device)
            
            # Prepare images
            kwargs = {}
            if use_rti and 'rti_images' in batch:
                kwargs['images'] = batch['rti_images'].to(device)
            
            if use_korniienko:
                if batch.get('korniienko_photo') is not None:
                    kwargs['korniienko_photo'] = batch['korniienko_photo'].to(device)
                
                if batch.get('korniienko_drawing') is not None:
                    kwargs['korniienko_drawing'] = batch['korniienko_drawing'].to(device)
            
            # Forward pass
            if model_type == 'transformer':
                # Use teacher forcing for transformer (since autoregressive fails)
                kwargs['text_indices'] = input_ids
                kwargs['text_mask'] = attention_mask
                if 'images' in kwargs:
                    kwargs['rti_images'] = kwargs.pop('images')
                outputs = model(**kwargs)
                logits = outputs['transcription_logits']
                # Decode predictions
                predictions = decode_prediction(logits, tokenizer)
            else:
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    languages=languages,
                    writing_systems=writing_systems,
                    **kwargs
                )
                # Decode predictions
                predictions = decode_prediction(logits, tokenizer)
            
            # Store results
            all_predictions.extend(predictions)
            all_ground_truth.extend(batch['transcription'])
            all_isialy_ids.extend(batch['isialy_id'])
            all_languages.extend(languages.cpu().numpy())
            all_writing_systems.extend(writing_systems.cpu().numpy())
    
    return {
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
        'isialy_ids': all_isialy_ids,
        'languages': all_languages,
        'writing_systems': all_writing_systems
    }


def calculate_metrics(results, dataset):
    """Calculate comprehensive evaluation metrics."""
    predictions = results['predictions']
    ground_truth = results['ground_truth']
    languages = results['languages']
    writing_systems = results['writing_systems']
    
    # Overall metrics
    cer_scores = [calculate_cer(gt, pred) for gt, pred in zip(ground_truth, predictions)]
    wer_scores = [calculate_wer(gt, pred) for gt, pred in zip(ground_truth, predictions)]
    
    # Sequence accuracy (exact match)
    exact_matches = [1 if gt == pred else 0 for gt, pred in zip(ground_truth, predictions)]
    
    metrics = {
        'overall': {
            'cer_mean': np.mean(cer_scores),
            'cer_std': np.std(cer_scores),
            'cer_median': np.median(cer_scores),
            'wer_mean': np.mean(wer_scores),
            'wer_std': np.std(wer_scores),
            'wer_median': np.median(wer_scores),
            'sequence_accuracy': np.mean(exact_matches),
            'num_samples': len(predictions)
        }
    }
    
    # Per-language metrics
    language_metrics = {}
    for lang_idx in set(languages):
        lang_name = dataset.idx_to_language[lang_idx]
        lang_mask = [l == lang_idx for l in languages]
        
        lang_cer = [cer_scores[i] for i, m in enumerate(lang_mask) if m]
        lang_wer = [wer_scores[i] for i, m in enumerate(lang_mask) if m]
        lang_acc = [exact_matches[i] for i, m in enumerate(lang_mask) if m]
        
        if lang_cer:
            language_metrics[lang_name] = {
                'cer_mean': np.mean(lang_cer),
                'wer_mean': np.mean(lang_wer),
                'sequence_accuracy': np.mean(lang_acc),
                'num_samples': len(lang_cer)
            }
    
    metrics['per_language'] = language_metrics
    
    # Per-writing-system metrics
    ws_metrics = {}
    for ws_idx in set(writing_systems):
        ws_name = dataset.idx_to_ws[ws_idx]
        ws_mask = [w == ws_idx for w in writing_systems]
        
        ws_cer = [cer_scores[i] for i, m in enumerate(ws_mask) if m]
        ws_wer = [wer_scores[i] for i, m in enumerate(ws_mask) if m]
        ws_acc = [exact_matches[i] for i, m in enumerate(ws_mask) if m]
        
        if ws_cer:
            ws_metrics[ws_name] = {
                'cer_mean': np.mean(ws_cer),
                'wer_mean': np.mean(ws_wer),
                'sequence_accuracy': np.mean(ws_acc),
                'num_samples': len(ws_cer)
            }
    
    metrics['per_writing_system'] = ws_metrics
    
    return metrics


def save_detailed_results(results, metrics, output_dir):
    """Save detailed evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ Metrics saved: {metrics_file}")
    
    # Save predictions CSV
    predictions_df = pd.DataFrame({
        'isialy_id': results['isialy_ids'],
        'ground_truth': results['ground_truth'],
        'prediction': results['predictions'],
        'language': results['languages'],
        'writing_system': results['writing_systems']
    })
    predictions_file = output_dir / 'predictions.csv'
    predictions_df.to_csv(predictions_file, index=False, encoding='utf-8')
    print(f"✓ Predictions saved: {predictions_file}")
    
    # Save error analysis
    predictions_df['cer'] = [
        calculate_cer(gt, pred) 
        for gt, pred in zip(results['ground_truth'], results['predictions'])
    ]
    predictions_df['wer'] = [
        calculate_wer(gt, pred)
        for gt, pred in zip(results['ground_truth'], results['predictions'])
    ]
    predictions_df['exact_match'] = [
        gt == pred
        for gt, pred in zip(results['ground_truth'], results['predictions'])
    ]
    
    # Sort by error rate (worst first)
    error_analysis = predictions_df.sort_values('cer', ascending=False)
    error_file = output_dir / 'error_analysis.csv'
    error_analysis.to_csv(error_file, index=False, encoding='utf-8')
    print(f"✓ Error analysis saved: {error_file}")
    
    # Save best predictions (for qualitative analysis)
    best_predictions = predictions_df[predictions_df['exact_match'] == True].head(50)
    if len(best_predictions) > 0:
        best_file = output_dir / 'best_predictions.csv'
        best_predictions.to_csv(best_file, index=False, encoding='utf-8')
        print(f"✓ Best predictions saved: {best_file}")
    
    # Save worst predictions (for error analysis)
    worst_predictions = error_analysis.head(50)
    worst_file = output_dir / 'worst_predictions.csv'
    worst_predictions.to_csv(worst_file, index=False, encoding='utf-8')
    print(f"✓ Worst predictions saved: {worst_file}")
    
    # Generate summary report
    generate_summary_report(metrics, output_dir)


def generate_summary_report(metrics, output_dir):
    """Generate human-readable summary report."""
    report_file = output_dir / 'EVALUATION_REPORT.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Evaluation Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall metrics
        f.write("## Overall Performance\n\n")
        overall = metrics['overall']
        f.write(f"- **Samples**: {overall['num_samples']}\n")
        f.write(f"- **Character Error Rate (CER)**:\n")
        f.write(f"  - Mean: {overall['cer_mean']:.2%}\n")
        f.write(f"  - Median: {overall['cer_median']:.2%}\n")
        f.write(f"  - Std: {overall['cer_std']:.2%}\n")
        f.write(f"- **Word Error Rate (WER)**:\n")
        f.write(f"  - Mean: {overall['wer_mean']:.2%}\n")
        f.write(f"  - Median: {overall['wer_median']:.2%}\n")
        f.write(f"  - Std: {overall['wer_std']:.2%}\n")
        f.write(f"- **Sequence Accuracy**: {overall['sequence_accuracy']:.2%}\n\n")
        
        # Character accuracy (inverse of CER)
        char_acc = 1.0 - overall['cer_mean']
        word_acc = 1.0 - overall['wer_mean']
        f.write(f"### Derived Metrics\n\n")
        f.write(f"- **Character Accuracy**: {char_acc:.2%}\n")
        f.write(f"- **Word Accuracy**: {word_acc:.2%}\n")
        
        # Target achievement
        f.write(f"\n### Target Achievement\n\n")
        if word_acc >= 0.5:
            f.write(f" **TARGET MET**: Word accuracy {word_acc:.2%} >= 50%\n\n")
        else:
            gap = 0.5 - word_acc
            f.write(f" **TARGET NOT MET**: Word accuracy {word_acc:.2%} < 50% (gap: {gap:.2%})\n\n")
        
        # Per-language metrics
        if 'per_language' in metrics and metrics['per_language']:
            f.write("## Per-Language Performance\n\n")
            f.write("| Language | CER | WER | Accuracy | Samples |\n")
            f.write("|----------|-----|-----|----------|----------|\n")
            
            for lang, lang_metrics in sorted(metrics['per_language'].items()):
                f.write(f"| {lang} | {lang_metrics['cer_mean']:.2%} | ")
                f.write(f"{lang_metrics['wer_mean']:.2%} | ")
                f.write(f"{lang_metrics['sequence_accuracy']:.2%} | ")
                f.write(f"{lang_metrics['num_samples']} |\n")
            f.write("\n")
        
        # Per-writing-system metrics
        if 'per_writing_system' in metrics and metrics['per_writing_system']:
            f.write("## Per-Writing-System Performance\n\n")
            f.write("| Writing System | CER | WER | Accuracy | Samples |\n")
            f.write("|----------------|-----|-----|----------|----------|\n")
            
            for ws, ws_metrics in sorted(metrics['per_writing_system'].items()):
                f.write(f"| {ws} | {ws_metrics['cer_mean']:.2%} | ")
                f.write(f"{ws_metrics['wer_mean']:.2%} | ")
                f.write(f"{ws_metrics['sequence_accuracy']:.2%} | ")
                f.write(f"{ws_metrics['num_samples']} |\n")
            f.write("\n")
        
        # Interpretation
        f.write("## Interpretation\n\n")
        if overall['cer_mean'] < 0.15:
            f.write(" **Excellent**: CER < 15% indicates high-quality recognition.\n\n")
        elif overall['cer_mean'] < 0.25:
            f.write(" **Good**: CER < 25% indicates acceptable recognition quality.\n\n")
        elif overall['cer_mean'] < 0.35:
            f.write(" **Fair**: CER < 35% indicates room for improvement.\n\n")
        else:
            f.write(" **Poor**: CER > 35% indicates significant recognition errors.\n\n")
        
        f.write("### Next Steps\n\n")
        if word_acc < 0.5:
            f.write("- Increase training epochs\n")
            f.write("- Try different model architectures\n")
            f.write("- Add more training data\n")
            f.write("- Use full multi-modal training (RTI + Korniienko)\n")
            f.write("- Tune hyperparameters (learning rate, batch size)\n")
        else:
            f.write("- Fine-tune on specific languages/writing systems\n")
            f.write("- Analyze error patterns for targeted improvements\n")
            f.write("- Consider ensemble methods\n")
    
    print(f"✓ Summary report saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Saint Sophia Graffiti Recognition Models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model and checkpoint
    parser.add_argument('--model', type=str, required=True,
                        choices=['multichannel', 'enhanced', 'transformer'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str,
                        default='/home/aram/GRIDH/Saint_Sophia/sophia-epigraphic-ai',
                        help='Base data directory')
    parser.add_argument('--test_csv', type=str, default='data/test_comprehensive.csv',
                        help='Test CSV file (relative to data_dir)')
    
    # Modality selection
    parser.add_argument('--use_rti', action='store_true', default=False,
                        help='Use RTI images')
    parser.add_argument('--no_rti', action='store_true', default=False,
                        help='Disable RTI images')
    parser.add_argument('--use_korniienko', action='store_true', default=False,
                        help='Use Korniienko images')
    parser.add_argument('--no_korniienko', action='store_true', default=False,
                        help='Disable Korniienko images')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum text sequence length')
    
    # Tokenizer selection
    parser.add_argument('--tokenizer', type=str, default='xlm',
                        choices=['xlm', 'character'],
                        help='Tokenizer type: xlm (XLM-RoBERTa) or character (character-level)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-generated if not provided)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Device
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Handle modality flags
    if not args.use_rti and not args.no_rti and not args.use_korniienko and not args.no_korniienko:
        args.use_rti = True
        args.use_korniienko = True
    elif args.use_rti:
        args.use_rti = True
    elif args.no_rti:
        args.use_rti = False
    else:
        args.use_rti = True
    
    if args.use_korniienko:
        args.use_korniienko = True
    elif args.no_korniienko:
        args.use_korniienko = False
    else:
        args.use_korniienko = True
    
    # Auto-generate output directory if not provided
    if args.output_dir is None:
        checkpoint_name = Path(args.checkpoint).stem
        modalities = []
        if args.use_rti:
            modalities.append('rti')
        if args.use_korniienko:
            modalities.append('korniienko')
        modality_str = '+'.join(modalities) if modalities else 'none'
        
        args.output_dir = f"evaluation_results/{args.model}_{checkpoint_name}_{modality_str}"
    
    # Print configuration
    print("=" * 70)
    print("SAINT SOPHIA GRAFFITI RECOGNITION - EVALUATION")
    print("=" * 70)
    print(f"\n Model: {args.model.upper()}")
    print(f" Checkpoint: {args.checkpoint}")
    print(f"  Modalities:")
    print(f"   RTI Images: {'✓' if args.use_rti else '✗'}")
    print(f"   Korniienko: {'✓' if args.use_korniienko else '✗'}")
    print(f"\n Data:")
    print(f"   Base Dir: {args.data_dir}")
    print(f"   Test CSV: {args.test_csv}")
    print(f"\n Output: {args.output_dir}")
    print(f" Device: {args.device}")
    print("=" * 70)
    
    # Load checkpoint
    print("\n Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    print(f" Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Initialize tokenizer
    print("\n Initializing tokenizer...")
    if args.tokenizer == 'character':
        tokenizer = CharacterTokenizer()
        vocab_size = tokenizer.vocab_size
        print(f"✓ Character-level tokenizer loaded (vocab size: {vocab_size})")
        print(f"  ✓ Preserves all ancient Cyrillic characters (ꙗ, ꙅ, ѧ, etc.)")
    else:  # xlm
        if XLMRobertaTokenizer is not None:
            try:
                tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
                vocab_size = tokenizer.vocab_size
                print(f" XLM-RoBERTa tokenizer loaded (vocab size: {vocab_size})")
                print(f"   Warning: May corrupt ancient Cyrillic characters")
            except:
                print("  XLM-RoBERTa failed, falling back to character tokenizer")
                tokenizer = CharacterTokenizer()
                vocab_size = tokenizer.vocab_size
        else:
            print("  XLM-RoBERTa not available, using character tokenizer")
            tokenizer = CharacterTokenizer()
            vocab_size = tokenizer.vocab_size
    
    # Create dataset
    print("\n Loading test dataset...")
    # Determine split from CSV filename
    csv_filename = Path(args.test_csv).stem  # 'test_comprehensive', 'val_comprehensive', etc.
    if 'test' in csv_filename:
        split = 'test'
    elif 'val' in csv_filename:
        split = 'val'
    else:
        split = 'test'  # default
    
    # Handle CSV paths - if they contain 'data/', remove it since data_dir already points to data/
    test_csv_path = args.test_csv.replace('data/', '') if 'data/' in args.test_csv else args.test_csv
    
    test_dataset = SophiaMultiModalDataset(
        csv_file=Path(args.data_dir) / test_csv_path,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_rti=args.use_rti,
        use_korniienko=args.use_korniienko,
        model_type=args.model,
        image_size=args.image_size,
        augment=False,
        split=split
    )
    
    print(f"✓ Test samples: {len(test_dataset)}")
    
    if len(test_dataset) == 0:
        print("\n No test samples found! Check data availability.")
        return 1
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create model
    print("\n  Creating model...")
    
    # Check if checkpoint has language/writing system info (new format)
    # If not, we need to use training dataset counts to match the trained model
    if 'num_languages' in checkpoint and 'num_writing_systems' in checkpoint:
        num_languages = checkpoint['num_languages']
        num_writing_systems = checkpoint['num_writing_systems']
        print(f"  Using checkpoint metadata: {num_languages} languages, {num_writing_systems} writing systems")
    else:
        # Old checkpoint format - load training dataset to get correct counts
        print("  Old checkpoint format detected. Loading training dataset to determine model dimensions...")
        import pandas as pd
        train_df = pd.read_csv('data/train_comprehensive.csv')
        # Count unique languages and writing systems from training data
        num_languages = train_df['language_name'].nunique() + 1  # +1 for unknown
        num_writing_systems = train_df['writing_system_name'].nunique() + 1
        print(f"  Using training dataset counts: {num_languages} languages, {num_writing_systems} writing systems")
    
    model = create_model(
        model_type=args.model,
        vocab_size=vocab_size,
        num_languages=num_languages,
        num_writing_systems=num_writing_systems,
        use_korniienko=args.use_korniienko
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {args.model}")
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION START")
    print("=" * 70)
    
    results = evaluate_model(
        model, test_loader, tokenizer, args.device, args.model,
        args.use_rti, args.use_korniienko
    )
    
    print("\n Calculating metrics...")
    metrics = calculate_metrics(results, test_dataset)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    overall = metrics['overall']
    print(f"\n Overall Performance ({overall['num_samples']} samples):")
    print(f"   Character Error Rate (CER): {overall['cer_mean']:.2%} ± {overall['cer_std']:.2%}")
    print(f"   Word Error Rate (WER):      {overall['wer_mean']:.2%} ± {overall['wer_std']:.2%}")
    print(f"   Sequence Accuracy:          {overall['sequence_accuracy']:.2%}")
    print(f"\n   Character Accuracy: {(1 - overall['cer_mean']):.2%}")
    print(f"   Word Accuracy:      {(1 - overall['wer_mean']):.2%}")
    
    # Target achievement
    word_acc = 1.0 - overall['wer_mean']
    if word_acc >= 0.5:
        print(f"\n    TARGET MET: Word accuracy {word_acc:.2%} >= 50%")
    else:
        gap = 0.5 - word_acc
        print(f"\n    TARGET NOT MET: Word accuracy {word_acc:.2%} < 50% (gap: {gap:.2%})")
    
    # Save results
    print("\n Saving results...")
    save_detailed_results(results, metrics, args.output_dir)
    
    print("\n" + "=" * 70)
    print(" EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n Results saved in: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  • metrics.json           - Complete metrics")
    print(f"  • predictions.csv        - All predictions")
    print(f"  • error_analysis.csv     - Sorted by error rate")
    print(f"  • worst_predictions.csv  - Top 50 worst cases")
    print(f"  • best_predictions.csv   - Top 50 best cases")
    print(f"  • EVALUATION_REPORT.md   - Human-readable summary")
    
    print("\n Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

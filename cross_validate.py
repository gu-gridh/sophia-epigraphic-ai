#!/usr/bin/env python3
"""
K-Fold Cross-Validation for Saint Sophia Graffiti Recognition
==============================================================

Implements stratified k-fold cross-validation to provide:
- More robust performance estimates
- Confidence intervals for metrics
- Per-fold analysis
- Language-stratified splits

This addresses reviewer concerns about evaluation bias on small datasets.

Usage:
    # 5-fold cross-validation with Enhanced model
    python cross_validate.py --model enhanced --folds 5 --epochs 10
    
    # Quick validation (3-fold, fewer epochs)
    python cross_validate.py --model enhanced --folds 3 --epochs 5 --quick
    
    # Full training with all modalities
    python cross_validate.py --model transformer --folds 5 --use_korniienko --use_rti
"""

import os
import sys
import argparse
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import from training script
from train import (
    SophiaMultiModalDataset,
    CharacterTokenizer,
    collate_fn,
    create_model,
    build_tokenizer,
    default_device,
    load_initial_weights,
    train_epoch,
)
from decoding import get_special_ids
from models.models_trocr import DEFAULT_TROCR, DEFAULT_VIEWS
from decoding import shift_tokens, build_image_inputs, language_inputs, greedy_generate, decode_ids


def get_language_text(lang_str):
    """Extract language text from JSON or string."""
    try:
        if pd.isna(lang_str) or lang_str == '':
            return 'Unknown'
        # Check if it's already a simple string
        if isinstance(lang_str, str) and not lang_str.startswith('{'):
            return lang_str
        lang_dict = eval(lang_str) if isinstance(lang_str, str) else lang_str
        return lang_dict.get('text', 'Unknown') if isinstance(lang_dict, dict) else str(lang_str)
    except:
        return 'Unknown'


def create_stratified_folds(df, n_folds=5, random_seed=42, language_col='language_name'):
    """
    Create stratified k-fold splits based on language distribution.
    
    Args:
        df: DataFrame with data
        n_folds: Number of folds
        random_seed: Random seed for reproducibility
        language_col: Column name for language (for stratification)
        
    Returns:
        List of (train_df, val_df) tuples
    """
    np.random.seed(random_seed)
    
    # Extract language for stratification
    df = df.copy().reset_index(drop=True)
    
    # Get language column
    if language_col in df.columns:
        df['_language'] = df[language_col].fillna('Unknown')
    elif 'language' in df.columns:
        df['_language'] = df['language'].apply(get_language_text)
    else:
        df['_language'] = 'Unknown'
    
    df['_fold'] = -1
    
    # Assign folds per language group
    for lang in df['_language'].unique():
        lang_mask = df['_language'] == lang
        lang_indices = df[lang_mask].index.tolist()
        np.random.shuffle(lang_indices)
        
        # Assign to folds
        for i, idx in enumerate(lang_indices):
            df.loc[idx, '_fold'] = i % n_folds
    
    # Create fold splits (return DataFrames, not indices)
    folds = []
    for fold_idx in range(n_folds):
        val_df = df[df['_fold'] == fold_idx].drop(columns=['_language', '_fold']).reset_index(drop=True)
        train_df = df[df['_fold'] != fold_idx].drop(columns=['_language', '_fold']).reset_index(drop=True)
        folds.append((train_df, val_df))
    
    return folds


def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate using Levenshtein distance."""
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0
    if len(hypothesis) == 0:
        return 1.0
    
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
                d[i][j] = min(d[i-1][j] + 1,      # deletion
                             d[i][j-1] + 1,      # insertion
                             d[i-1][j-1] + 1)    # substitution
    
    cer = d[len(reference)][len(hypothesis)] / len(reference)
    return min(cer, 1.0)  # Cap CER at 100%


def decode_prediction(logits, tokenizer):
    """Decode model output to text with proper EOS handling (from evaluate.py)."""
    # Get predicted tokens (greedy decoding)
    predicted_ids = torch.argmax(logits, dim=-1)  # [batch, seq_len]
    
    predictions = []
    for ids in predicted_ids:
        ids = ids.cpu().numpy()
        
        # Character tokenizer decoding
        chars = []
        for idx in ids:
            # Skip special tokens
            if idx in [0, 1]:  # PAD, SOS
                continue
            if idx == 2:  # EOS - stop here!
                break
            if idx == 3:  # UNK - skip
                continue
                
            # Get character
            if idx in tokenizer.idx_to_char:
                chars.append(tokenizer.idx_to_char[idx])
                
        text = ''.join(chars).strip()
        predictions.append(text)
    
    return predictions


def evaluate_model(model, dataloader, tokenizer, device, use_rti=True, use_korniienko=True,
                   decoding='autoregressive', image_ablation='none', max_length=256,
                   use_language=False):
    """
    Evaluate a model.

    decoding='autoregressive' generates from the images only (the real task).
    decoding='teacher_forced_legacy' feeds the ground truth to the decoder and scores the
    unshifted argmax (the protocol behind the published numbers; it does not measure recognition).
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_cer = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lang_inputs = language_inputs(batch, device, use_language)
            image_inputs = build_image_inputs(batch, device, use_rti, use_korniienko,
                                              ablation=image_ablation)
            gt_transcriptions = batch.get('transcription', [])
            
            if decoding == 'autoregressive':
                generated = greedy_generate(
                    model, image_inputs, tokenizer, max_length=max_length,
                    **lang_inputs,
                    batch_size=input_ids.size(0), device=device
                )
                predictions = decode_ids(generated.cpu().tolist(), tokenizer)
            elif decoding == 'teacher_forced_legacy':
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **lang_inputs,
                    **image_inputs
                )
                predictions = decode_prediction(logits, tokenizer)
            else:
                raise ValueError(f"Unknown decoding mode: {decoding}")
            
            # Calculate CER for each sample
            for pred, gt in zip(predictions, gt_transcriptions):
                all_predictions.append(pred)
                all_targets.append(gt)
                all_cer.append(calculate_cer(gt, pred))
    
    # Calculate metrics
    avg_cer = np.mean(all_cer) if all_cer else 1.0
    avg_cer = min(avg_cer, 1.0)  # Cap CER at 100%
    sequence_accuracy = np.mean([p == t for p, t in zip(all_predictions, all_targets)]) if all_predictions else 0.0
    char_accuracy = max(0.0, 1 - avg_cer)  # Cap at 0 minimum
    
    return {
        'cer': avg_cer,
        'char_accuracy': char_accuracy,
        'sequence_accuracy': sequence_accuracy,
        'predictions': all_predictions,
        'targets': all_targets,
        'per_sample_cer': all_cer
    }

def train_one_epoch(model, dataloader, optimizer, criterion, tokenizer, device,
                    use_rti=True, use_korniienko=True, use_language=False,
                    modality_dropout=0.0, amp=False, grad_accum=1):
    """One epoch of next-token training (shared implementation in train.py)."""
    return train_epoch(model, dataloader, optimizer, criterion, device, None,
                       use_rti, use_korniienko, scheduler=None, use_language=use_language,
                       modality_dropout=modality_dropout, amp=amp, grad_accum=grad_accum)


def run_cross_validation(args):
    """Run k-fold cross-validation."""
    print("="*70)
    print("SAINT SOPHIA GRAFFITI - K-FOLD CROSS-VALIDATION")
    print("="*70)
    
    # Setup device
    device = torch.device(args.device)
    print(f"\nDevice: {device}")
    
    # Load data
    print(f"\nLoading data from: {args.data_csv}")
    df = pd.read_csv(args.data_csv)
    print(f"Total samples loaded: {len(df)}")
    
    # Filter to rows with transcription
    text_col = 'transcription_clean'
    if text_col not in df.columns:
        print(f"Warning: '{text_col}' not found, looking for alternatives...")
        for col in ['transcription', 'text']:
            if col in df.columns:
                text_col = col
                break
    
    has_text = df[text_col].notna() & (df[text_col].astype(str).str.strip() != '')
    df = df[has_text].reset_index(drop=True)
    print(f"Samples with transcription: {len(df)}")
    
    # Create tokenizer
    tokenizer = build_tokenizer(args.model, 'character', args.pretrained)
    print(f"Tokenizer: {type(tokenizer).__name__}, vocabulary size: {tokenizer.vocab_size}")
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"cv_{args.folds}fold_{args.model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp_splits"
    temp_dir.mkdir(exist_ok=True)
    
    # Language / writing-system indices from the full corpus (same for every fold)
    languages = sorted(df['language_name'].fillna('unknown').unique()) if 'language_name' in df.columns else ['unknown']
    writing_systems = sorted(df['writing_system_name'].fillna('unknown').unique()) if 'writing_system_name' in df.columns else ['unknown']
    language_to_idx = {lang: i for i, lang in enumerate(languages)}
    ws_to_idx = {ws: i for i, ws in enumerate(writing_systems)}
    
    # Create stratified folds
    print(f"\nCreating {args.folds}-fold stratified splits...")
    folds = create_stratified_folds(df, n_folds=args.folds, random_seed=args.seed)
    
    # Print fold distribution
    for fold_idx, (train_df, val_df) in enumerate(folds):
        print(f"  Fold {fold_idx+1}: train={len(train_df)}, val={len(val_df)}")
    
    # Results storage
    fold_results = []
    all_predictions = []
    
    # Run each fold
    for fold_idx, (train_df, test_df) in enumerate(folds):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{args.folds}")
        print(f"{'='*70}")
        
        # Hold out part of the training portion (language-stratified) for choosing the best
        # epoch, so the reported test fold is never used for model selection.
        n_inner = max(2, int(round(1.0 / args.inner_val_frac)))
        fit_df, select_df = create_stratified_folds(train_df, n_folds=n_inner,
                                                    random_seed=args.seed + fold_idx)[0]
        print(f"Train: {len(fit_df)}, model-selection: {len(select_df)}, test: {len(test_df)} samples")
        
        def make_dataset(frame, name, augment, split):
            csv_path = temp_dir / f"fold{fold_idx+1}_{name}.csv"
            frame.to_csv(csv_path, index=False)
            return SophiaMultiModalDataset(
                csv_file=str(csv_path),
                data_dir=args.data_dir,
                tokenizer=tokenizer,
                max_length=args.max_length,
                use_rti=args.use_rti,
                use_korniienko=args.use_korniienko,
                model_type=args.model,
                image_size=args.image_size,
                augment=augment,
                split=split,
                keep_aspect=args.keep_aspect
            )
        
        try:
            train_dataset = make_dataset(fit_df, 'train', True, 'train')
            select_dataset = make_dataset(select_df, 'select', False, 'val')
            test_dataset = make_dataset(test_df, 'test', False, 'val')
        except Exception as e:
            print(f"  Error creating datasets: {e}")
            print(f"  Skipping fold {fold_idx+1}")
            continue
        
        if len(train_dataset) == 0 or len(select_dataset) == 0 or len(test_dataset) == 0:
            print(f"  Empty dataset, skipping fold {fold_idx+1}")
            continue
        
        loader_kwargs = dict(batch_size=args.batch_size, collate_fn=collate_fn,
                             num_workers=args.num_workers,
                             pin_memory=device.type == 'cuda')
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        select_loader = DataLoader(select_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        
        # Language indices must be consistent across the three splits: use the full-corpus mapping
        for ds in (train_dataset, select_dataset, test_dataset):
            ds.language_to_idx, ds.ws_to_idx = language_to_idx, ws_to_idx
        
        model = create_model(
            model_type=args.model,
            vocab_size=tokenizer.vocab_size,
            num_languages=len(language_to_idx),
            num_writing_systems=len(ws_to_idx),
            use_korniienko=args.use_korniienko,
            max_length=args.max_length,
            pretrained=args.pretrained,
            views=args.views,
            gradient_checkpointing=args.gradient_checkpointing
        )
        if args.init_from:
            load_initial_weights(model, args.init_from)
        model = model.to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(ignore_index=get_special_ids(tokenizer)[2])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        eval_kwargs = dict(use_rti=args.use_rti, use_korniienko=args.use_korniienko,
                           decoding=args.decoding, max_length=args.max_length,
                           use_language=args.use_language)
        
        # Training loop: select the best epoch on the model-selection split
        best_cer = float('inf')
        best_epoch = None
        best_model_state = None
        epochs_without_improvement = 0
        
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, tokenizer, device,
                                         args.use_rti, args.use_korniienko, args.use_language,
                                         args.modality_dropout, args.amp, args.gradient_accumulation)
            scheduler.step()
            
            sel_metrics = evaluate_model(model, select_loader, tokenizer, device, **eval_kwargs)
            print(f"  Epoch {epoch+1}/{args.epochs}: "
                  f"loss={train_loss:.4f}, "
                  f"select CER={sel_metrics['cer']:.4f}, "
                  f"SeqAcc={sel_metrics['sequence_accuracy']:.4f}")
            
            if sel_metrics['cer'] < best_cer:
                best_cer = sel_metrics['cer']
                best_epoch = epoch + 1
                # Deep copy: state_dict() tensors alias the live weights
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if args.patience and epochs_without_improvement >= args.patience:
                    print(f"  Early stopping after epoch {epoch+1} (best epoch {best_epoch})")
                    break
        
        # Load best model and evaluate ONCE on the untouched test fold
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        final_metrics = evaluate_model(model, test_loader, tokenizer, device, **eval_kwargs)
        
        # Sanity check: how much does performance depend on the images?
        ablation_metrics = {}
        if args.ablation_check:
            for mode in ('zero', 'shuffle'):
                m = evaluate_model(model, test_loader, tokenizer, device,
                                   **{**eval_kwargs, 'image_ablation': mode})
                ablation_metrics[f'cer_images_{mode}'] = m['cer']
                ablation_metrics[f'sequence_accuracy_images_{mode}'] = m['sequence_accuracy']
            print(f"  Image ablation: CER real={final_metrics['cer']:.4f}, "
                  f"zeroed={ablation_metrics['cer_images_zero']:.4f}, "
                  f"shuffled={ablation_metrics['cer_images_shuffle']:.4f}")
        
        fold_result = {
            'fold': fold_idx + 1,
            'train_size': len(train_dataset),
            'select_size': len(select_dataset),
            'val_size': len(test_dataset),
            'best_epoch': best_epoch,
            'cer': final_metrics['cer'],
            'char_accuracy': final_metrics['char_accuracy'],
            'sequence_accuracy': final_metrics['sequence_accuracy'],
            'decoding': args.decoding,
            **ablation_metrics
        }
        fold_results.append(fold_result)
        
        # Per-sample metadata straight from the (filtered, unshuffled) test dataset
        meta = test_dataset.df
        for i, (pred, target, cer_val) in enumerate(zip(
            final_metrics['predictions'], 
            final_metrics['targets'],
            final_metrics['per_sample_cer']
        )):
            row = meta.iloc[i] if i < len(meta) else {}
            all_predictions.append({
                'fold': fold_idx + 1,
                'sample_idx': i,
                'id': row.get('id', None) if len(meta) else None,
                'language_name': row.get('language_name', None) if len(meta) else None,
                'target': target,
                'prediction': pred,
                'cer': cer_val
            })
        
        # Save fold model
        if best_model_state is not None:
            torch.save({
                'model_state_dict': best_model_state,
                'epoch': best_epoch,
                'config': {**vars(args), 'tokenizer': 'trocr' if args.model == 'trocr' else 'character'},
                'num_languages': len(language_to_idx),
                'num_writing_systems': len(ws_to_idx),
                'language_to_idx': language_to_idx,
                'ws_to_idx': ws_to_idx,
                'vocab_size': tokenizer.vocab_size,
            }, output_dir / f"fold{fold_idx+1}_model.pt")
        
        print(f"\n  Fold {fold_idx+1} test: CER={final_metrics['cer']:.4f}, "
              f"SeqAcc={final_metrics['sequence_accuracy']:.4f} (best epoch {best_epoch})")
    
    # Check if we have any results
    if len(fold_results) == 0:
        print("\n❌ No folds completed successfully!")
        return None
    
    # Aggregate results
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS")
    print("="*70)
    
    cer_values = [r['cer'] for r in fold_results]
    char_acc_values = [r['char_accuracy'] for r in fold_results]
    seq_acc_values = [r['sequence_accuracy'] for r in fold_results]
    
    print(f"\n{len(fold_results)}-Fold Cross-Validation Results:")
    print("-"*50)
    
    for result in fold_results:
        print(f"  Fold {result['fold']}: CER={result['cer']:.4f}, "
              f"CharAcc={result['char_accuracy']:.4f}, "
              f"SeqAcc={result['sequence_accuracy']:.4f}")
    
    print("-"*50)
    print(f"\nAGGREGATE METRICS:")
    print(f"  CER:              {np.mean(cer_values):.4f} ± {np.std(cer_values):.4f}")
    print(f"  Char Accuracy:    {np.mean(char_acc_values):.4f} ± {np.std(char_acc_values):.4f}")
    print(f"  Sequence Accuracy: {np.mean(seq_acc_values):.4f} ± {np.std(seq_acc_values):.4f}")
    
    # 95% confidence interval
    n = len(cer_values)
    cer_ci = 1.96 * np.std(cer_values) / np.sqrt(n)
    print(f"\n  CER 95% CI: [{np.mean(cer_values) - cer_ci:.4f}, {np.mean(cer_values) + cer_ci:.4f}]")
    
    # Save results
    results_summary = {
        'model': args.model,
        'folds': args.folds,
        'epochs': args.epochs,
        'use_rti': args.use_rti,
        'use_korniienko': args.use_korniienko,
        'total_samples': len(df),
        'fold_results': fold_results,
        'aggregate': {
            'cer_mean': float(np.mean(cer_values)),
            'cer_std': float(np.std(cer_values)),
            'cer_ci_95': float(cer_ci),
            'char_accuracy_mean': float(np.mean(char_acc_values)),
            'char_accuracy_std': float(np.std(char_acc_values)),
            'sequence_accuracy_mean': float(np.mean(seq_acc_values)),
            'sequence_accuracy_std': float(np.std(seq_acc_values))
        },
        'decoding': args.decoding,
        'use_language': args.use_language,
        'image_size': args.image_size,
        'inner_val_frac': args.inner_val_frac,
    }
    for key in ('cer_images_zero', 'cer_images_shuffle'):
        vals = [r[key] for r in fold_results if key in r]
        if vals:
            results_summary['aggregate'][f'{key}_mean'] = float(np.mean(vals))
            print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    with open(output_dir / "cv_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save all predictions
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(output_dir / "all_predictions.csv", index=False)
    
    # Per-language analysis
    print("\n" + "="*70)
    print("PER-LANGUAGE PERFORMANCE")
    print("="*70)
    
    # Prefer per-row language_name captured at prediction-time; fall back to legacy mapping.
    if len(predictions_df) > 0 and 'language_name' in predictions_df.columns:
        predictions_df['_language'] = predictions_df['language_name'].fillna('Unknown')
    elif 'sample_idx' in predictions_df.columns and len(predictions_df) > 0:
        df['_language'] = df['language'].apply(get_language_text)

        # Legacy (imprecise) mapping: sample_idx is fold-local and may not correspond to global df.index.
        idx_to_lang = dict(zip(df.index, df['_language']))
        predictions_df['_language'] = predictions_df['sample_idx'].map(idx_to_lang)
    else:
        predictions_df['_language'] = 'Unknown'
    
    if len(predictions_df) > 0 and '_language' in predictions_df.columns:
        lang_performance = predictions_df.groupby('_language').agg({
            'cer': ['mean', 'std', 'count']
        }).round(4)
        lang_performance.columns = ['CER_mean', 'CER_std', 'count']
        lang_performance = lang_performance.sort_values('count', ascending=False)
        
        print(f"\n{'Language':<20} {'Count':>8} {'CER Mean':>10} {'CER Std':>10}")
        print("-"*50)
        for lang, row in lang_performance.iterrows():
            print(f"{lang:<20} {int(row['count']):>8} {row['CER_mean']:>10.4f} {row['CER_std']:>10.4f}")
        
        lang_performance.to_csv(output_dir / "per_language_performance.csv")
    else:
        print("(Could not compute per-language metrics - sample indices not available)")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print("="*70)
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation for Saint Sophia Graffiti')
    
    # Data arguments
    parser.add_argument('--data_csv', type=str, 
                       default='data/complete_dataset.csv',
                       help='Path to comprehensive data CSV')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Base data directory')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='enhanced',
                       choices=['multichannel', 'enhanced', 'transformer', 'trocr'],
                       help='Model architecture')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum token sequence length (incl. SOS/EOS)')
    parser.add_argument('--image_size', type=int, nargs='+', default=None,
                       help='One value (square) or HEIGHT WIDTH, e.g. 128 512')
    parser.add_argument('--use_language', action='store_true',
                       help='Condition the decoder on gold language labels (off by default)')
    
    # Cross-validation arguments
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs per fold')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--pretrained', type=str, default=DEFAULT_TROCR,
                       help='TrOCR checkpoint (model=trocr)')
    parser.add_argument('--views', type=str, default=','.join(DEFAULT_VIEWS),
                       help='TrOCR views: original,blended,normal,texture,photo,drawing')
    parser.add_argument('--init_from', type=str, default=None,
                       help='Initialise every fold from this checkpoint (e.g. synthetic pretraining)')
    parser.add_argument('--modality_dropout', type=float, default=0.0,
                       help='Training: hide each available modality with this probability')
    parser.add_argument('--keep_aspect', action='store_true',
                       help='Letterbox images instead of stretching them')
    parser.add_argument('--amp', action='store_true', help='bf16 mixed precision on CUDA')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--device', type=str, default=default_device())
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Modality arguments
    parser.add_argument('--use_rti', action='store_true',
                       help='Use RTI images')
    parser.add_argument('--use_korniienko', action='store_true', default=True,
                       help='Use Korniienko images')
    parser.add_argument('--no_korniienko', action='store_true',
                       help='Disable Korniienko images')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    # Evaluation protocol
    parser.add_argument('--decoding', type=str, default='autoregressive',
                       choices=['autoregressive', 'teacher_forced_legacy'],
                       help='autoregressive = real recognition from images; '
                            'teacher_forced_legacy = old protocol, only to reproduce earlier numbers')
    parser.add_argument('--inner_val_frac', type=float, default=0.1,
                       help='Share of each training portion held out to choose the best epoch')
    parser.add_argument('--patience', type=int, default=0,
                       help='Early stopping patience in epochs on the selection split (0 = off)')
    parser.add_argument('--ablation_check', action='store_true',
                       help='Also evaluate each fold with zeroed and shuffled images')
    
    # Quick mode
    parser.add_argument('--quick', action='store_true',
                       help='Quick validation mode (fewer epochs)')
    
    args = parser.parse_args()
    if args.image_size is None:
        args.image_size = [384] if args.model == 'trocr' else [224]
    args.image_size = args.image_size[0] if len(args.image_size) == 1 else args.image_size[:2]
    if args.lr is None:
        args.lr = 3e-5 if args.model == 'trocr' else 1e-4
    args.views = [v.strip() for v in args.views.split(',') if v.strip()]
    
    # Handle modality flags
    if args.no_korniienko:
        args.use_korniienko = False
    
    if args.quick:
        args.epochs = min(args.epochs, 3)
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Folds: {args.folds}")
    print(f"  Epochs per fold: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Use RTI: {args.use_rti}")
    print(f"  Use Korniienko: {args.use_korniienko}")
    
    run_cross_validation(args)


if __name__ == '__main__':
    main()

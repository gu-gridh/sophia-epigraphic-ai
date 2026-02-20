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
    create_model
)


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


def evaluate_model(model, dataloader, tokenizer, device):
    """Evaluate model using teacher forcing (matching evaluate.py approach)."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_cer = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Get inputs - matching evaluate.py structure
            rti_images = batch.get('rti_images')
            if rti_images is not None:
                rti_images = rti_images.to(device)
            
            korniienko_photo = batch.get('korniienko_photo')
            if korniienko_photo is not None:
                korniienko_photo = korniienko_photo.to(device)
            
            korniienko_drawing = batch.get('korniienko_drawing')
            if korniienko_drawing is not None:
                korniienko_drawing = korniienko_drawing.to(device)
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            languages = batch.get('language')
            if languages is not None:
                languages = languages.to(device)
            
            writing_systems = batch.get('writing_system')
            if writing_systems is not None:
                writing_systems = writing_systems.to(device)
            
            gt_transcriptions = batch.get('transcription', [])
            
            # Forward pass with teacher forcing (same as evaluate.py)
            try:
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=rti_images,
                    languages=languages,
                    writing_systems=writing_systems,
                    korniienko_photo=korniienko_photo,
                    korniienko_drawing=korniienko_drawing
                )
            except Exception as e:
                print(f"Error in forward pass: {e}")
                continue
            
            # Decode predictions (using argmax, same as evaluate.py)
            predictions = decode_prediction(logits, tokenizer)
            
            # Calculate CER for each sample
            for i, (pred, gt) in enumerate(zip(predictions, gt_transcriptions)):
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


def train_one_epoch(model, dataloader, optimizer, criterion, tokenizer, device):
    """Train for one epoch - matching train.py approach."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        # Get inputs - matching train.py structure exactly
        rti_images = batch.get('rti_images')
        if rti_images is not None:
            rti_images = rti_images.to(device)
        
        korniienko_photo = batch.get('korniienko_photo')
        if korniienko_photo is not None:
            korniienko_photo = korniienko_photo.to(device)
        
        korniienko_drawing = batch.get('korniienko_drawing')
        if korniienko_drawing is not None:
            korniienko_drawing = korniienko_drawing.to(device)
        
        # Target is input_ids - use directly without shifting (matching train.py)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        languages = batch.get('language')
        if languages is not None:
            languages = languages.to(device)
        
        writing_systems = batch.get('writing_system')
        if writing_systems is not None:
            writing_systems = writing_systems.to(device)
        
        # Forward pass - matching train.py signature exactly
        try:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=rti_images,
                languages=languages,
                writing_systems=writing_systems,
                korniienko_photo=korniienko_photo,
                korniienko_drawing=korniienko_drawing
            )
        except Exception as e:
            print(f"Error in forward pass: {e}")
            continue
        
        # Calculate loss - matching train.py: logits vs input_ids directly
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1)
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def run_cross_validation(args):
    """Run k-fold cross-validation."""
    print("="*70)
    print("SAINT SOPHIA GRAFFITI - K-FOLD CROSS-VALIDATION")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    tokenizer = CharacterTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"cv_{args.folds}fold_{args.model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp_splits"
    temp_dir.mkdir(exist_ok=True)
    
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
    for fold_idx, (train_df, val_df) in enumerate(folds):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{args.folds}")
        print(f"{'='*70}")
        print(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples")
        
        # Save fold data to temporary CSV files
        train_csv = temp_dir / f"fold{fold_idx+1}_train.csv"
        val_csv = temp_dir / f"fold{fold_idx+1}_val.csv"
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        
        # Create datasets using CSV files
        try:
            train_dataset = SophiaMultiModalDataset(
                csv_file=str(train_csv),
                data_dir=args.data_dir,
                tokenizer=tokenizer,
                max_length=args.max_length,
                use_rti=args.use_rti,
                use_korniienko=args.use_korniienko,
                model_type=args.model,
                augment=True,
                split='train'
            )
            
            val_dataset = SophiaMultiModalDataset(
                csv_file=str(val_csv),
                data_dir=args.data_dir,
                tokenizer=tokenizer,
                max_length=args.max_length,
                use_rti=args.use_rti,
                use_korniienko=args.use_korniienko,
                model_type=args.model,
                augment=False,
                split='val'
            )
        except Exception as e:
            print(f"  Error creating datasets: {e}")
            print(f"  Skipping fold {fold_idx+1}")
            continue
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"  Empty dataset, skipping fold {fold_idx+1}")
            continue
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Create model
        # Get number of languages and writing systems from datasets
        num_languages = len(train_dataset.language_to_idx) if hasattr(train_dataset, 'language_to_idx') else 12
        num_writing_systems = len(train_dataset.ws_to_idx) if hasattr(train_dataset, 'ws_to_idx') else 8
        
        model = create_model(
            model_type=args.model,
            vocab_size=tokenizer.vocab_size,
            num_languages=num_languages,
            num_writing_systems=num_writing_systems,
            use_korniienko=args.use_korniienko
        ).to(device)
        
        # Optimizer and loss
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_idx['<PAD>'])
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Training loop
        best_cer = float('inf')
        best_model_state = None
        
        for epoch in range(args.epochs):
            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, tokenizer, device)
            scheduler.step()
            
            # Evaluate
            val_metrics = evaluate_model(model, val_loader, tokenizer, device)
            
            print(f"  Epoch {epoch+1}/{args.epochs}: "
                  f"loss={train_loss:.4f}, "
                  f"CER={val_metrics['cer']:.4f}, "
                  f"CharAcc={val_metrics['char_accuracy']:.4f}, "
                  f"SeqAcc={val_metrics['sequence_accuracy']:.4f}")
            
            # Save best model
            if val_metrics['cer'] < best_cer:
                best_cer = val_metrics['cer']
                best_model_state = model.state_dict().copy()
        
        # Load best model and final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        final_metrics = evaluate_model(model, val_loader, tokenizer, device)
        
        # Store results
        fold_result = {
            'fold': fold_idx + 1,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'cer': final_metrics['cer'],
            'char_accuracy': final_metrics['char_accuracy'],
            'sequence_accuracy': final_metrics['sequence_accuracy']
        }
        fold_results.append(fold_result)

        # Build a per-sample metadata table aligned with the evaluated validation samples.
        # This avoids incorrect global-index mappings when computing per-language results.
        def _exists_rel(p):
            if p is None:
                return False
            if isinstance(p, float) and np.isnan(p):
                return False
            s = str(p).strip()
            if not s:
                return False
            base_dir = Path(args.data_dir)
            return (base_dir / s).exists() or Path(s).exists()

        filtered_val_df = val_df.copy()
        if 'transcription_clean' in filtered_val_df.columns:
            t = filtered_val_df['transcription_clean'].fillna('').astype(str)
            mask_t = t.str.len() >= 1
        else:
            mask_t = np.ones(len(filtered_val_df), dtype=bool)

        if args.use_korniienko:
            photo = filtered_val_df.get('korniienko_photo', pd.Series([''] * len(filtered_val_df)))
            draw = filtered_val_df.get('korniienko_drawing', pd.Series([''] * len(filtered_val_df)))
            iiif = filtered_val_df.get('iiif_crop', pd.Series([''] * len(filtered_val_df)))
            mask_img = [(_exists_rel(ph) or _exists_rel(dr) or _exists_rel(ic)) for ph, dr, ic in zip(photo, draw, iiif)]
        elif args.use_rti:
            # RTI / multi-modal cropped images: require all four channels.
            o = filtered_val_df.get('original_image', pd.Series([''] * len(filtered_val_df)))
            b = filtered_val_df.get('blended_image', pd.Series([''] * len(filtered_val_df)))
            nrm = filtered_val_df.get('normal_image', pd.Series([''] * len(filtered_val_df)))
            tx = filtered_val_df.get('texture_image', pd.Series([''] * len(filtered_val_df)))
            mask_img = [(_exists_rel(p1) and _exists_rel(p2) and _exists_rel(p3) and _exists_rel(p4)) for p1, p2, p3, p4 in zip(o, b, nrm, tx)]
        else:
            mask_img = np.ones(len(filtered_val_df), dtype=bool)

        filtered_val_df = filtered_val_df[pd.Series(mask_t) & pd.Series(mask_img)].reset_index(drop=True)

        if len(filtered_val_df) != len(final_metrics.get('predictions', [])):
            print(
                f"  ⚠ Fold {fold_idx+1}: metadata/prediction length mismatch "
                f"(meta={len(filtered_val_df)} vs preds={len(final_metrics.get('predictions', []))}); "
                "per-language export may be incomplete."
            )
        
        # Store predictions with fold info
        for i, (pred, target, cer_val) in enumerate(zip(
            final_metrics['predictions'], 
            final_metrics['targets'],
            final_metrics['per_sample_cer']
        )):
            sample_id = None
            lang_name = None
            if i < len(filtered_val_df):
                if 'id' in filtered_val_df.columns:
                    sample_id = filtered_val_df.iloc[i]['id']
                if 'language_name' in filtered_val_df.columns:
                    lang_name = filtered_val_df.iloc[i]['language_name']

            all_predictions.append({
                'fold': fold_idx + 1,
                'sample_idx': i,
                'id': sample_id,
                'language_name': lang_name,
                'target': target,
                'prediction': pred,
                'cer': cer_val
            })
        
        # Save fold model
        if best_model_state is not None:
            torch.save(best_model_state, output_dir / f"fold{fold_idx+1}_model.pt")
        
        print(f"\n  Fold {fold_idx+1} Best: CER={final_metrics['cer']:.4f}, "
              f"CharAcc={final_metrics['char_accuracy']:.4f}")
    
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
        }
    }
    
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
                       choices=['multichannel', 'enhanced', 'transformer'],
                       help='Model architecture')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    
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
    parser.add_argument('--lr', type=float, default=1e-4,
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
    
    # Quick mode
    parser.add_argument('--quick', action='store_true',
                       help='Quick validation mode (fewer epochs)')
    
    args = parser.parse_args()
    
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

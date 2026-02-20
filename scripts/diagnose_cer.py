#!/usr/bin/env python3
"""
Diagnostic tool to identify why CER is high despite low validation loss.
Tests tokenizer encoding/decoding and model predictions.
"""

import torch
import pandas as pd
from pathlib import Path
from transformers import XLMRobertaTokenizer

# Test tokenizer with Cyrillic text
print("="*80)
print("TOKENIZER DIAGNOSTIC")
print("="*80)

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Test samples (from training data)
test_texts = [
    "помꙗни",  # Simple Cyrillic
    "гпомо рабоусвоемоу",  # More complex
    "мцѧгенварѧпре ставсѧлꙗвъ кꙅдн",  # Ancient Cyrillic with special chars
    "ПИК",  # Short
    "Г М"  # Very short
]

print("\n1. ENCODING/DECODING TEST")
print("-"*80)
for text in test_texts:
    # Encode
    encoded = tokenizer(text, return_tensors='pt', padding='max_length', 
                       max_length=128, truncation=True)
    input_ids = encoded['input_ids'][0]
    
    # Decode
    decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
    
    # Check if roundtrip works
    match = (text.strip() == decoded.strip())
    
    print(f"\nOriginal:  '{text}'")
    print(f"Decoded:   '{decoded}'")
    print(f"Match:     {match} {'✓' if match else '✗'}")
    print(f"Tokens:    {input_ids[:10].tolist()}... (first 10)")

# Check vocab coverage
print("\n" + "="*80)
print("2. VOCABULARY COVERAGE TEST")
print("-"*80)

# Load training data
df = pd.read_csv('data/train_comprehensive.csv')
all_chars = set()
for text in df['transcription_clean'].dropna():
    all_chars.update(str(text))

print(f"\nUnique characters in training data: {len(all_chars)}")
print(f"Sample characters: {list(all_chars)[:50]}")

# Check which characters are unknown
unknown_chars = []
for char in all_chars:
    tokens = tokenizer.encode(char, add_special_tokens=False)
    # If character is split into multiple tokens or becomes UNK
    if len(tokens) > 1 or tokens[0] == tokenizer.unk_token_id:
        unknown_chars.append(char)

print(f"\nCharacters that don't have single tokens: {len(unknown_chars)}")
if unknown_chars:
    print(f"Examples: {unknown_chars[:20]}")

# Test predictions
print("\n" + "="*80)
print("3. MODEL PREDICTION TEST")
print("-"*80)

# Load predictions
pred_df = pd.read_csv('evaluation_results/enhanced_phase3_fixed/predictions.csv')

print(f"\nTotal test samples: {len(pred_df)}")

# Analyze prediction patterns
exact_matches = (pred_df['ground_truth'] == pred_df['prediction']).sum()
print(f"Exact matches: {exact_matches} ({exact_matches/len(pred_df)*100:.1f}%)")

# Check for common prediction issues
pred_df['pred_len'] = pred_df['prediction'].str.len()
pred_df['gt_len'] = pred_df['ground_truth'].str.len()
pred_df['len_diff'] = pred_df['pred_len'] - pred_df['gt_len']

print(f"\nLength statistics:")
print(f"  Avg GT length: {pred_df['gt_len'].mean():.1f}")
print(f"  Avg Pred length: {pred_df['pred_len'].mean():.1f}")
print(f"  Avg length diff: {pred_df['len_diff'].mean():.1f}")

# Check for common prefix issues
has_prefix_g = pred_df['prediction'].str.startswith('г ').sum()
has_prefix_c = pred_df['prediction'].str.startswith('С ').sum()
print(f"\nCommon prefix patterns:")
print(f"  Starts with 'г ': {has_prefix_g} ({has_prefix_g/len(pred_df)*100:.1f}%)")
print(f"  Starts with 'С ': {has_prefix_c} ({has_prefix_c/len(pred_df)*100:.1f}%)")

# Show worst examples
print("\n" + "="*80)
print("4. WORST PREDICTIONS (by length difference)")
print("-"*80)
worst = pred_df.nlargest(5, 'len_diff')
for _, row in worst.iterrows():
    print(f"\nID {row['isialy_id']}:")
    print(f"  GT ({len(str(row['ground_truth']))} chars): {str(row['ground_truth'])[:80]}")
    print(f"  Pred ({len(str(row['prediction']))} chars): {str(row['prediction'])[:80]}")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)

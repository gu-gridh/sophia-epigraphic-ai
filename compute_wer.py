"""Compute WER for our models and baselines to add to the paper."""
import csv
import numpy as np

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]

def wer(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)

# Our model predictions
print("=== OUR MODELS ===")
for name, path in [
    ('Multi-Channel CNN', 'evaluation_results/full_experiments_20260205_155232/multichannel/cv_5fold_multichannel_20260205_220228/all_predictions.csv'),
    ('Enhanced CNN', 'evaluation_results/full_experiments_20260205_155232/enhanced/cv_5fold_enhanced_20260205_155235/all_predictions.csv'),
    ('Transformer', 'evaluation_results/full_experiments_20260205_155232/transformer/cv_5fold_transformer_20260205_185851/all_predictions.csv'),
]:
    wers = []
    with open(path) as f:
        for row in csv.DictReader(f):
            wers.append(wer(row['target'], row['prediction']))
    wers = np.array(wers)
    print(f"{name}: WER = {wers.mean()*100:.2f}% +/- {wers.std()*100:.2f}%")
    if 'Multi' in name:
        print(f"  Median WER: {np.median(wers)*100:.2f}%")
        print(f"  Perfect (WER=0): {(wers==0).sum()}/{len(wers)} ({(wers==0).mean()*100:.1f}%)")

# Baselines
print("\n=== BASELINES ===")
for name, path in [
    ('EasyOCR', 'evaluation_results/baselines/easyocr/detailed_results.csv'),
    ('TrOCR', 'evaluation_results/baselines/trocr_base/detailed_results.csv'),
    ('VLM', 'evaluation_results/baselines/vlm_qwen25vl7b/detailed_results.csv'),
]:
    wers = []
    with open(path) as f:
        for row in csv.DictReader(f):
            gt = row['ground_truth']
            pred = row.get('cleaned_prediction', row.get('prediction', ''))
            wers.append(wer(gt, pred))
    wers = np.array(wers)
    print(f"{name}: WER = {wers.mean()*100:.2f}% +/- {wers.std()*100:.2f}%")

# Per-language WER for our best model
print("\n=== PER-LANGUAGE WER (Multi-Channel) ===")
dataset = {}
with open('data/complete_dataset.csv') as f:
    for row in csv.DictReader(f):
        dataset[row.get('transcription_clean', '')] = row.get('language_name', '')

pred_file = 'evaluation_results/full_experiments_20260205_155232/multichannel/cv_5fold_multichannel_20260205_220228/all_predictions.csv'
lang_wers = {}
with open(pred_file) as f:
    for row in csv.DictReader(f):
        t = row['target']
        lang = dataset.get(t, 'Unknown')
        if lang not in lang_wers:
            lang_wers[lang] = []
        lang_wers[lang].append(wer(t, row['prediction']))

for lang in sorted(lang_wers, key=lambda x: np.mean(lang_wers[x])):
    arr = np.array(lang_wers[lang])
    print(f"  {lang:20s}: WER={arr.mean()*100:.2f}% (n={len(arr)})")

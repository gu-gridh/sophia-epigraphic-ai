#!/bin/bash
# =============================================================================
# Pretrained TrOCR + synthetic pretraining experiments (run on a CUDA GPU)
# =============================================================================
#
#   1. Generate synthetic incised inscriptions (all views) from TRAINING text only
#   2. Pretrain the Cyrillic TrOCR on the synthetic data
#   3. 5-fold CV on the real corpus:
#        a) TrOCR fine-tuned directly from the public checkpoint
#        b) TrOCR fine-tuned from the synthetic-pretrained checkpoint
#        c) Enhanced CNN from scratch (reference)
#      every run reports CER with real, zeroed and shuffled images (--ablation_check)
#
# Notes
# - Image paths in the CSVs are relative to DATA_DIR.
# - The synthetic text pool uses re-composed words/fragments/random strings from the
#   training CSV, never exact lines, and skips anything in the val/test CSVs. In 5-fold CV
#   words of a test fold can still appear in the pool; for the strictest number, report
#   the held-out test CSV result (step 4) as well.
# - If you run out of GPU memory: lower BATCH, add --gradient_checkpointing, or use
#   --gradient_accumulation 2.
# =============================================================================
set -euo pipefail

DATA_DIR=${DATA_DIR:-data}
FULL_CSV=${FULL_CSV:-$DATA_DIR/complete_dataset.csv}
TRAIN_CSV=${TRAIN_CSV:-$DATA_DIR/train_comprehensive.csv}
VAL_CSV=${VAL_CSV:-$DATA_DIR/val_comprehensive.csv}
TEST_CSV=${TEST_CSV:-$DATA_DIR/test_comprehensive.csv}
SYN_NUM=${SYN_NUM:-30000}
WORKERS=${WORKERS:-8}
BATCH=${BATCH:-8}

COMMON_TROCR="--model trocr --use_rti --use_korniienko --views original,normal,photo,drawing \
  --image_size 192 576 --keep_aspect --max_length 384 --amp"

# ---- 1. synthetic data ------------------------------------------------------
if [ ! -f "$DATA_DIR/synthetic/train.csv" ]; then
  python scripts/generate_synthetic.py \
    --text_csv "$TRAIN_CSV" --exclude_csv "$VAL_CSV" "$TEST_CSV" \
    --font_dirs "$DATA_DIR/fonts" \
    --out_dir "$DATA_DIR/synthetic" --data_dir "$DATA_DIR" \
    --num "$SYN_NUM" --workers "$WORKERS"
fi

# ---- 2. synthetic pretraining ----------------------------------------------
python train.py $COMMON_TROCR --data_dir "$DATA_DIR" \
  --train_csv synthetic/train.csv --val_csv synthetic/val.csv \
  --epochs 3 --batch_size "$BATCH" --lr 5e-5 --modality_dropout 0.3 \
  --num_workers "$WORKERS" --checkpoint_dir checkpoints/trocr_synthetic
SYN_CKPT=$(ls -td checkpoints/trocr_synthetic/trocr/*/ | head -1)best_model.pt
echo "Synthetic-pretrained checkpoint: $SYN_CKPT"

# ---- 3. cross-validation on the real corpus ---------------------------------
CV_COMMON="--data_csv $FULL_CSV --data_dir $DATA_DIR --folds 5 --epochs 20 \
  --batch_size $BATCH --patience 4 --ablation_check --num_workers $WORKERS"

python cross_validate.py $COMMON_TROCR $CV_COMMON --modality_dropout 0.2 \
  --output_dir evaluation_results/trocr_public_init

python cross_validate.py $COMMON_TROCR $CV_COMMON --modality_dropout 0.2 \
  --init_from "$SYN_CKPT" --output_dir evaluation_results/trocr_synthetic_init

python cross_validate.py --model enhanced --use_rti --use_korniienko $CV_COMMON --epochs 30 \
  --keep_aspect --output_dir evaluation_results/enhanced_scratch

# ---- 4. held-out test split (strictest number) -------------------------------
python train.py $COMMON_TROCR --data_dir "$DATA_DIR" \
  --train_csv "$(basename "$TRAIN_CSV")" --val_csv "$(basename "$VAL_CSV")" \
  --epochs 20 --batch_size "$BATCH" --modality_dropout 0.2 --init_from "$SYN_CKPT" \
  --num_workers "$WORKERS" --checkpoint_dir checkpoints/trocr_final
FINAL_CKPT=$(ls -td checkpoints/trocr_final/trocr/*/ | head -1)best_model.pt
for ABL in none zero shuffle; do
  python evaluate.py --model trocr --checkpoint "$FINAL_CKPT" --use_rti --use_korniienko \
    --data_dir "$DATA_DIR" --test_csv "$(basename "$TEST_CSV")" --image_ablation "$ABL" \
    --output_dir "evaluation_results/trocr_final_test_$ABL"
done

#!/bin/bash
cd /home/aram/GRIDH/Saint_Sophia/sophia-epigraphic-ai
conda activate sophia-ai
python evaluate.py \
  --model transformer \
  --checkpoint checkpoints/transformer_phase3_full/transformer/20251016_112905/best_model.pt \
  --use_rti \
  --use_korniienko \
  --data_dir . \
  --test_csv data/val_comprehensive.csv \
  --output_dir evaluation_results/transformer_phase3_val

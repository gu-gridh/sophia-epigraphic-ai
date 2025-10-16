#!/bin/bash

# Train models with RTI and Korniienko data
python train.py --model transformer --use_rti --use_korniienko --epochs 20 --batch_size 8 --data_dir . --checkpoint_dir checkpoints/transformer_phase3_full
python train.py --model enhanced --use_rti --use_korniienko --epochs 20 --batch_size 8 --data_dir . --checkpoint_dir checkpoints/enhanced_phase3_full
python train.py --model multichannel --use_rti --use_korniienko --epochs 20 --batch_size 8 --data_dir . --checkpoint_dir checkpoints/multichannel_phase3_full

# Evaluate models with RTI and Korniienko data
python evaluate.py --model transformer --checkpoint checkpoints/transformer_phase3_full/transformer/datestampfolder/best_model.pt --use_rti --use_korniienko --data_dir . --test_csv data/test_comprehensive.csv --output_dir evaluation_results/transformer_phase3_new 2>&1 | tail -n 60
python evaluate.py --model enhanced --checkpoint checkpoints/enhanced_phase3_full/enhanced/datestampfolder/best_model.pt --use_rti --use_korniienko --data_dir . --test_csv data/test_comprehensive.csv --output_dir evaluation_results/enhanced_phase3_new 2>&1 | tail -n 60
python evaluate.py --model multichannel --checkpoint checkpoints/multichannel_phase3_full/multichannel/datestampfolder/best_model.pt --use_rti --use_korniienko --data_dir . --test_csv data/test_comprehensive.csv --output_dir evaluation_results/multichannel_phase3_new 2>&1 | tail -n 60
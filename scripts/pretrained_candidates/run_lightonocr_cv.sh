#!/bin/bash
# 5-fold LoRA fine-tuning of LightOnOCR-2-1B (starting from the Old Church Slavonic adapter).
# Folds and inner model-selection splits are identical to cross_validate.py, so the numbers
# can be compared directly with the TrOCR / CNN runs.
#
# Needs a separate environment with transformers>=5 and peft (do NOT upgrade the project venv):
#   python -m venv .venv-vlm && .venv-vlm/bin/pip install torch torchvision "transformers>=5" peft accelerate pillow numpy
#
# Usage: VPY=.venv-vlm/bin/python bash scripts/pretrained_candidates/run_lightonocr_cv.sh
set -euo pipefail
VPY=${VPY:-python}
DATA_DIR=${DATA_DIR:-data}
FULL_CSV=${FULL_CSV:-$DATA_DIR/complete_dataset.csv}
VIEWS=${VIEWS:-drawing,photo}
OUT=${OUT:-runs/lightonocr_ocs_cv}

for FOLD in 1 2 3 4 5; do
  $VPY scripts/pretrained_candidates/finetune_vlm_lora.py \
    --cv_csv "$FULL_CSV" --data_dir "$DATA_DIR" --folds 5 --fold "$FOLD" \
    --views "$VIEWS" --epochs 10 --patience 3 \
    --batch_size 4 --grad_accum 2 --eval_batch_size 8 --lr 1e-4 \
    --longest_edge 1024 --gradient_checkpointing \
    --out "$OUT/fold$FOLD"
done

# Aggregate test results over folds
$VPY - "$OUT" <<'EOF'
import json, sys, glob, statistics
runs = sorted(glob.glob(f"{sys.argv[1]}/fold*/results.json"))
for view in ('drawing', 'photo', 'all'):
    vals = {k: [json.load(open(r))['test'][view][k] for r in runs if view in json.load(open(r)).get('test', {})]
            for k in ('cer_raw', 'cer_norm', 'exact_norm')}
    if vals['cer_raw']:
        print(f"{view:8s} folds={len(vals['cer_raw'])}  " + '  '.join(
            f"{k}={statistics.mean(v):.1%}±{statistics.pstdev(v):.1%}" for k, v in vals.items()))
EOF

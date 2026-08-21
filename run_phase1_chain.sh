#!/usr/bin/env bash
# Compact TASK_SPEC 6.3 phase-1: 84 inject seq * 2 ctx * 3 seed * 1 layer
# (>=500 / condition), 196 cal seq for 50k clean tokens, 20-seq test check.
set -euo pipefail
cd /opt/data/data/tolerance-exp
# shellcheck disable=SC1091
source /opt/data/data/anaconda3/bin/activate vllm0.8.5
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
MODEL=/opt/data/data/models/Qwen3-8B
OUTA=/opt/data/data/tolerance-exp/results/expA_phase1
OUTB=/opt/data/data/tolerance-exp/results/expB_phase1
OUTC=/opt/data/data/tolerance-exp/results/expC_phase1

echo "[chain] A calibration $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python run_exp_a.py --profile phase1 --model-path "$MODEL" --output "$OUTA" --resume --split calibration
echo "[chain] A test $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python run_exp_a.py --profile phase1 --model-path "$MODEL" --output "$OUTA" --resume --split test --skip-consistency
echo "[chain] B calibration $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python run_exp_b.py --profile phase1 --model-path "$MODEL" --output "$OUTB" --resume --split calibration \
  --s-rec "$OUTA/tables/recommendation.json"
echo "[chain] B test $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python run_exp_b.py --profile phase1 --model-path "$MODEL" --output "$OUTB" --resume --split test --skip-consistency \
  --s-rec "$OUTA/tables/recommendation.json"
echo "[chain] C $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python run_exp_c.py --profile phase1 --model-path "$MODEL" --from-b "$OUTB" --output "$OUTC" \
  --s-rec "$OUTA/tables/recommendation.json"
echo "[chain] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"

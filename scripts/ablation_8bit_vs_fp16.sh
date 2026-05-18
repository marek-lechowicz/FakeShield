#!/bin/bash
set -e

# Ablation: 8-bit vs fp16+CPU offload on 10 DSO-1 images.
# Subset = 5 normals where 8-bit produced false positives + 5 spliced true positives.
# Goal: see whether moving to fp16 fixes the FP-on-clean problem.

WEIGHT_PATH=./weight/fakeshield-v1-22b
BASELINE_8BIT=./playground/dso1_dte_fdm.jsonl              # already exists
SUBSET_JSONL=./playground/dso1_ablation_subset.jsonl
DTE_FP16_OUTPUT=./playground/dso1_dte_fdm_fp16_subset.jsonl

# 1. Build subset (5 normal-FPs + 5 spliced-TPs based on 8-bit run)
python3 scripts/build_ablation_subset.py \
    --question-jsonl ./playground/dso1_question.jsonl \
    --baseline-output "$BASELINE_8BIT" \
    --output-jsonl "$SUBSET_JSONL" \
    --n-per-class 5

# 2. Run DTE-FDM in fp16 + CPU offload, greedy (temperature=0 for determinism),
#    short max_new_tokens because verdict lives in the first ~50 tokens.
echo "==== fp16 + CPU offload run on 10 images (this takes ~10-20 min) ===="
pip install -q transformers==4.37.2 > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 \
python ./DTE-FDM/llava/eval/model_vqa.py \
    --model-path ${WEIGHT_PATH}/DTE-FDM \
    --DTG-path ${WEIGHT_PATH}/DTG.pth \
    --question-file "$SUBSET_JSONL" \
    --image-folder / \
    --answers-file "$DTE_FP16_OUTPUT" \
    --cpu-offload-gib 14 \
    --temperature 0 \
    --max-new-tokens 256

# 3. Compare verdicts side-by-side
echo
echo "==== Verdicts: 8-bit vs fp16+offload ===="
python3 scripts/compare_ablation.py \
    --subset-jsonl "$SUBSET_JSONL" \
    --baseline-output "$BASELINE_8BIT" \
    --fp16-output "$DTE_FP16_OUTPUT"

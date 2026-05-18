#!/bin/bash
set -e

# DSO-1 evaluation pipeline:
#   * DTE-FDM in 8-bit (~13 GB VRAM, near-lossless)
#   * MFLM in native bf16 (~16 GB VRAM)
# Reports:
#   * pixel-level IoU on the 100 spliced images (vs DSO-1 GT masks)
#   * image-level Accuracy / Precision / Recall / MCC / AUC over all 200 images

WEIGHT_PATH=./weight/fakeshield-v1-22b
DSO_ROOT=./dataset/tifs-database/tifs-database/DSO-1
MASK_ROOT=./dataset/tifs-database/tifs-database/DSO-1-Fake-Images-Masks

QUESTION_JSONL=./playground/dso1_question.jsonl
DTE_FDM_OUTPUT=./playground/dso1_dte_fdm.jsonl
MFLM_OUTPUT=./playground/dso1_mflm_output

mkdir -p "$MFLM_OUTPUT"

# echo "==== Step 1: Preparing question file ===="
# python scripts/prepare_dso1_jsonl.py \
#     --dso-root "$DSO_ROOT" \
#     --mask-root "$MASK_ROOT" \
#     --output-jsonl "$QUESTION_JSONL"

# echo "==== Step 2: DTE-FDM Detection (8-bit) ===="
# pip install -q transformers==4.37.2 > /dev/null 2>&1
# CUDA_VISIBLE_DEVICES=0 \
# BNB_CUDA_VERSION=118 \
# LD_LIBRARY_PATH=$PWD/.venv/lib/python3.9/site-packages/nvidia/cusparse/lib:$PWD/.venv/lib/python3.9/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH \
# python ./DTE-FDM/llava/eval/model_vqa.py \
#     --model-path ${WEIGHT_PATH}/DTE-FDM \
#     --DTG-path ${WEIGHT_PATH}/DTG.pth \
#     --question-file "$QUESTION_JSONL" \
#     --image-folder / \
#     --answers-file "$DTE_FDM_OUTPUT" \
#     --load-8bit

echo "==== Step 3: MFLM Localization (bf16) ===="
pip install -q transformers==4.28.0 > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 \
python ./MFLM/test.py \
    --version ${WEIGHT_PATH}/MFLM \
    --DTE-FDM-output "$DTE_FDM_OUTPUT" \
    --MFLM-output "$MFLM_OUTPUT"

echo "==== Step 4: Metrics ===="
python scripts/calculate_metrics_dso1.py \
    --question-jsonl "$QUESTION_JSONL" \
    --dte-output "$DTE_FDM_OUTPUT" \
    --mflm-dir "$MFLM_OUTPUT" \
    --invert-gt-mask

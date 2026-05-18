#!/bin/bash

# Configuration
LIST_PATH=$1
if [ -z "$LIST_PATH" ]; then
    echo "Usage: $0 <path_to_list_file.txt>"
    exit 1
fi

SHORT_NAME=$(basename "$LIST_PATH" _test_list.txt)
SHORT_NAME=${SHORT_NAME#fakeflickr_}

WEIGHT_PATH=./weight/fakeshield-v1-22b
QUESTION_JSONL=./playground/fakeflickr_${SHORT_NAME}_question.jsonl
DTE_FDM_OUTPUT=./playground/fakeflickr_${SHORT_NAME}_dte_fdm.jsonl
MFLM_OUTPUT=./playground/fakeflickr_${SHORT_NAME}_mflm_output

mkdir -p "$MFLM_OUTPUT"

# 1. Prepare Question File
echo "==== Step 1: Preparing Question File ===="
python scripts/prepare_fakeflickr_jsonl.py \
    --list-path "$LIST_PATH" \
    --output-jsonl "$QUESTION_JSONL"

# 2. Stage 1: DTE-FDM Detection
echo "==== Step 2: DTE-FDM Detection ===="
pip install -q transformers==4.37.2  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 \
BNB_CUDA_VERSION=118 \
LD_LIBRARY_PATH=$PWD/.venv/lib/python3.9/site-packages/nvidia/cusparse/lib:$PWD/.venv/lib/python3.9/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH \
python ./DTE-FDM/llava/eval/model_vqa.py \
    --model-path ${WEIGHT_PATH}/DTE-FDM  \
    --DTG-path ${WEIGHT_PATH}/DTG.pth \
    --question-file "$QUESTION_JSONL" \
    --image-folder / \
    --answers-file "$DTE_FDM_OUTPUT" \
    --load-4bit

# 3. Stage 2: MFLM Localization
echo "==== Step 3: MFLM Localization ===="
pip install -q transformers==4.28.0  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 \
BNB_CUDA_VERSION=118 \
LD_LIBRARY_PATH=$PWD/.venv/lib/python3.9/site-packages/nvidia/cusparse/lib:$PWD/.venv/lib/python3.9/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH \
python ./MFLM/test.py \
    --version ${WEIGHT_PATH}/MFLM \
    --DTE-FDM-output "$DTE_FDM_OUTPUT" \
    --MFLM-output "$MFLM_OUTPUT" \
    --load-4bit

# 4. Calculate Metrics
echo "==== Step 4: Metric Calculation ===="
python scripts/calculate_metrics.py \
    --question-jsonl "$QUESTION_JSONL" \
    --output-dir "$MFLM_OUTPUT"

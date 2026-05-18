WEIGHT_PATH=./weight/fakeshield-v1-22b
IMAGE_PATH=./playground/images/14813436.png
DTE_FDM_OUTPUT=./playground/DTE-FDM_output.jsonl
MFLM_OUTPUT=./playground/MFLM_output

# bitsandbytes: torch is +cu116 but the installed bnb wheel only ships the
# cuda118 binary, so we force bnb to cuda118 and point LD at the bundled
# NVIDIA cu118 libs. Needed whenever DTE-FDM uses --load-4bit or --load-8bit.

# DTE-FDM
# Pick ONE quantization mode below by uncommenting:
#   --load-4bit         : ~7 GB VRAM, fastest, ~1-3% quality drop
#   --load-8bit         : ~13 GB VRAM, near-lossless (recommended for paper)
#   --cpu-offload-gib N : fp16 + CPU offload, paper-grade quality, slow
pip install -q transformers==4.37.2  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 \
BNB_CUDA_VERSION=118 \
LD_LIBRARY_PATH=$PWD/.venv/lib/python3.9/site-packages/nvidia/cusparse/lib:$PWD/.venv/lib/python3.9/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH \
python3 -m llava.serve.cli \
    --model-path  ${WEIGHT_PATH}/DTE-FDM \
    --DTG-path ${WEIGHT_PATH}/DTG.pth \
    --image-path ${IMAGE_PATH} \
    --output-path ${DTE_FDM_OUTPUT} \
    --load-8bit
    # --load-8bit
    # --cpu-offload-gib 20

# MFLM: native bf16 (~16 GB), fits entirely on a 24 GB GPU. No quantization.
pip install -q transformers==4.28.0  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 \
python3 ./MFLM/cli_demo.py \
    --version ${WEIGHT_PATH}/MFLM \
    --DTE-FDM-output ${DTE_FDM_OUTPUT} \
    --MFLM-output ${MFLM_OUTPUT}

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# activate venv if local
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# ---------------------------------------------------------------------------
# Model: Qwen2.5-3B-Instruct Q8_0 GGUF
# ---------------------------------------------------------------------------
MODEL_DIR="$SCRIPT_DIR/qwen2.5-3b-instruct-q8"
MODEL_FILE="$MODEL_DIR/qwen2.5-3b-instruct-q8_0.gguf"
HF_REPO_ID="Qwen/Qwen2.5-3B-Instruct-GGUF"
HF_FILENAME="qwen2.5-3b-instruct-q8_0.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Model not found at $MODEL_FILE — downloading from HuggingFace..."
    mkdir -p "$MODEL_DIR"
    python - <<EOF
from huggingface_hub import hf_hub_download
import shutil, os

dest = hf_hub_download(
    repo_id="$HF_REPO_ID",
    filename="$HF_FILENAME",
    local_dir="$MODEL_DIR",
)
print(f"Downloaded to: {dest}")
EOF
    echo "Download complete."
else
    echo "Model already present at $MODEL_FILE — skipping download."
fi

# start infinity in background
# infinity_emb v2 \
#     --model-id NeuML/pubmedbert-base-embeddings \
#     --model-id BAAI/bge-reranker-base \
#     --port "${INFINITY_PORT:-7997}" &
# echo "Infinity started on port ${INFINITY_PORT:-7997}"
# run mainly on gpu if gpu is available, or else automatically run on cpu

# start llama-cpp in foreground
echo "Starting llama-cpp on port ${LLAMA_PORT:-8001}..."
python -m llama_cpp.server \
    --model "$MODEL_FILE" \
    --n_gpu_layers -1 \
    --n_ctx 8192 \
    --port "${LLAMA_PORT:-8001}"
# llama-server \
#     --model "$MODEL_FILE" \
#     --n-gpu-layers -1 \
#     --ctx-size 8192 \
#     --port "${LLAMA_PORT:-8001}"
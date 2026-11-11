#!/usr/bin/env bash
set -euo pipefail
root=/home/jdillman/work/flashnext-integration
source /home/jdillman/work/rtx8000-toolchain/toolchain.env
source "$root/.venv/bin/activate"
: "${VLLM_API_KEY:?Set VLLM_API_KEY in the protected service environment file}"
export HOST=192.168.0.51 PORT=8000
export VLLM_SERVER_DEV_MODE=0
export ENABLE_VISION=1 ENABLE_PREFIX_CACHING=1 MAX_MODEL_LEN=131200 MAX_NUM_SEQS=5
export MAX_BATCHED_TOKENS=2048 GPU_MEMORY_UTILIZATION=0.90
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$root/source"
exec bash "$root/deployment/serve_flashnext_rtx8000.sh" \
  "$root/models/Qwen3.8-Flash-Next-experts-W8A16-channel" \
  --served-model-name Qwen3.8-Flash-Next-W8A16 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}'

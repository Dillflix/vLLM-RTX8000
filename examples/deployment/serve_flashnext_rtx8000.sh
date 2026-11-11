#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Tested TP2 PP2 profile; qualification scope is in docs/features/engram.md.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/Qwen3.8-Flash-Next-FP8 [additional vllm serve arguments]" >&2
  exit 2
fi
model=$1
shift

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export VLLM_SKIP_P2P_CHECK=0
export VLLM_RTX8000_NATIVE_QSA=${VLLM_RTX8000_NATIVE_QSA:-1}
export VLLM_RTX8000_NATIVE_DENSE=${VLLM_RTX8000_NATIVE_DENSE:-1}
export VLLM_RTX8000_GDN_TILES=${VLLM_RTX8000_GDN_TILES:-1}
export VLLM_RTX8000_NATIVE_GDN=${VLLM_RTX8000_NATIVE_GDN:-1}
export VLLM_RTX8000_NATIVE_GDN_H=${VLLM_RTX8000_NATIVE_GDN_H:-1}
export VLLM_RTX8000_MARLIN_TILES=${VLLM_RTX8000_MARLIN_TILES:-1}
read -r -a numa_nodes <<< "${NUMA_BIND_NODES:-0 0 1 1}"
read -r -a numa_cpus <<< "${NUMA_BIND_CPUS:-0-9,20-29 0-9,20-29 10-19,30-39 10-19,30-39}"
if [[ ${#numa_nodes[@]} -ne 4 || ${#numa_cpus[@]} -ne 4 ]]; then
  echo "NUMA_BIND_NODES and NUMA_BIND_CPUS each require four space-separated entries." >&2
  exit 2
fi
command -v numactl >/dev/null
case ${ENABLE_PREFIX_CACHING:-1} in
  1) prefix_cache_flag=--enable-prefix-caching ;;
  0) prefix_cache_flag=--no-enable-prefix-caching ;;
  *) echo "ENABLE_PREFIX_CACHING must be 0 or 1." >&2; exit 2 ;;
esac

case ${ENABLE_VISION:-0} in
  1)
    multimodal_flags=(
      --mm-encoder-attn-backend TORCH_SDPA
      --limit-mm-per-prompt '{"image":{"count":4,"width":1024,"height":1024},"video":{"count":1,"num_frames":16,"width":512,"height":512}}'
      --media-io-kwargs '{"video":{"num_frames":16}}'
      --mm-processor-kwargs '{"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":1048576}},"videos_kwargs":{"num_frames":16,"size":{"shortest_edge":4096,"longest_edge":262144}}}'
    )
    ;;
  0) multimodal_flags=(--language-model-only) ;;
  *) echo "ENABLE_VISION must be 0 or 1." >&2; exit 2 ;;
esac

exec vllm serve "$model" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-18381}" \
  --served-model-name Qwen3.8-Flash-Next-FP8 \
  --dtype half --kv-cache-dtype float16 \
  --tensor-parallel-size 2 --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --numa-bind --numa-bind-nodes "${numa_nodes[@]}" \
  --numa-bind-cpus "${numa_cpus[@]}" \
  --kernel-config '{"moe_backend":"marlin","linear_backend":"marlin"}' \
  --attention-config '{"indexer_kv_dtype":"fp16"}' \
  --engram-config '{"cpu_offload":true,"disk_offload":false}' \
  "${multimodal_flags[@]}" --generation-config vllm \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --max-num-seqs "${MAX_NUM_SEQS:-5}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-2048}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --enable-chunked-prefill "$prefix_cache_flag" \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","decode_cudagraph_capture_request_sizes":[1,2,3,4,5]}' \
  "$@"

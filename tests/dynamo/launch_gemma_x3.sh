#!/usr/bin/env bash
# Dynamo pilot: frontend (KV-aware router) + N dynamo.vllm workers on ONE GPU.
# Runs INSIDE nvcr.io/nvidia/ai-dynamo/vllm-runtime (see run_pilot.sh). Mirrors the
# official examples/backends/vllm/launch/agg_router.sh.
#
#   PRESET=gemma-x3  (default) google/gemma-4-E4B-it ×3 co-located, our recipe from docs/vllm-colocation.md
#   PRESET=gptoss    openai/gpt-oss-120b ×1, same engine settings as production (max_len 32768, util 0.95)
set -euo pipefail
trap 'echo "[launch] cleaning up"; kill 0' EXIT

PRESET="${PRESET:-gemma-x3}"
HTTP_PORT="${HTTP_PORT:-18000}"
ROUTER_MODE="${ROUTER_MODE:-kv}"          # kv | round-robin | least-loaded ...
BLOCK_SIZE="${BLOCK_SIZE:-64}"            # must match on router + every worker
STAGGER_S="${STAGGER_S:-25}"              # gap between worker starts (profiling overlap)
EXTRA_WORKER_ARGS="${EXTRA_WORKER_ARGS:-}"

case "$PRESET" in
  gemma-x3)
    MODEL="${MODEL:-google/gemma-4-E4B-it}"; NUM_WORKERS="${NUM_WORKERS:-3}"
    ENGINE_ARGS="--gpu-memory-utilization ${GPU_UTIL:-0.32} --kv-cache-memory-bytes ${KV_BYTES:-8589934592} \
      --max-model-len ${MAX_LEN:-32768} --max-num-seqs ${MAX_SEQS:-512} --max-num-batched-tokens ${MAX_BATCHED:-4096}"
    PARSER_ARGS="--dyn-reasoning-parser gemma4 --dyn-tool-call-parser gemma4" ;;
  gptoss)
    MODEL="${MODEL:-openai/gpt-oss-120b}"; NUM_WORKERS="${NUM_WORKERS:-1}"
    ENGINE_ARGS="--gpu-memory-utilization ${GPU_UTIL:-0.95} --max-model-len ${MAX_LEN:-32768}"
    PARSER_ARGS="--dyn-reasoning-parser gpt_oss --dyn-tool-call-parser harmony" ;;
  *) echo "unknown PRESET=$PRESET"; exit 2 ;;
esac

# Deterministic hashing across processes is required for KV-event radix trees.
export PYTHONHASHSEED=0
export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV="${DYN_FILE_KV:-/tmp/dynamo_store_kv}"
export DYN_LOG="${DYN_LOG:-info}"
rm -rf "$DYN_FILE_KV"; mkdir -p "$DYN_FILE_KV"

echo "[launch] preset=$PRESET model=$MODEL workers=$NUM_WORKERS frontend :$HTTP_PORT router-mode=$ROUTER_MODE block=$BLOCK_SIZE"
python3 -m dynamo.frontend \
  --http-port "$HTTP_PORT" \
  --router-mode "$ROUTER_MODE" \
  --kv-cache-block-size "$BLOCK_SIZE" \
  > /logs/frontend.log 2>&1 &

# ONE_GPU_PER_WORKER=1: container sees N GPUs, worker i is pinned to GPU i (multi-GPU
# replica test). Default (unset): all workers share the single visible GPU (co-location).
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
for i in $(seq 0 $((NUM_WORKERS - 1))); do
  SYS_PORT=$((18081 + i))
  KV_PORT=$((20080 + i))
  if [ "${ONE_GPU_PER_WORKER:-0}" = "1" ]; then export CUDA_VISIBLE_DEVICES=$((i % NGPU)); fi
  echo "[launch] worker $i  system:$SYS_PORT kv-events:$KV_PORT gpu=${CUDA_VISIBLE_DEVICES:-shared}"
  DYN_SYSTEM_PORT=$SYS_PORT \
  python3 -m dynamo.vllm \
    --model "$MODEL" \
    --block-size "$BLOCK_SIZE" \
    $ENGINE_ARGS \
    $PARSER_ARGS \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$KV_PORT\",\"enable_kv_cache_events\":true}" \
    $EXTRA_WORKER_ARGS \
    > "/logs/worker$i.log" 2>&1 &
  if [ "$i" -lt $((NUM_WORKERS - 1)) ]; then sleep "$STAGGER_S"; fi
done

echo "[launch] all processes started; waiting"
wait -n
echo "[launch] a process exited; tearing down"

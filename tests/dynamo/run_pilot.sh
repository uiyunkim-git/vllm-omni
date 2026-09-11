#!/usr/bin/env bash
# Start the Dynamo pilot container on ONE GPU of this host (host networking, HF cache mounted).
#   tests/dynamo/run_pilot.sh            # GPU 0, 3 workers, kv routing
#   GPU=0 NUM_WORKERS=3 ROUTER_MODE=round-robin tests/dynamo/run_pilot.sh
#   PRESET=gptoss GPU=0,1 NUM_WORKERS=2 ONE_GPU_PER_WORKER=1 tests/dynamo/run_pilot.sh   # 1 worker per GPU
#   docker logs -f dynamo-pilot ; docker stop dynamo-pilot
set -euo pipefail
IMAGE="${IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2}"
GPU="${GPU:-0}"
NAME="${NAME:-dynamo-pilot}"
HF_CACHE="${HF_CACHE:-/home/uiyunkim/.cache/huggingface}"
LOG_DIR="${LOG_DIR:-/tmp/dynamo-pilot-logs}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$LOG_DIR"

docker rm -f "$NAME" >/dev/null 2>&1 || true
# The image runs as uid 1000 (dynamo). Our uid (1001) has no passwd entry in the image and
# PyTorch's getpass.getuser() then raises KeyError, so run as root: it can read our HF cache
# and write the log dir. Pilot-only; fine.
docker run -d --name "$NAME" \
  --gpus "\"device=$GPU\"" \
  --network host --ipc host --shm-size 16g \
  --user 0:0 \
  -e HOME=/root -e HF_HOME="$HF_CACHE" \
  -e PRESET -e MODEL -e NUM_WORKERS -e HTTP_PORT -e ROUTER_MODE -e BLOCK_SIZE -e GPU_UTIL -e KV_BYTES -e ONE_GPU_PER_WORKER \
  -e MAX_LEN -e MAX_SEQS -e MAX_BATCHED -e STAGGER_S -e EXTRA_WORKER_ARGS -e DYN_LOG \
  -v "$HF_CACHE:$HF_CACHE" \
  -v "$HERE/launch_gemma_x3.sh:/launch.sh:ro" \
  -v "$LOG_DIR:/logs" \
  --entrypoint /bin/bash \
  "$IMAGE" /launch.sh
echo "started $NAME (logs: $LOG_DIR, frontend http://localhost:${HTTP_PORT:-18000})"

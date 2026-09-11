#!/usr/bin/env bash
# SGLang gpt-oss-120b on ONE GPU of this host, OpenAI-compatible server on $PORT (plain HTTP).
#   tests/sglang/run_gptoss.sh                 # GPU 1, port 62001
#   GPU=1 PORT=62001 EXTRA_ARGS="--attention-backend triton --moe-runner-backend triton" tests/sglang/run_gptoss.sh
#   docker logs -f sglang-gptoss ; docker stop sglang-gptoss
set -euo pipefail
IMAGE="${IMAGE:-lmsysorg/sglang:latest}"
GPU="${GPU:-1}"
PORT="${PORT:-62001}"
NAME="${NAME:-sglang-gptoss}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
CTX="${CTX:-32768}"                 # matches our vLLM gpt-oss max_model_len
MEM_FRAC="${MEM_FRAC:-0.90}"
HF_CACHE="${HF_CACHE:-/home/uiyunkim/.cache/huggingface}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --gpus "\"device=$GPU\"" \
  --network host --ipc host --shm-size 16g \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --tp 1 \
    --context-length "$CTX" \
    --mem-fraction-static "$MEM_FRAC" \
    --reasoning-parser gpt-oss \
    --tool-call-parser gpt-oss \
    --enable-metrics \
    $EXTRA_ARGS
echo "started $NAME on GPU $GPU → http://localhost:$PORT  (docker logs -f $NAME)"

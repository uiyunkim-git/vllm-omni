# 서빙 엔진 비교: vLLM vs Dynamo vs SGLang (gpt-oss-120b, RTX PRO 6000 ×1)

2026-09-11, neuron(143.248.74.105). 모든 측정은 GPU 1장, 같은 클라이언트(`tests/bench_concurrency.py`,
raw HTTP/1.1 keep-alive — httpx 아님*), 같은 호스트에서 각 엔진의 엔드포인트로 직접.

\* 이전 "라우터 경유 37 req/s" 수치는 httpx 커넥션 풀이 2048 keep-alive 연결에서 O(n²)로 느려져
클라이언트가 병목이 된 결과였다. 같은 라우터를 raw 클라이언트로 재측정하면 125 req/s(엔진 한계).

## 워크로드: believe 실제 잡 형태

believe 파이프라인(30일 1,470만 요청, 잡당 1,000요청·동시성 1024 버스트)의 요청을 그대로 재현
(`--workload believe`): Harmony `system`("Reasoning: high" 텍스트) + 고정 `developer` 리뷰어 지시(~270tok,
전 요청 공유) + `user`(가설+제목+초록 ~500tok, 요청마다 다름), `max_tokens`/`temperature` 미지정,
non-stream, **동시성 1024 × 1000요청**. 프롬프트 ≈ 830tok.

## 결과 (req/s = 잡 처리 속도, tok/s = completion 토큰(reasoning 포함) 생산 속도)

| 엔진 (GPU) | reasoning | req/s | tok/s | p50 | p99 | 출력 평균 tok | wall |
|---|---|---|---|---|---|---|---|
| **vLLM 0.26.0** `vllm serve` (GPU4, 프로덕션 설정) | medium(as-is) | **21.6** | 4,614 | 38.6s | 45.8s | 213 | 46s |
| **Dynamo 1.4.2** frontend(Rust)+`dynamo.vllm` 0.26.0 (GPU0) | medium(as-is, system 병합†) | **22.6** | 4,870 | 35.1s | 43.5s | 216 | 44s |
| vLLM 0.26.0 (GPU4) | high(API) | **11.6** | 5,327 | 60.7s | 83.7s | 461 | 87s |
| Dynamo 1.4.2 (GPU0) | high(API, system 병합†) | **12.1** | 5,395 | 54.8s | 80.6s | 445 | 83s |
| Dynamo 1.4.2 (GPU0) | high(API, **as-is 3메시지**) | 1.8 | 2,966 | 172s | 365s | 1,632‡ | 550s |
| **SGLang 0.5.19** (GPU1, mem 0.90 → KV 37k tok, FlashInfer CUTLASS MXFP4 MoE) | medium(as-is§) | **2.1** | 1,033 | 250s | 475s | 496 | 480s |
| SGLang 0.5.19 (GPU1, mem 0.95 → KV 113k tok) | medium(as-is§) | **3.8** | 1,930 | 137s | 252s | 503 | 260s |
| SGLang 0.5.19 (GPU1, `--moe-runner-backend triton`, mem 0.93) | — | 기동 실패 (CUDA OOM, 가중치 87GB) | | | | | |

† Dynamo 1.4.2의 Harmony 렌더러는 `system`과 `developer`가 함께 오면 **하나를 버린다**
(as-is 569 prompt tok = developer 제거 시와 동일; 하나의 `system`으로 합치면 829 ≈ vLLM 835).
‡ 그 결과 리뷰어 지시가 사라져 reasoning이 폭주(1,632tok) — 엔진 성능이 아니라 렌더러 결함.
§ SGLang도 같은 562 prompt tok → 마찬가지로 system/developer 중 하나를 버린다. 3메시지 Harmony 요청을
  올바르게(835tok) 렌더링하는 것은 셋 중 vLLM만이다.

### 해석
- **Dynamo ≈ vLLM (+4~5% req/s, p50 −10%)**. gpt-oss-120b는 GPU 1장에서 이미 연산 포화라 서빙층(uvicorn vs Rust
  프론트엔드)이 처리량을 바꾸지 못한다. 엔진(vLLM 0.26.0, MXFP4=MARLIN, attn=TRITON_ATTN)이 같으니 당연한 결과.
  Dynamo가 유효한 곳은 유입이 병목인 작은 모델(gemma-4 단일 인스턴스 GPU 58~72%)과 KV-aware 라우팅(멀티턴/공유 prefix).
- **SGLang은 이 GPU에서 gpt-oss-120b에 부적합**: MXFP4 가중치를 82~87GB로 펼쳐(vLLM ~61GB) KV 캐시가 3.7만 토큰만
  남는다(vLLM은 수십만). 1024 동시성이 사실상 직렬화되어 1/10 처리량. sm_120용 MXFP4 MoE 경로 미성숙.
- **reasoning_effort는 API 필드로만 먹는다.** system 텍스트의 "Reasoning: high"는 vLLM/Dynamo/SGLang 어느 렌더러도
  인식하지 않는다(동일 요청: as-is 194tok vs `reasoning_effort:"high"` 792tok). believe 프로덕션은 사실상 medium으로
  돌고 있으며, high로 바꾸면 처리량 절반(21.6 → 11.6 req/s/GPU). believe 측 재현: low로 두면 판정이 뒤집힘(NEUTRAL 오판)
  → **엔진 비교는 반드시 같은 reasoning_effort에서**.

## 재현
```
ulimit -n 16384
tests/bench_concurrency.py --url <endpoint> --model openai/gpt-oss-120b --workload believe \
    --concurrency 1024 --total 1000 --max-tokens 0 [--reasoning-effort high] [--merge-system] [--insecure]
tests/dynamo/run_pilot.sh   (PRESET=gptoss GPU=0)      # Dynamo frontend + dynamo.vllm, :18000
tests/sglang/run_gptoss.sh  (GPU=1 PORT=62001)         # SGLang, :62001
```

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

### 부하를 더 올리면 (believe medium, 병렬 측정 — vLLM GPU4 / Dynamo GPU0 동시)

| 동시성 × 요청 | 엔진 | req/s | tok/s | p50 | p99 | 오류 |
|---|---|---|---|---|---|---|
| 4096 × 4096 | vLLM | 25.0 | 5,476 | 109s | 163s | 0 |
| 4096 × 4096 | Dynamo | 25.4 | 5,520 | 99s | 160s | 0 |
| **8192 × 8192** | vLLM | 25.8 | 5,588 | 160s | 264s | **1,335 실패 (16%)**: ConnectionReset 1,254 · Aborted 80 · 500 1 |
| **8192 × 8192** | Dynamo | 26.1 | 5,646 | 178s | 311s | **0** |

처리량은 어느 부하에서도 같다(GPU 100%, 엔진 running ~760~865 + 대기 수천). 차이는 **유입층의 견고성**:
8k 동시 연결에서 vLLM의 단일 프로세스 uvicorn(+TLS)은 연결을 끊기 시작하고(believe 클라이언트는 재시도로
흡수하지만 재시도 트래픽이 얹힘), Dynamo Rust 프론트엔드는 전부 수용했다.

### 라우팅 층 비교: 우리 라우터 vs Dynamo 프론트엔드 (각 gpt-oss 2워커, believe medium, 동시 측정)

우리: `vllm_router_p2c`(:11434, least_connections, HTTPS→워커, `--max-concurrent-requests 2048`, 큐 100) → GPU2·3.
Dynamo: `dynamo.frontend --router-mode kv`(:18000, TCP 요청 플레인) → GPU0·1.

| 동시성 × 요청 | 경로 | req/s | tok/s | p50 | p99 | 결과 |
|---|---|---|---|---|---|---|
| 2048 × 8192 | **우리 라우터** | **54.7** | **11,429** | 35.0s | 65.8s | 8192/8192 OK |
| 2048 × 8192 | Dynamo | 51.6 | 11,098 | 36.1s | 66.7s | 8192/8192 OK |
| 8192 × 8192 | 우리 라우터 | (24.6) | (5,260) | 59.6s | — | **1,849/8192만 성공**: 429 2,600 · 503 2,926 · 500 718 · 408 99 |
| 8192 × 8192 | **Dynamo** | **44.1** | **9,498** | 98.7s | 183.5s | 8192/8192 OK |

- 허용 한도 안(2048)에서는 **우리 라우터가 6% 더 빠르다** — 두 워커를 809/802로 정확히 균등 분배, Dynamo는 701/706.
- 8k 버스트에서는 우리 라우터의 **admission control이 거부 모드로 붕괴**: `--max-concurrent-requests 2048` 초과분은
  큐(100)로, 큐가 차면 429; 큐 대기 60초 초과는 408; 그 실패들이 circuit breaker를 열어(closed→open 2회) 503 폭주.
  엔진은 워커당 51 running으로 거의 비어 있었다(mid-run). Dynamo는 전부 받아 워커 큐에 쌓고(각 ~760 running + 1,300~2,600 대기)
  두 GPU를 100% 유지.
- 즉 차이는 라우팅 알고리즘이 아니라 **과부하 정책**: 우리는 "빨리 거절"(believe 클라이언트가 최대 30회 재시도로 흡수),
  Dynamo는 "전부 수용·대기".

**한도를 16384로 올려 재측정 (8192 × 8192):**

| 경로 | req/s | ok | 실패 | 원인 |
|---|---|---|---|---|
| 우리 라우터 (`--max-concurrent-requests 16384`) | 48.0 (성공분) | 2,840 | **5,352** | 라우터→워커 `error sending request` ×5,352 (워커당 ~2,670) → CB open 2회 |
| Dynamo | 50.8 | 8,192 | 0 | — |

429는 사라졌지만 이번엔 **라우터→워커 연결 단계**에서 실패했다: 워커당 ~4,000개의 새 HTTPS 연결이 한꺼번에 열리면
vLLM의 단일 프로세스 uvicorn(accept backlog 2048 + 순차 TLS 핸드셰이크)이 받아내지 못해 연결이 거절/리셋되고, 그 실패가
circuit breaker를 열어 나머지도 503. 워커 로그에는 아무것도 남지 않는다(앱까지 도달하지 못함). 직접 4096 동시성 테스트가
무오류였던 것과 일치(연결 4k는 버티고 8k는 못 버팀 — 라우터는 2워커에 각 4k를 열었고, 아이들 커넥션은 50초 뒤 닫혀 매 버스트가
콜드 스타트). Dynamo는 프론트엔드↔워커가 **영속 TCP 요청 플레인**(msgpack)이라 연결 폭풍 자체가 없다.

우리 라우터에서 같은 견고성을 얻으려면: 워커당 in-flight 상한(예: 1024)과 라우터 측 큐(엔진은 어차피 ~800 이상 못 돌림),
`pool_idle_timeout` 연장(웜 커넥션 유지), 연결 실패를 CB 실패로 즉시 세지 않기. 이는 Dynamo가 설계로 가진 것을 우리가 구현하는 일.

### 전환 후: Dynamo 프론트엔드 + neuron 5워커 (central 배포, engine=dynamo)

| 동시성 × 요청 | req/s | tok/s | p50 | p99 | 결과 |
|---|---|---|---|---|---|
| 8192 × 8192 (believe medium) | **102.6** | **21,951** | 53.3s | 77.0s | **8192/8192 OK** |

단일 GPU(21.6~23.4 req/s)의 정확히 5배 — 라우팅 손실 없음. 워커별 처리 1,408~1,941(KV 라우터의 캐시 친화 편향).

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

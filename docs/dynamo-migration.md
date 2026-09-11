# Dynamo 전환 (데이터플레인 = NVIDIA Dynamo, 컨트롤플레인 = central)

결정: 2026-09-11. 근거는 `docs/engine-bench.md` — 처리량은 같지만(GPU 포화) 8k 동시 버스트에서 우리
라우터(admission 거절 → CB → 503, 라우터→uvicorn HTTPS 연결 폭풍)는 붕괴하고 Dynamo 프론트엔드(영속 TCP
요청 플레인, 전부 수용·대기)는 무오류. Dynamo가 설계로 가진 것을 우리가 다시 만드는 대신 채택.

## 아키텍처

```
클라이언트 ──HTTP──▶ dynamo.frontend (Rust, OpenAI 호환, KV-aware 라우터)  :11434 (전환 완료 후; 지금은 :11435)
                          │ etcd watch                    │ TCP 요청 플레인(msgpack) + ZMQ KV 이벤트 구독
                          ▼                               ▼
                    etcd :2379 (neuron)          dynamo.vllm 워커 (호스트별, GPU별 1프로세스; uvicorn 없음)
                          ▲                               ▲
                          └── 등록(lease) ────────────────┘
central (FastAPI) ── /api/deploy(engine=dynamo) ──▶ worker agent ──▶ docker compose (templates/dynamo_node.j2)
```

| 구성요소 | 위치 | 역할 |
|---|---|---|
| `vllm_omni_etcd` | neuron, `docker-compose.central.yml` | 워커 디스커버리(lease). 단일 노드. |
| `vllm_omni_dynamo_frontend` | neuron, 같은 compose | HTTP 유입·토크나이즈·채팅 템플릿·라우팅(`--router-mode kv`, `--migration-limit 3`). `/metrics`(`dynamo_frontend_*`)는 HTTP 포트에서. |
| `dynamo_<deploy>_<worker>_<gpu>` | 각 워커 호스트 | `python -m dynamo.vllm` — vLLM 0.26.0 엔진만 구동. host 네트워크. |
| central | neuron | GPU 배치·배포·정지·health·버전/자동 업데이트·UI (변경 없음). Dynamo 워커는 라우터 등록 대신 etcd. |
| `vllm_router_p2c` | neuron | 전환 기간 동안 vLLM 엔진 배포용으로 유지 → 전 모델 이전 후 제거. |

### 워커 포트 (인스턴스당 3개, 기존 할당 슬롯 `port`에서 파생)
| 용도 | 포트 | 비고 |
|---|---|---|
| system (health/metrics) | `port + 40000` (= 기존 vLLM API 슬롯, 61xxx) | `/health` → `{"status":"ready"}`, `/metrics`에 `vllm:*` 포함 |
| TCP 응답 스트림(요청 플레인) | `port + 42000` (63xxx) | `DYN_TCP_RESPONSE_STREAM_HOST/PORT`로 고정·광고 |
| ZMQ KV 이벤트 | `port + 44000` (65xxx) | `--kv-events-config`; `DYN_EVENT_PLANE_HOST`로 광고 |

광고 주소(`advertise_host`)는 central이 그 워커에 도달하는 IP(`worker.host`)를 그대로 전달. 방화벽은 61xxx·63xxx·65xxx 인바운드 허용 필요.

### 배포 요청 (central `/api/deploy`)
```json
{"name":"GPT-OSS 120B (dynamo)","deployment_type":"replicas","engine":"dynamo",
 "model":"openai/gpt-oss-120b","gpus":["neuron-worker-4"],"max_len":32768,"gpu_util":0.95}
```
파서는 모델명으로 추론(gpt-oss → `gpt_oss`/`harmony`, gemma-4 → `gemma4`, qwen3 → `qwen3`/`hermes`), `reasoning_parser`/
`tool_call_parser`로 명시 가능. 임베딩은 `is_embedding: true` → `--embedding-worker --runner pooling`. `block_size` 기본 64
(프론트 `--kv-cache-block-size`와 반드시 동일).

## 알려진 차이·주의
1. **Harmony `system`+`developer` 동시 전송 시 Dynamo 렌더러가 하나를 버림** (1.4.2). gpt-oss 클라이언트는 둘을 하나의
   `system`으로 합쳐야 함 — believe에 전달·동의됨. vLLM(`vllm serve`)에서는 문제 없었음.
2. `reasoning_effort`는 API 필드로만 동작(system 텍스트 "Reasoning: high"는 어느 엔진도 무시).
3. 워커↔프론트 구간은 평문 TCP(기존 HTTPS 자체서명 대체). 캠퍼스망 내부 + tailnet(WireGuard) 구간이라 수용.
4. 프론트엔드 이미지에 `ip`가 없어 tailnet 라우트를 못 넣음 → kbds처럼 tailnet 전용 워커를 붙이려면 프론트를
   `network_mode: service:tailscale`로 옮겨야 함(컷오버 때 처리).
5. 워커 시스템 포트 `/metrics`의 `vllm:*`는 그대로이지만 라우터 지표(`vllm_router_*`)는 `dynamo_frontend_*`로 바뀜 →
   central 메트릭 스크레이퍼 매핑 필요(Phase 3).
6. 각 워커 호스트에 `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2`(20.6GB) 풀 필요 — Endpoints → Images.

## 단계
- [x] Phase 1 — 인프라: etcd + 프론트엔드(:11435) compose, 워커 `engine=dynamo` 경로, central 전달/health/등록 스킵.
- [x] Phase 1 검증: neuron GPU4에 gpt-oss(dynamo) central 배포 → 프론트 `/v1/models` 노출 → believe 벤치.
- [ ] Phase 2 — neuron의 gpt-oss 전부 dynamo로(GPU0,1,2,3,4), 라우터 경유 vLLM 인스턴스 제거.
- [ ] Phase 3 — central 메트릭/UI를 `dynamo_frontend_*` 기준으로; 워커 목록은 etcd/프론트 기준.
- [ ] Phase 4 — 타 호스트(hubble/heart3/cubis/kbds) 이미지 풀 + 임베딩·기타 모델 이전; 프론트를 tailscale netns로.
- [ ] Phase 5 — 컷오버: 프론트 :11434, `vllm_router_p2c` 제거, believe에 알림.

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
| system (health/metrics) | `port + 10000` (31xxx) | `/health` → `{"status":"ready"}`, `/metrics`에 `vllm:*` 포함. Dynamo가 `DYN_SYSTEM_PORT`를 i16으로 파싱해 32767 이하만 가능 |
| TCP 요청 플레인 리스너 | `port + 12000` (33xxx) | `DYN_TCP_RPC_PORT`. 미지정 시 OS가 임의 포트를 잡아 방화벽 정책이 불가능 |
| TCP 응답 스트림 | `port + 42000` (63xxx) | `DYN_TCP_RESPONSE_STREAM_HOST/PORT`로 고정·광고 |
| ZMQ KV 이벤트 | `port + 44000` (65xxx) | `--kv-events-config`; `DYN_EVENT_PLANE_HOST`로 광고 |

광고 주소(`advertise_host`)는 central이 그 워커에 도달하는 IP(`worker.host`)를 그대로 전달.
**방화벽**: vLLM 컨테이너는 `-p` 퍼블리시라 Docker가 호스트 방화벽(FORWARD)을 스스로 열어 줬지만, host 네트워크의 Dynamo
워커는 INPUT 체인을 타서 기본-DROP(ufw) 호스트(heart3·hubble)에서는 리슨 중이어도 timeout이 났다(2026-09-11: 프론트가
`fetching http://host:31xxx/...` 실패 → 임베딩 워커가 etcd에는 있는데 등록 불가). 해결: 워커 에이전트가 배포 시 그 인스턴스의
포트 4개에 `iptables -I INPUT ACCEPT`를 멱등으로 넣는다(`_ensure_host_ports_open`, 호스트 iptables를 nsenter로 실행 —
Docker가 퍼블리시 포트에 하는 것과 같은 효과). 재부팅 후에는 재배포/재시작 시 다시 적용된다.

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
- [x] Phase 1 검증: neuron GPU4에 gpt-oss(dynamo) central 배포 → 프론트 `/v1/models` 노출 → central health → believe 벤치 (23.4 req/s, 파일럿과 동일).
- [x] Phase 2 — neuron의 gpt-oss 전부 dynamo로(GPU0,1,2,3,4; 배포 f7b3788a/ba486e8f/57781b37), vLLM 인스턴스 제거.
      검증: 8192×8192 believe 버스트 → 102.6 req/s · 21,951 tok/s · 무오류.
- [x] Phase 3 — central 수집기가 Dynamo 프론트(`dynamo_frontend_requests_total`) + 워커 system 포트(`dynamo_component_*`)를
      기존 맵/링버퍼에 합침; 라우터 스크레이프는 선택적. (UI 라벨은 `neuron-worker:31xxx`.)
- [x] Phase 5 — **컷오버 완료 (2026-09-11 20:0x KST)**: 전 호스트 배포 정지·`deployments.json` 초기화 후 프론트가 `:11434`,
      `vllm_router_p2c` 컨테이너 제거(코드는 남김). gpt-oss-120b ×5(neuron, 배포 `ce5877fe`) Dynamo로 재배포. believe merge 배포 확인 후 진행.
      (라우터→Dynamo 하이브리드는 불가: 라우터가 SGLang 전용 필드를 넣어 Dynamo가 400.)
- [ ] Phase 4 — 타 호스트: 이미지 풀 [x hubble, heart3, cubis]. **방화벽**(heart3·hubble: 31000-33999, 63000-65999/tcp from
      143.248.74.105) 과 **cubis 워커 에이전트 재빌드**(5ab7e3d 고착) 뒤 → 임베딩 0.6B/8B(believe qwen_retriever 의존), `-low`(설정 확인 필요).
- [ ] Phase 6 — omniserve 웹(central UI) Dynamo 중심 전면 개편 → `docs/omniserve-ui-redesign.md`(별도 세션).

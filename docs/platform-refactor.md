# 플랫폼 리팩토링 (Dynamo 네이티브) — 2026-09-12

목표: 라우터(P2C) 시대의 전제를 코드·UI·테스트에서 걷어내고, "central = 컨트롤플레인, Dynamo = 데이터플레인,
vLLM = 엔진" 구조를 그대로 드러내는 깔끔한 플랫폼으로 정리한다. 동시에 vLLM 기능이 Dynamo 경로에서
어떻게 노출되는지 검토해 배포 폼/프리셋에 반영한다.

## 1. 아키텍처 (최종)

```
client ──HTTP──▶ dynamo.frontend (:11434, Rust)  ── etcd(:2379) ──  dynamo.vllm 워커 (GPU마다 1개, 호스트 net)
                        ▲ /metrics(dynamo_frontend_*)                    ▲ system :31xxx (/health, /metrics = dynamo_component_* + vllm:*)
central (:8080) ── 배포/정지/health/버전/UI ── worker agent (:8085, 호스트마다 1개) ── docker compose (dynamo_node.j2)
```

엔진 종류: `dynamo`(기본, 유일한 게이트웨이 경로). `vllm`(레거시 직결 vLLM, 게이트웨이 없음 — 디버그용)만 남기고
`ollama`는 제거(게이트웨이가 없어 서비스 경로가 없음).

## 2. vLLM 기능 ↔ Dynamo 경로 검토

| vLLM 기능 | Dynamo에서 | 플랫폼 노출 |
|---|---|---|
| 엔진 인자(`--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`, `--kv-cache-dtype`, quantization, `--enable-prefix-caching`, `--block-size`…) | `dynamo.vllm`이 전부 통과 | 폼 필드(max_len, gpu_util, block_size 고정) + `extra_args` + 프리셋 |
| OpenAI API 서버(uvicorn), `--api-key`, `--ssl-*` | **없음** — 프론트가 대신 | 제거(TLS 인증서 생성 코드 삭제) |
| 채팅 템플릿 / `--chat-template` | 프론트가 HF 템플릿 렌더 (`--custom-jinja-template` 워커 플래그) | 폼: 커스텀 템플릿 경로(옵션) |
| reasoning/tool 파서 (`--reasoning-parser`, `--tool-call-parser`) | 프론트 측 `--dyn-reasoning-parser/--dyn-tool-call-parser`(워커가 광고) | 모델명 추론 + 오버라이드 (구현됨) |
| `reasoning_effort` | API 필드로 지원 | 문서 |
| 구조화 출력(`response_format`, guided) | 워커 `--reasoning-parser <vllm>` + `--dyn-reasoning-parser` 병행 필요 | 프리셋 메모 |
| 임베딩(pooling) | `--embedding-worker --runner pooling`; 프론트 `kv` 모드 불가 → `least-loaded` | 폼 토글 (구현됨) |
| 멀티모달 | Dynamo 경로 지원(문서) — 미검증 | 후속 |
| LoRA | Dynamo `DYN_LORA_ENABLED` 경로 — 미검증 | 후속 |
| Speculative decoding | 엔진 인자 통과 | extra_args |
| prefix caching / KV 이벤트 | 워커가 ZMQ로 발행(kv 라우팅 시 사용) | block_size 64 고정 |
| 요청 취소 / 마이그레이션 | 프론트 `--migration-limit` | compose env |
| sm_120 MXFP4 = MARLIN 폴백, TRITON_ATTN | 엔진 문제(Dynamo 무관) | 후속 성능 과제로 기록 |

## 3. 작업 트랙

### A. central 백엔드 (main.py / manager.py) — 완료 (3076abb, 58ccc72)
- [x] P2C 라우터 코드 제거: `_p2c_register/_p2c_deregister/sync_p2c_workers`, `P2C_ROUTER_URL`, `ROUTER_METRICS_URL/ROUTER_WORKERS_URL`,
      `vllm_router_*` 파싱, CB/retry 지표 전부.
- [x] 엔진 기본값 `dynamo`; `ollama` 제거; `vllm`은 legacy 플래그.
- [x] Dynamo 네이티브 API: `GET /api/frontend`(health, models, 요약 지표), `GET /api/instances`(etcd 인스턴스 ⨝ 배포 노드 ⨝ system 포트 지표),
      `GET /api/prometheus_stats`·`/api/rps_history`를 프론트/워커 지표 기준으로 재작성.
- [x] etcd 조회: gRPC-gateway(`POST /v3/kv/range`)로 `v1/instances/` 프리픽스 읽기.
- [x] 배포 스키마 정리(Dynamo 필드 1급), 배포 레코드에 설정 저장(max_len/gpu_util/extra_args/parsers) — 재배포·이전 가능하게.
- [x] health: system `/health` + 프론트 `/v1/models` 포함 여부.

### B. worker 에이전트 — 완료 (3076abb)
- [x] 기본 엔진 dynamo, ollama 템플릿/분기 제거, dynamo 경로에서 TLS 인증서 생성 제거.
- [ ] (남음) GPU 상태를 19GB vLLM 이미지 대신 호스트 `nvidia-smi`(nsenter)로.
- [x] 시작 시 기존 dynamo 배포의 방화벽 규칙 재적용(`_ensure_host_ports_open`).
- [x] `list_vllm_images` → 엔진 이미지 전체(dynamo 포함).

### C. UI (omniserve 웹) — 완료 (36b1490 머지) — `docs/omniserve-ui-redesign.md`
- [x] 대시보드/모델/배포/호스트/로그/API 페이지를 Dynamo 정보모델로 재구성, 라우터 페이지(metrics.html 'Router Metrics', gateway) 대체.

### D. 테스트
- [ ] `tests/integration/test_router.py`(Rust 라우터) 제거 → `tests/integration/test_frontend.py`(라이브 프론트 또는 mock).
- [ ] 단위: 네임스페이스 슬러그, 파서 추론, node_url, Dynamo 지표 파싱, 워커 템플릿 렌더.

### E. 정리 — 완료 (a6031c4)
- [x] `central/router`(Rust) 삭제(별도 커밋, 되돌리기 쉽게), Dockerfile.router·compose 잔재·문서 정리, README 갱신.

## 전환 중 발견해 고친 것 (2026-09-11~14)

| 증상 | 원인 | 조치 |
|---|---|---|
| 워커가 `DYN_SYSTEM_PORT` 파싱 실패로 크래시 루프 | Dynamo가 i16으로 읽음(≤32767) | system 포트 = base+10000 |
| 두 번째 모델 등록 거부 | 네임스페이스당 모델 1개 | 모델별 `dynamo-<slug>` + 프론트 `--namespace-prefix` |
| 임베딩 500 "KV routing should not call generate on PushRouter" | pooling 모델은 kv 라우팅 불가 | 프론트 `--router-mode least-loaded` |
| 원격(heart3) 워커 요청 120초 타임아웃 | 프론트가 브리지 IP(172.18.x)를 광고 → 워커가 응답 스트림 역방향 연결 불가 | 프론트 `network_mode: host` |
| 전 모델 404, 등록 무한 재시도 | `DYN_TCP_RESPONSE_STREAM_HOST`에 IP 리터럴을 넣으면 1.4.2가 "Interface not found" | 해당 변수 제거(호스트 netns 자동 감지) |
| 워커 요청 플레인 포트가 매번 랜덤 | `DYN_TCP_RPC_PORT` 미지정 시 OS 할당 | base+12000으로 고정 |
| host-net 워커 포트가 원격에서 timeout | Docker `-p`가 열어주던 방화벽 규칙이 host 네트워크에는 없음 | 배포 시/에이전트 기동 시 INPUT ACCEPT 삽입 |
| GPU 제거 후에도 인스턴스가 계속 보임 | `stop_replica`가 `gpus`만 지우고 `nodes`를 남김 | 노드도 함께 제거 |

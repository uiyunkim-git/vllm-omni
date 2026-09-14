# omniserve 웹(central UI) — Dynamo 중심 전면 개편 계획

배경: 데이터플레인이 Dynamo(`docs/dynamo-migration.md`)로 바뀌어 UI가 전제하던 것들이 사라졌다 —
P2C 라우터(least_connections, circuit breaker, retry), 워커별 HTTPS 엔드포인트, `vllm_router_*` 지표.
현재 UI는 "적응"만 된 상태(Dynamo 워커가 `neuron-worker:31xxx`로 라우터 워커처럼 보임). 새 모델로 다시 설계한다.

## 정보 모델 (새 UI가 보여줘야 하는 것)

| 개념 | 출처 | 비고 |
|---|---|---|
| 프론트엔드 | `vllm_omni_dynamo_frontend` `/health`, `/metrics`(`dynamo_frontend_*`), `/v1/models` | 단일 유입점. 요청/TTFT/ITL/큐/마이그레이션/거부 |
| 모델(서빙 이름) | 프론트 `/v1/models` + `dynamo_frontend_model_*` (ready, kv blocks, max_num_seqs, context) | 네임스페이스 `dynamo-<slug>` 1:1 |
| 인스턴스(워커) | etcd `v1/instances/<ns>/backend/generate/<id>` + central 배포 노드 + system 포트 `/health`,`/metrics` | GPU·호스트·엔진 지표(`vllm:*`), `dynamo_component_*` |
| 배포 | central `deployments.json` | engine 기본 `dynamo`; 파서·임베딩·block_size·namespace |
| 호스트/GPU | central `/api/endpoints` | 그대로 |
| 디스커버리 | etcd health, lease 수 | 새 카드 |

## 화면

1. **Overview** — 프론트 상태 카드(health, 모델 수, 인스턴스 수, etcd), 요청/초·TTFT p50/p95·ITL·활성/큐 요청(프론트 지표),
   모델별 미니 카드. 라우터 관련 카드(CB, retry rate, policy decisions) 삭제.
2. **Models** — 서빙 이름별: 인스턴스 목록(호스트/GPU/health/running/waiting/KV 사용률), KV 라우터 히트율(`dynamo_component_router_kv_hit_rate`),
   마이그레이션 수. 인스턴스 추가(=배포 폼 프리필)/개별 정지(per-GPU stop).
3. **Deploy** — 폼 재설계: 모델, 서빙 이름, 호스트/GPU 선택, `engine`(dynamo 기본; vllm은 레거시 숨김), 파서(모델명 추론 + 오버라이드),
   임베딩 토글(`--embedding-worker --runner pooling`), max_len/gpu_util/extra_args, block_size(프론트와 동일 고정 표시).
   저장된 Config는 Dynamo 필드 포함해 마이그레이션.
4. **Hosts** — 기존 GPU/에이전트 목록 유지 + Dynamo 이미지 존재 여부, 방화벽 포트 체크(31xxx/33xxx/63xxx/65xxx 도달성 프로브), 버전/업데이트.
5. **Logs** — 인스턴스 로그 스트림 유지; 프론트 로그 추가.

## 백엔드 변경 (central)
- `/api/dynamo/frontend` (health/models/metrics 요약), `/api/dynamo/instances` (etcd + 배포 조인), `/api/prometheus_stats` 재정의:
  프론트 지표를 1차 소스로, 워커별 `dynamo_component_*`/`vllm:*`를 2차로. 라우터 지표 코드 제거.
- 배포 요청 스키마 정리(`reasoning_parser`, `tool_call_parser`, `is_embedding`, `block_size`), engine 기본값 dynamo.
- health 체크: system 포트 `/health` + 프론트 `/v1/models` 포함 여부(모델 단위 ready).
- 제거: `P2C_ROUTER_URL`, `_p2c_register/deregister`, `sync_p2c_workers`, `ROUTER_*` env, `vllm_router_*` 파싱.

## 프론트엔드(vanilla JS) 원칙
- 카드가 아닌 리스트/테이블(사용자 선호), 가로 스크롤 금지, 15초 폴링 유지, 창(window) 선택 유지.
- 용어: Router → Frontend, Worker(URL) → Instance(host:gpu), CB → 없음, Retry → Migration.

## API 계약 (백엔드가 제공, UI가 소비) — 2026-09-12 확정

| 엔드포인트 | 응답 |
|---|---|
| `GET /api/frontend` | `{url, healthy, router_mode, models:[{id, namespace, instances, ready}], metrics:{active_requests, queued_requests, requests_total, output_tokens_total}}` |
| `GET /api/instances` | `[{deployment_id, deployment_name, model, served_model_name, engine, worker_id, host, gpu, system_url, healthy, running, waiting, kv_cache_usage_pct, inflight, requests_total, errors_total}]` |
| `GET /api/deployments` | 기존 + `engine`, 배포 설정(`max_len, gpu_util, extra_args, is_embedding, reasoning_parser, tool_call_parser, block_size, image`), `nodes[].url`(system URL) |
| `GET /api/prometheus_stats?window=&served_model_name=` | `{window_seconds, requests_window, rps_window, avg_latency_window_s, ttft_p50_s, ttft_p95_s, active_requests, queued_requests, per_worker:[{url,name,deployment_id,deployment_name,served_model_name,processed,processed_window,running,waiting,errors}], latency_histogram_window:[{le,count}], migrations_window, rejections_window, allowed_windows}` — `cb_*`, `retry*`, `decisions` 제거 |
| `GET /api/rps_history?window=&served_model_name=` | 기존 형태 유지 |
| `POST /api/deploy` | `{name, model, served_model_name?, deployment_type, gpus[], tp?, max_len?, gpu_util?, extra_args?, image?, is_embedding?, reasoning_parser?, tool_call_parser?}` — `engine`는 기본 `dynamo`(레거시 `vllm`만 허용) |
| `GET /api/endpoints`, `/api/gpus`, `/api/version`, `POST /api/workers/{id}/update`, `/api/workers/update_all`, `/api/stop/...`, 로그/이미지/모델 다운로드 | 기존 유지 |

용어: Router → Frontend, Worker(URL) → Instance, CB/Retry → 없음(Migration/Rejection으로 대체).

## 순서
1. 백엔드 API 추가(읽기 전용)와 지표 재정의 → 2. Overview/Models 화면 → 3. Deploy 폼 → 4. Hosts 확장 → 5. 레거시 제거.
각 단계 끝에 `tests/` e2e(프론트 health, 모델 목록, 배포→ready) 추가.

## 미결
- `openai/gpt-oss-120b-low`의 의미(설정) 확인 후 배포 프리셋으로.
- kbds 같은 tailnet 전용 워커를 붙일 경우 프론트를 tailscale netns로.

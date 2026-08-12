# vllm-omni test suite

Layered tests for the serving platform: the Rust router (`central/router/`),
the Central FastAPI service (`central/main.py`), and the worker agent
(`worker/manager.py`). Each layer can be run and debugged independently.

```
tests/
├── run_tests.sh        # venv bootstrap + pytest wrapper
├── requirements.txt
├── pytest.ini
├── conftest.py         # import machinery for central/main.py & worker/manager.py
├── mock_worker.py      # standalone mock vLLM worker (OpenAI-compatible surface)
├── unit/
│   ├── test_central_metrics.py   # prometheus parsing / ring buffers / windows / rps history
│   └── test_worker_manager.py    # port allocation / deploy locking / stop_replica
├── integration/
│   ├── conftest.py               # docker router + mock-worker fixtures
│   └── test_router.py            # real router binary vs mock workers
└── e2e/
    └── smoke_live.py             # OPT-IN read-only smoke against the live stack
```

## Quick start

```bash
cd tests

./run_tests.sh unit               # fast (<10s), no docker needed, always green
./run_tests.sh integration        # needs docker + the router image (see below)
./run_tests.sh                    # both; integration skips cleanly w/o docker
./run_tests.sh unit -k window -v  # any extra args go straight to pytest
```

`run_tests.sh` creates/reuses `tests/.venv` and pip-installs
`requirements.txt` on every run (fast no-op when already satisfied).

## Layer 1 — unit (no docker, no network)

Prerequisites: python3 + venv. That's it.

* `test_central_metrics.py` imports `central/main.py` directly (see
  `conftest.py` for the sys.modules stubbing of `manager` and the `os`
  injection workaround) and tests `_parse_prometheus_full`, `_push_sample`,
  `_window_delta` and the `/api/rps_history` bucketing as pure functions.
  No FastAPI server is started.
* `test_worker_manager.py` imports `worker/manager.py` with
  docker/subprocess fully mocked and `DATA_DIR` pointed at a tmp dir.
  Real production jinja templates (`worker/templates/`) are rendered.

## Layer 2 — integration (real router binary in docker)

Prerequisites:

* Linux (the router container uses `--network host` so it can reach mock
  workers bound to `127.0.0.1` on the host).
* docker, and a router image containing the built binary at
  `/app/target/release/vllm-router`.

Image selection: env var `ROUTER_IMAGE` (default `vllm-router-check:audit`).
To build one from the production Dockerfile:

```bash
docker build -f central/router/Dockerfile.router -t vllm-omni-router:test central/router
ROUTER_IMAGE=vllm-omni-router:test ./run_tests.sh integration
```

If docker or the image is missing, every integration test **skips** with a
clear reason — the suite never goes red because of a missing environment.

What is covered (each test boots a fresh router container so retry/CB flags
never leak between tests):

| test | validates |
|---|---|
| `test_add_remove_and_list_workers` | `/add_worker?url=`, `/remove_worker`, `/workers`, `/list_workers`, model_id discovery from `/v1/models` |
| `test_least_connections_prefers_fast_worker` | least_connections steers load away from a slow worker |
| `test_streaming_passthrough_delivers_chunks_and_done` | SSE chunks + `data: [DONE]` pass through untouched |
| `test_client_disconnect_before_first_token_cancels_upstream` | client abort mid-stream propagates upstream (cancellation-propagation fix); asserted via the mock's `aborted_streams` counter |
| `test_500_worker_triggers_retry_on_other_worker` | 5xx retry lands on the healthy worker; failures invisible to clients |
| `test_repeated_429_opens_circuit_breaker` | repeated 429 trips the per-worker breaker; open circuit gets zero traffic |
| `test_concurrent_load_balances_between_equal_workers` | 32 concurrent requests split ~evenly; per-worker `peak_concurrency`; `/metrics` sanity |
| `test_embeddings_routing` | `/v1/embeddings` non-stream routing |
| `test_unknown_model_gets_503` | requests for unregistered models are rejected, not misrouted |

Mock workers use plain `http://` URLs on purpose: the router accepts http
worker URLs (its `danger_accept_invalid_certs` only relaxes *https*
verification), and `/add_worker` health-waits on the URL exactly as given —
so no self-signed-cert setup is needed in tests.

### The mock worker (`mock_worker.py`)

Standalone, also usable by hand:

```bash
python mock_worker.py --port 21001 --model my-model --delay-ms 200
curl localhost:21001/v1/models
curl -X POST localhost:21001/v1/chat/completions -d '{"stream": true}'
curl localhost:21001/_stats     # requests / in_flight / peak_concurrency / aborted_streams
```

Failure modes (`--fail-mode` / `MOCK_FAIL_MODE` / `POST /_config`):
`500`, `429`, `hang`, `die_mid_stream`. `--fail-n N` limits failures to the
first N requests (`-1` = forever). `aborted_streams` increments whenever a
client (or the router acting for its client) disconnects before `[DONE]`.

## Layer 3 — e2e live smoke (opt-in, read-only)

Runs against the deployed stack. Never mutates deployments; sends two tiny
chat completions. Guarded by `RUN_LIVE=1` and intentionally NOT collected by
pytest:

```bash
RUN_LIVE=1 API_KEY=sk-... \
  ROUTER_URL=http://127.0.0.1:11434 CENTRAL_URL=http://127.0.0.1:8080 \
  python e2e/smoke_live.py
```

Checks: `/workers` non-empty, one non-stream + one stream completion through
the gateway, and `central /api/prometheus_stats` `rps_window` vs
`requests_window` consistency. Prints a PASS/FAIL table; exit code 0 iff all
pass.

## Known production issues pinned by this suite (do NOT "fix" the tests)

1. **`_parse_prometheus_full` truncates label values containing escaped
   quotes** (`\"`). Pinned by
   `test_escaped_quote_in_label_current_limitation`.
2. **`worker/manager.py` port-conflict retry leaves `dep["ports"]` stale**:
   after an "address already in use" retry the record's `nodes[].port` is
   the new port but `ports` still lists the originally allocated one.
   Noted in `test_address_in_use_retries_on_next_port`.

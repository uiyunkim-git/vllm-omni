# vllm-omni test suite

Tests for the Dynamo-native platform:

```
client ──HTTP──▶ dynamo frontend (:11434) ──etcd(:2379)──▶ dynamo.vllm workers
                                                              ▲ system port = base+10000
central (:8080, FastAPI control plane) ──▶ worker agent (:8085) ──▶ docker compose
```

There is no Rust P2C router any more, so there is no router layer to test. What
the suite pins instead is the contract every other part now depends on:
`central/dynamo.py` (namespaces, ports, metric meaning, etcd), the metrics layer
in `central/main.py`, the deploy plumbing in `central/manager.py`, and the
worker's `dynamo_node.j2` compose template.

```
tests/
├── run_tests.sh        # venv bootstrap + pytest wrapper
├── requirements.txt
├── pytest.ini          # default collection = unit/ ; markers: unit/integration/live
├── conftest.py         # import machinery for central/{dynamo,db,manager,main}.py + worker/manager.py
├── mock_worker.py      # mock WORKER AGENT + the dynamo instance system port it exposes
├── fixtures/
│   ├── frontend_metrics.txt    # trimmed real /metrics of the Dynamo frontend
│   └── instance_metrics.txt    # trimmed real /metrics of one dynamo.vllm worker
├── unit/
│   ├── test_dynamo.py          # namespaces, parsers, node URLs, prometheus, percentiles, etcd
│   ├── test_central_metrics.py # ring buffers, windows, /api/prometheus_stats, /api/rps_history
│   ├── test_central_manager.py # _dynamo_fields, deployment persistence (temp sqlite)
│   ├── test_dynamo_template.py # dynamo_node.j2 -> valid compose YAML
│   └── test_worker_manager.py  # engine selection, derived ports, firewall, port allocation
├── integration/
│   ├── conftest.py             # mock worker agent subprocess fixtures
│   └── test_deploy_flow.py     # central deploy + health check vs. the mock agent (loopback)
├── e2e/
│   └── test_live_smoke.py      # OPT-IN read-only smoke against a running stack
├── bench_concurrency.py        # standalone load generator (not collected by pytest)
└── dynamo/, sglang/            # manual pilot/launch scripts (not collected by pytest)
```

## Quick start

```bash
cd tests

./run_tests.sh                     # unit layer: offline, no docker, no network, ~3s
./run_tests.sh unit -k namespace   # extra args go straight to pytest
./run_tests.sh integration         # central vs. mock worker agent (loopback sockets only)
./run_tests.sh all                 # unit + integration
RUN_LIVE=1 ./run_tests.sh e2e      # opt-in smoke against the RUNNING stack
```

`run_tests.sh` creates/reuses `tests/.venv` and pip-installs `requirements.txt`
on every run (fast no-op when already satisfied). If your `python3` has no
`ensurepip`, the script falls back to `~/anaconda3/bin/python`; `PYTHON=...`
overrides the choice.

## Layer 1 — unit (no docker, no network, no GPU)

Everything is imported directly from `central/` and `worker/`; no server is
started and no socket is opened. `conftest.py` does the awkward part: those
modules `os.makedirs("/app/data")`, open sqlite there, instantiate a manager at
import time and mount CWD-relative static dirs, so the fixtures patch that away
and point `central/db.py:DB_PATH` at a per-test temp file.

| file | covers |
|---|---|
| `test_dynamo.py` | `namespace_for` (slug, prefix, 63-char cap, replicas sharing a namespace), `infer_parsers`, `node_url`/`node_api_port` for both engines, `parse_prometheus`, `frontend_summary`/`instance_summary` against the fixtures, `percentile_from_buckets`, `_range_end` prefix-scan semantics, `generate_instances_by_namespace`, and the httpx call sites via `MockTransport` |
| `test_central_metrics.py` | `_push_sample` counter-reset clearing + retention, `_window_delta` wall-clock anchoring (a stale series reads as "no activity"), `/api/prometheus_stats` + `/api/instances` + `/api/rps_history` shapes driven by an injected fake scrape, and that no router-era field (`cb_*`, `retries`, `decisions`, …) survives anywhere in the payloads |
| `test_central_manager.py` | `_dynamo_fields` (advertise host, etcd endpoint, namespace, inferred vs explicit parsers, block size, `is_embedding`), `DEPLOY_CONFIG_KEYS` round-trip through `save_deployments`/`load_deployments` on a temp sqlite, engine defaulting to `dynamo`, worker registry |
| `test_dynamo_template.py` | renders `worker/templates/dynamo_node.j2` with jinja2 and parses the result as YAML: host networking, `DYN_SYSTEM_PORT`/`DYN_TCP_RPC_PORT`/`DYN_TCP_RESPONSE_STREAM_PORT`, the KV-events JSON and its port, the embedding variant (`--embedding-worker --runner pooling`, no KV events/block size), parser flags, and that **no `--router-mode` flag is emitted** (`dynamo.vllm` rejects it) |
| `test_worker_manager.py` | engine defaults to `dynamo`, unknown engines rejected, TLS certs only for `engine=vllm`, the four derived ports reaching both the compose file and the host firewall rules, `reapply_host_ports` touching dynamo deployments only, port allocation + the address-in-use retry, deploy locking, `stop_replica` id reconstruction |

### Port scheme (asserted on both sides)

A deployment node stores one base `port` (21001+); everything else is derived,
and `central/dynamo.py` and `worker/manager.py` must agree:

| listener | offset | example |
|---|---|---|
| dynamo system (`/health`, `/metrics`) | `+10000` | 31001 (must fit in an i16!) |
| dynamo TCP request plane | `+12000` | 33001 |
| dynamo TCP response stream | `+42000` | 63001 |
| dynamo ZMQ KV events | `+44000` | 65001 |
| legacy vLLM HTTPS API | `+40000` | 61001 |

## Layer 2 — integration (loopback only, still no docker)

`mock_worker.py` implements the two things central talks to:

* the **worker agent API** (`POST /api/internal/deploy` with the current
  request shape including the dynamo fields, `stop`, `stop_replica`, `images`,
  `version`), and
* the **dynamo instance system port** it would have started: a small HTTP
  server on `base + 10000` answering `/health` with `{"status":"ready"}` and
  serving a plausible `/metrics` page (`dynamo_component_*` + `vllm:*`).

`test_deploy_flow.py` drives the real `central/manager.py` against it: the
worker gets the right namespace/parsers/block size, the deployment is persisted
with its nodes, `run_health_checks()` probes the system port and flips the
deployment to `running`, metrics are scrapable, stop reaches the worker, and a
failing deploy is rolled back.

Run the mock by hand:

```bash
python mock_worker.py --port 8085 --worker-id mock-worker
curl -s localhost:8085/_state | jq
curl -s -XPOST localhost:8085/_config -d '{"ready": false, "running": 7}'
```

## Layer 3 — e2e live smoke (opt-in, read-only)

Guarded by `RUN_LIVE=1`; skipped otherwise, so a normal run never touches the
cluster. It never mutates deployments — two throwaway inference requests plus
dashboard reads.

```bash
RUN_LIVE=1 FRONTEND_URL=http://127.0.0.1:11434 CENTRAL_URL=http://127.0.0.1:8080 \
  [API_KEY=sk-...] [SMOKE_MODEL=...] [SMOKE_EMBED_MODEL=...] ./run_tests.sh e2e
```

Checks: frontend `/health` + `/v1/models`, one chat completion (streaming and
non-streaming) and one embeddings call, and field presence on central's
`/api/frontend`, `/api/instances`, `/api/prometheus_stats` and
`/api/rps_history`. Models are auto-detected from `/v1/models`; the embedding
check skips when no embedding model is registered.

## Conventions

* Unit tests must stay offline and docker-free — `./run_tests.sh` is expected
  to be green on a laptop with no cluster access.
* Prefer small readable fixtures (`tests/fixtures/*.txt`) over mocking our own
  functions; only httpx transports and `subprocess` are faked.
* If a test documents a production bug, say so in the test name/docstring and
  list it below instead of quietly asserting the broken behaviour.

## Known production issues found while writing this suite

Deliberately **not** asserted (the tests would have to encode the broken
behaviour); fix them in `central/`, then add the test:

1. **Partial-deploy rollback is dead code.**
   `central/manager.py:_deploy_model_inner` starts with
   `touched_wids: set = set()`, rebinding the parameter that
   `deploy_model()` passes in. The caller's set therefore stays empty and its
   `except` branch never issues the `/api/internal/stop/{deploy_id}` calls, so
   a failure on replica N of M leaves replicas 1..N-1 running on the workers,
   unrecorded and pinning their GPUs. Fix: drop the re-assignment on
   `manager.py:324`. `tests/integration/test_deploy_flow.py::TestDeployFailure`
   covers only the "first replica fails" case, which the bug does not affect.
2. **Namespace truncation can collide.** `dynamo.namespace_for` caps the
   namespace at 63 chars, so two models with a long common prefix map to the
   same Dynamo namespace — and Dynamo rejects the second model registering on
   an occupied `<ns>/backend/generate`. Only a hash suffix would make this
   safe; the 63-char cap itself is pinned by `test_capped_at_63_chars`.

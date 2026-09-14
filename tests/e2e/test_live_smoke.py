"""OPT-IN live smoke test against a RUNNING vllm-omni stack.

Read-only: it never deploys, stops or reconfigures anything — it sends two
throwaway inference requests and reads central's dashboards.

Skipped entirely unless RUN_LIVE=1 (same opt-in gate the old smoke script
used), so a normal `./run_tests.sh` never touches the cluster:

    RUN_LIVE=1 ./run_tests.sh e2e

    # or, pointing at another stack / through a tunnel
    RUN_LIVE=1 FRONTEND_URL=http://127.0.0.1:11434 \
      CENTRAL_URL=http://127.0.0.1:8080 [API_KEY=sk-...] \
      [SMOKE_MODEL=openai/gpt-oss-120b] [SMOKE_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B] \
      ./run_tests.sh e2e

Checks:
  frontend   /health, /v1/models, one chat completion, one embeddings call
  central    /api/frontend, /api/instances, /api/prometheus_stats field presence
"""

import os

import httpx
import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE") != "1",
        reason="live smoke test: set RUN_LIVE=1 (and FRONTEND_URL/CENTRAL_URL) to run",
    ),
]

FRONTEND_URL = os.environ.get(
    "FRONTEND_URL", os.environ.get("DYNAMO_FRONTEND_URL", "http://127.0.0.1:11434")
).rstrip("/")
CENTRAL_URL = os.environ.get("CENTRAL_URL", "http://127.0.0.1:8080").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


@pytest.fixture(scope="module")
def client():
    with httpx.Client(timeout=60.0, verify=False) as c:
        yield c


@pytest.fixture(scope="module")
def models(client) -> list:
    r = client.get(f"{FRONTEND_URL}/v1/models", headers=HEADERS)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        pytest.skip(f"no models registered on the frontend at {FRONTEND_URL}")
    return [m["id"] for m in data]


@pytest.fixture(scope="module")
def chat_model(models) -> str:
    explicit = os.environ.get("SMOKE_MODEL")
    if explicit:
        return explicit
    generative = [m for m in models if "embed" not in m.lower()]
    if not generative:
        pytest.skip("no generative model registered on the frontend")
    return generative[0]


@pytest.fixture(scope="module")
def embedding_model(models):
    explicit = os.environ.get("SMOKE_EMBED_MODEL")
    if explicit:
        return explicit
    candidates = [m for m in models if "embed" in m.lower()]
    if not candidates:
        pytest.skip("no embedding model registered on the frontend")
    return candidates[0]


# ---------------------------------------------------------------------------
# Dynamo frontend (the single ingress)
# ---------------------------------------------------------------------------

class TestFrontend:
    def test_health(self, client):
        r = client.get(f"{FRONTEND_URL}/health", timeout=10.0)
        assert r.status_code == 200, r.text

    def test_models_listed(self, models):
        assert models, "frontend must advertise at least one model"

    def test_chat_completion(self, client, chat_model):
        r = client.post(
            f"{FRONTEND_URL}/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": chat_model,
                "messages": [{"role": "user", "content": "Say OK and nothing else."}],
                "max_tokens": 16,
                "stream": False,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["choices"], body
        assert "content" in body["choices"][0]["message"]

    def test_streaming_chat_completion(self, client, chat_model):
        chunks, done = 0, False
        with client.stream(
            "POST",
            f"{FRONTEND_URL}/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": chat_model,
                "messages": [{"role": "user", "content": "Count to three."}],
                "max_tokens": 24,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                if line[len("data:"):].strip() == "[DONE]":
                    done = True
                    break
                chunks += 1
        assert chunks >= 1 and done

    def test_embeddings(self, client, embedding_model):
        r = client.post(
            f"{FRONTEND_URL}/v1/embeddings",
            headers=HEADERS,
            json={"model": embedding_model, "input": "hello world"},
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data and len(data[0]["embedding"]) > 0


# ---------------------------------------------------------------------------
# central control plane
# ---------------------------------------------------------------------------

class TestCentral:
    def test_api_frontend(self, client):
        r = client.get(f"{CENTRAL_URL}/api/frontend")
        assert r.status_code == 200, r.text
        body = r.json()
        for key in ("url", "healthy", "kv_block_size", "models", "metrics"):
            assert key in body, f"missing {key} in /api/frontend"
        for key in ("active_requests", "queued_requests", "requests_total",
                    "output_tokens_total"):
            assert key in body["metrics"]
        for m in body["models"]:
            for key in ("id", "namespace", "instances", "ready"):
                assert key in m
        assert body["healthy"] is True

    def test_api_instances(self, client):
        r = client.get(f"{CENTRAL_URL}/api/instances")
        assert r.status_code == 200, r.text
        rows = r.json()
        assert isinstance(rows, list)
        for row in rows:
            for key in ("url", "name", "deployment_id", "served_model_name", "engine",
                        "host", "healthy", "running", "waiting", "kv_cache_usage_pct",
                        "processed", "errors"):
                assert key in row, f"missing {key} in /api/instances row"

    def test_api_prometheus_stats(self, client):
        r = client.get(f"{CENTRAL_URL}/api/prometheus_stats", params={"window": 300})
        assert r.status_code == 200, r.text
        body = r.json()
        for key in ("timestamp", "window_seconds", "allowed_windows", "active_workers",
                    "active_requests", "queued_requests", "total_in_flight",
                    "rps_window", "requests_window", "output_tokens_window",
                    "avg_latency_window_s", "avg_ttft_window_s", "ttft_p50_s",
                    "ttft_p95_s", "migrations_window", "rejections_window",
                    "total_requests", "per_worker", "latency_histogram_window",
                    "ttft_histogram_window", "deployments"):
            assert key in body, f"missing {key} in /api/prometheus_stats"
        assert body["window_seconds"] in body["allowed_windows"]
        # the router era must stay gone
        for key in body:
            assert not any(m in key.lower() for m in ("cb_", "retries", "decision"))

    def test_api_rps_history(self, client):
        r = client.get(f"{CENTRAL_URL}/api/rps_history", params={"window": 3600})
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body) == {"window_seconds", "samples", "allowed"}
        for s in body["samples"]:
            assert set(s) == {"ts", "rps"}

    def test_deployed_models_are_served_by_the_frontend(self, client, models):
        """Every running deployment central knows about should be discoverable
        through the frontend (that is the whole point of etcd registration)."""
        r = client.get(f"{CENTRAL_URL}/api/deployments")
        assert r.status_code == 200, r.text
        running = [
            d for d in r.json()
            if d.get("status") == "running" and d.get("engine", "dynamo") == "dynamo"
        ]
        missing = [
            d["served_model_name"] for d in running
            if d.get("served_model_name") not in models
        ]
        assert not missing, f"running deployments not visible on the frontend: {missing}"

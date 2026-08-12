"""Integration tests against the REAL Rust router binary (in docker) with
mock vLLM workers running on the host.

Skips as a whole when docker / the router image is unavailable
(see integration/conftest.py). Each test gets a fresh router container so
circuit-breaker / retry settings never leak between tests.

Mock workers speak plain http:// — the router accepts http worker URLs
(danger_accept_invalid_certs only relaxes https verification) and health-waits
on the URL exactly as given at /add_worker time.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest


def _wait_stat(worker, key, minimum, timeout_s=10.0):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = worker.stats()
        if last[key] >= minimum:
            return last
        time.sleep(0.2)
    raise AssertionError(
        f"worker {worker.url}: {key}={last and last[key]} never reached {minimum}; "
        f"stats={last}"
    )


# ---------------------------------------------------------------------------
# (a) worker registration
# ---------------------------------------------------------------------------

def test_add_remove_and_list_workers(router_factory, mock_worker_factory):
    router = router_factory()
    w1 = mock_worker_factory(model="reg-model")
    w2 = mock_worker_factory(model="reg-model")

    router.add_worker(w1.url)
    router.add_worker(w2.url)

    listing = router.workers()
    urls = {w["url"] for w in listing["workers"]}
    assert urls == {w1.url, w2.url}
    for w in listing["workers"]:
        # model_id is fetched from the mock's /v1/models at registration time
        assert w["model_id"] == "reg-model"
        assert w["is_healthy"] is True
    assert listing["total"] == 2

    # legacy listing endpoint agrees
    assert set(router.list_worker_urls()) == {w1.url, w2.url}

    router.remove_worker(w2.url)
    assert {w["url"] for w in router.workers()["workers"]} == {w1.url}


# ---------------------------------------------------------------------------
# (b) least_connections steers load away from a slow worker
# ---------------------------------------------------------------------------

def test_least_connections_prefers_fast_worker(router_factory, mock_worker_factory):
    """The policy treats loads within +/-2 of the minimum as a tie (random
    pick), so we drive 40 requests through a bounded pool of 8 clients: the
    slow worker's in-flight count quickly exceeds the tie threshold and the
    router must steer the remaining traffic to the fast worker."""
    router = router_factory(policy="least_connections")
    fast = mock_worker_factory(model="lc-model", delay_ms=25)
    slow = mock_worker_factory(model="lc-model", delay_ms=1000)
    router.add_worker(fast.url)
    router.add_worker(slow.url)

    def one(_):
        return router.chat("lc-model", timeout=30.0).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(one, range(40)))
    assert codes.count(200) == 40

    fast_n = fast.stats()["requests"]
    slow_n = slow.stats()["requests"]
    assert fast_n + slow_n == 40
    assert fast_n > slow_n, (
        f"least_connections should favor the fast worker "
        f"(fast={fast_n}, slow={slow_n})"
    )
    assert fast_n >= 24, f"fast worker should absorb most load (fast={fast_n}, slow={slow_n})"


# ---------------------------------------------------------------------------
# (c) SSE streaming passthrough
# ---------------------------------------------------------------------------

def test_streaming_passthrough_delivers_chunks_and_done(
    router_factory, mock_worker_factory
):
    router = router_factory()
    w = mock_worker_factory(model="sse-model", stream_chunks=4, interchunk_ms=20)
    router.add_worker(w.url)

    payload = {
        "model": "sse-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    events = []
    with httpx.Client(timeout=30.0) as client:
        with client.stream(
            "POST", f"{router.url}/v1/chat/completions", json=payload
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    events.append(line[len("data:"):].strip())

    assert events[-1] == "[DONE]"
    content_chunks = [json.loads(e) for e in events[:-1]]
    assert len(content_chunks) == 4
    assert all(
        c["choices"][0]["delta"]["content"].startswith("tok")
        for c in content_chunks
    )
    assert w.stats()["completed_streams"] == 1


# ---------------------------------------------------------------------------
# (d) client-disconnect mid-stream propagates to the worker
# ---------------------------------------------------------------------------

def test_client_disconnect_before_first_token_cancels_upstream(
    router_factory, mock_worker_factory
):
    """Validates the cancellation-propagation fix: the client aborts while the
    worker is still sleeping before its first SSE chunk; the router must tear
    down the upstream connection, which the mock worker records as an aborted
    stream (its generator's finally runs without reaching [DONE])."""
    router = router_factory()
    w = mock_worker_factory(model="cancel-model", delay_ms=5000, stream_chunks=2)
    router.add_worker(w.url)

    payload = {
        "model": "cancel-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    with httpx.Client(timeout=httpx.Timeout(10.0, read=10.0)) as client:
        with client.stream(
            "POST", f"{router.url}/v1/chat/completions", json=payload
        ) as resp:
            # Headers arrive immediately (200 + SSE); abort before any chunk.
            assert resp.status_code == 200
        # leaving the context closes the response -> client disconnect

    stats = _wait_stat(w, "aborted_streams", 1, timeout_s=10.0)
    assert stats["aborted_streams"] >= 1, (
        "router did not propagate the client disconnect to the worker "
        f"(worker stats: {stats})"
    )
    assert stats["completed_streams"] == 0


# ---------------------------------------------------------------------------
# (e) 500 from one worker -> retry lands on the other
# ---------------------------------------------------------------------------

def test_500_worker_triggers_retry_on_other_worker(
    router_factory, mock_worker_factory
):
    router = router_factory(extra_args=[
        "--retry-max-retries", "4",
        "--cb-failure-threshold", "5",
        "--cb-timeout-duration-secs", "120",
    ])
    bad = mock_worker_factory(model="retry-model", fail_mode="500")
    good = mock_worker_factory(model="retry-model")
    router.add_worker(bad.url)
    router.add_worker(good.url)

    codes = [router.chat("retry-model").status_code for _ in range(10)]

    bad_stats = bad.stats()
    good_stats = good.stats()
    assert bad_stats["requests"] >= 1, "failing worker never received an attempt"
    assert good_stats["requests"] >= 8, f"healthy worker served too few: {good_stats}"
    # Retries must make the failures invisible to the client.
    assert codes.count(200) >= 9, f"codes={codes}"
    assert codes[-5:] == [200] * 5, "after CB opens, everything must go to the healthy worker"


# ---------------------------------------------------------------------------
# (f) repeated 429 opens the circuit breaker
# ---------------------------------------------------------------------------

def test_repeated_429_opens_circuit_breaker(router_factory, mock_worker_factory):
    router = router_factory(extra_args=[
        "--retry-max-retries", "4",
        "--cb-failure-threshold", "3",
        "--cb-timeout-duration-secs", "120",   # stays open for the whole test
    ])
    throttled = mock_worker_factory(model="cb-model", fail_mode="429")
    healthy = mock_worker_factory(model="cb-model")
    router.add_worker(throttled.url)
    router.add_worker(healthy.url)

    # Phase 1: generate enough 429s to trip the breaker (threshold 3).
    for _ in range(10):
        assert router.chat("cb-model").status_code == 200

    assert throttled.stats()["requests"] >= 3, "throttled worker was never attempted"

    # Phase 2: with the breaker open the throttled worker must be skipped.
    throttled.reset()
    for _ in range(8):
        assert router.chat("cb-model").status_code == 200
    assert throttled.stats()["requests"] == 0, (
        "circuit should be open: throttled worker must receive no traffic"
    )
    assert healthy.stats()["requests"] >= 18


# ---------------------------------------------------------------------------
# (g) concurrency balance under least_connections + metrics sanity
# ---------------------------------------------------------------------------

def test_concurrent_load_balances_between_equal_workers(
    router_factory, mock_worker_factory
):
    router = router_factory(policy="least_connections")
    w1 = mock_worker_factory(model="conc-model", delay_ms=500)
    w2 = mock_worker_factory(model="conc-model", delay_ms=500)
    router.add_worker(w1.url)
    router.add_worker(w2.url)

    def one(_):
        return router.chat("conc-model", timeout=60.0).status_code

    with ThreadPoolExecutor(max_workers=32) as pool:
        codes = list(pool.map(one, range(32)))
    assert codes.count(200) == 32

    s1, s2 = w1.stats(), w2.stats()
    assert s1["requests"] + s2["requests"] == 32
    # least_connections should split ~16/16; allow slack for scheduling noise.
    assert 10 <= s1["requests"] <= 22, (s1["requests"], s2["requests"])
    assert 10 <= s2["requests"] <= 22, (s1["requests"], s2["requests"])
    # Both workers actually ran requests in parallel.
    assert s1["peak_concurrency"] >= 6, s1
    assert s2["peak_concurrency"] >= 6, s2
    assert s1["in_flight"] == 0 and s2["in_flight"] == 0

    # Router prometheus endpoint reflects the traffic.
    metrics = router.metrics()
    assert "vllm_router_processed_requests_total" in metrics
    assert w1.url in metrics and w2.url in metrics


# ---------------------------------------------------------------------------
# bonus: embeddings passthrough (non-stream + stream)
# ---------------------------------------------------------------------------

def test_embeddings_routing(router_factory, mock_worker_factory):
    router = router_factory()
    w = mock_worker_factory(model="embed-model")
    router.add_worker(w.url)

    r = httpx.post(
        f"{router.url}/v1/embeddings",
        json={"model": "embed-model", "input": ["a", "b"]},
        timeout=30.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 2
    assert w.stats()["requests"] == 1


# ---------------------------------------------------------------------------
# bonus: unknown model -> no workers for it
# ---------------------------------------------------------------------------

def test_unknown_model_gets_503(router_factory, mock_worker_factory):
    router = router_factory()
    w = mock_worker_factory(model="known-model")
    router.add_worker(w.url)

    r = router.chat("some-other-model")
    assert r.status_code == 503, (
        f"expected 503 no-available-workers for unknown model, got {r.status_code}: "
        f"{r.text[:200]}"
    )
    assert w.stats()["requests"] == 0

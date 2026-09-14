"""Unit tests for the metrics layer in central/main.py.

central/main.py is imported directly (see tests/conftest.py) with a fake
manager, so nothing starts a server or opens a socket. The scrape is faked by
writing into the module's own ring buffers (``_metric_history``) and last-scrape
snapshot (``_latest_scrape``) — exactly what ``_collect_metrics`` would have
produced from the fixtures in tests/fixtures/.
"""

import time

import pytest

pytestmark = pytest.mark.unit

MODEL = "openai/gpt-oss-120b"
NODE_URL = "http://10.0.0.7:31001"

DEPLOYMENT = {
    "id": "ce5877fe",
    "name": "GPT-OSS 120B (dynamo)",
    "model": MODEL,
    "served_model_name": MODEL,
    "engine": "dynamo",
    "deployment_type": "replicas",
    "status": "running",
    "gpus": ["neuron-worker-0"],
    "nodes": [
        {
            "name": "dynamo_ce5877fe_neuron-worker_0",
            "host": "10.0.0.7",
            "port": 21001,
            "is_healthy": True,
        }
    ],
}

# Fields the P2C router era exported and the Dynamo platform must not resurrect.
REMOVED_FIELD_MARKERS = ("cb_", "circuit", "retries", "retry", "decision", "p2c")


def _all_keys(obj) -> set:
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            out |= _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _all_keys(v)
    return out


def assert_no_router_era_fields(payload) -> None:
    for key in _all_keys(payload):
        low = key.lower()
        assert not any(m in low for m in REMOVED_FIELD_MARKERS), (
            f"router-era field {key!r} resurfaced in the API payload"
        )


# ---------------------------------------------------------------------------
# _push_sample
# ---------------------------------------------------------------------------

class TestPushSample:
    def test_appends_monotonic_samples(self, central_main):
        t = time.time()
        central_main._push_sample("k", t, 10)
        dq = central_main._push_sample("k", t + 5, 15)
        assert list(dq) == [(t, 10), (t + 5, 15)]

    def test_counter_reset_clears_history(self, central_main):
        """A drop means the frontend or worker restarted: the old history is
        not comparable and must be discarded, not replayed as negative rate."""
        t = time.time()
        central_main._push_sample("k", t, 100)
        central_main._push_sample("k", t + 5, 110)
        dq = central_main._push_sample("k", t + 10, 5)
        assert list(dq) == [(t + 10, 5)]

    def test_equal_value_is_not_a_reset(self, central_main):
        t = time.time()
        central_main._push_sample("k", t, 7)
        dq = central_main._push_sample("k", t + 5, 7)
        assert len(dq) == 2

    def test_retention_drops_samples_older_than_the_buffer(self, central_main):
        t = time.time()
        max_s = central_main._BUFFER_SECONDS_MAX
        central_main._push_sample("k", t, 1)
        central_main._push_sample("k", t + 10, 2)
        dq = central_main._push_sample("k", t + max_s + 5, 3)
        # t fell out of the 6h window (cutoff = t+5); t+10 is still inside it.
        assert [v for _, v in dq] == [2, 3]

    def test_retention_keeps_at_least_one_sample(self, central_main):
        t = time.time()
        max_s = central_main._BUFFER_SECONDS_MAX
        central_main._push_sample("k", t, 1)
        central_main._push_sample("k", t + 10, 2)
        dq = central_main._push_sample("k", t + max_s + 11, 3)
        assert list(dq) == [(t + max_s + 11, 3)]

    def test_series_are_independent(self, central_main):
        t = time.time()
        central_main._push_sample("a", t, 5)
        central_main._push_sample("b", t, 1)
        central_main._push_sample("b", t + 1, 0)  # reset on b only
        assert len(central_main._metric_history["a"]) == 1
        assert list(central_main._metric_history["b"]) == [(t + 1, 0)]


# ---------------------------------------------------------------------------
# _window_delta
# ---------------------------------------------------------------------------

class TestWindowDelta:
    def test_missing_key_returns_none(self, central_main):
        assert central_main._window_delta("nope", 60) is None

    def test_fewer_than_two_samples_returns_none(self, central_main):
        central_main._push_sample("k", time.time(), 5)
        assert central_main._window_delta("k", 60) is None

    def test_now_anchored_window(self, central_main):
        now = time.time()
        central_main._push_sample("k", now - 10, 100)
        central_main._push_sample("k", now - 5, 110)
        central_main._push_sample("k", now - 0.01, 120)
        delta, duration = central_main._window_delta("k", 30)
        assert delta == 20
        assert 9 <= duration <= 11

    def test_window_excludes_older_samples(self, central_main):
        now = time.time()
        central_main._push_sample("k", now - 100, 0)  # outside the 30s window
        central_main._push_sample("k", now - 20, 50)
        central_main._push_sample("k", now - 0.01, 80)
        delta, duration = central_main._window_delta("k", 30)
        assert delta == 30  # 80 - 50, NOT 80 - 0
        assert 19 <= duration <= 21

    def test_stale_series_returns_none(self, central_main):
        """A worker that went away leaves a frozen counter. Anchoring to
        wall-clock `now` (not to the newest sample) makes that read as "no
        activity" instead of replaying its whole historical climb."""
        now = time.time()
        central_main._push_sample("k", now - 500, 100)
        central_main._push_sample("k", now - 400, 200)
        assert central_main._window_delta("k", 60) is None

    def test_series_that_just_stopped_still_counts_inside_the_window(self, central_main):
        now = time.time()
        central_main._push_sample("k", now - 50, 100)
        central_main._push_sample("k", now - 20, 130)
        delta, duration = central_main._window_delta("k", 60)
        assert delta == 30
        assert 29 <= duration <= 31

    def test_never_returns_a_negative_delta(self, central_main):
        now = time.time()
        dq = central_main._metric_history.setdefault("k", central_main.deque())
        dq.append((now - 10, 100))
        dq.append((now - 1, 40))  # bypasses _push_sample's reset handling
        delta, _ = central_main._window_delta("k", 60)
        assert delta == 0.0

    def test_reset_mid_window(self, central_main):
        now = time.time()
        central_main._push_sample("k", now - 20, 100)
        central_main._push_sample("k", now - 10, 5)  # reset -> buffer cleared
        assert central_main._window_delta("k", 60) is None
        central_main._push_sample("k", now, 8)
        delta, duration = central_main._window_delta("k", 60)
        assert delta == 3
        assert 9 <= duration <= 11


# ---------------------------------------------------------------------------
# /api/prometheus_stats
# ---------------------------------------------------------------------------

EXPECTED_STATS_KEYS = {
    "timestamp", "window_seconds", "allowed_windows",
    "active_workers", "active_requests", "queued_requests", "total_in_flight",
    "rps_window", "requests_window", "output_tokens_window",
    "avg_latency_window_s", "avg_ttft_window_s", "ttft_p50_s", "ttft_p95_s",
    "migrations_window", "rejections_window",
    "total_requests",
    "per_worker", "latency_histogram_window", "ttft_histogram_window",
    "deployments",
}


@pytest.fixture
def fake_scrape(central_main, central_dynamo, frontend_metrics_text, instance_metrics_text):
    """Inject one scrape cycle's worth of state, plus ~a minute of history.

    Returns the wall-clock span the seeded samples cover, so tests can reason
    about rates.
    """
    now = time.time()
    fe = central_dynamo.frontend_summary(central_dynamo.parse_prometheus(frontend_metrics_text))
    inst = central_dynamo.instance_summary(central_dynamo.parse_prometheus(instance_metrics_text))

    central_main.manager.deployments = [DEPLOYMENT]
    central_main._latest_scrape.update(
        {"now": now, "frontend": fe, "instances": {NODE_URL: inst}}
    )

    # Both samples sit strictly inside a 60s window (its left edge is re-read
    # from the wall clock, so a sample exactly at now-60 would fall out).
    span = 54.0

    def series(key, start, end):
        central_main._push_sample(key, now - 55, start)
        central_main._push_sample(key, now - 1, end)

    series(f"fe:req:{MODEL}", 1000, 1200)          # 200 requests in ~54s
    series(f"fe:tok:{MODEL}", 200000, 254000)      # 54000 output tokens
    series(f"fe:lat_sum:{MODEL}", 3000.0, 3600.0)  # 600s over 300 requests -> 2.0s avg
    series(f"fe:lat_cnt:{MODEL}", 900, 1200)
    series(f"fe:ttft_sum:{MODEL}", 200.0, 260.0)   # 60s over 300 -> 0.2s avg
    series(f"fe:ttft_cnt:{MODEL}", 900, 1200)
    series(f"fe:mig:{MODEL}", 3, 4)
    series(f"fe:rej:{MODEL}", 0, 1)
    # TTFT histogram deltas in-window: 100 / 200 / 400 (cumulative by `le`).
    series(f"fe:ttftb:0.1:{MODEL}", 100, 200)
    series(f"fe:ttftb:0.25:{MODEL}", 600, 800)
    series(f"fe:ttftb:0.5:{MODEL}", 700, 1100)
    # Latency histogram deltas in-window: 20 / 60 / 100.
    series(f"fe:bucket:0.5:{MODEL}", 80, 100)
    series(f"fe:bucket:1:{MODEL}", 340, 400)
    series(f"fe:bucket:2.5:{MODEL}", 800, 900)
    series(f"inst:req:{NODE_URL}", 700, 830)
    series(f"inst:err:{NODE_URL}", 1, 2)
    return span


class TestPrometheusStatsShape:
    async def test_no_scrape_yet_returns_the_full_shape_with_zeroes(self, central_main):
        out = await central_main.get_prometheus_stats(window=60)
        assert set(out) == EXPECTED_STATS_KEYS
        assert out["active_workers"] == 0
        assert out["requests_window"] == 0
        assert out["rps_window"] == 0
        assert out["per_worker"] == []
        assert out["deployments"] == []
        assert_no_router_era_fields(out)

    async def test_window_is_clamped_to_an_allowed_value(self, central_main):
        out = await central_main.get_prometheus_stats(window=47)
        assert out["window_seconds"] in central_main._ALLOWED_WINDOWS
        assert out["allowed_windows"] == central_main._ALLOWED_WINDOWS

    async def test_full_shape_after_a_scrape(self, central_main, fake_scrape):
        out = await central_main.get_prometheus_stats(window=60)
        assert set(out) == EXPECTED_STATS_KEYS
        assert_no_router_era_fields(out)

    async def test_live_gauges_come_from_the_latest_scrape(self, central_main, fake_scrape):
        out = await central_main.get_prometheus_stats(window=60)
        assert out["active_workers"] == 1
        assert out["active_requests"] == 3       # frontend gauge, gpt-oss only
        assert out["queued_requests"] == 2
        assert out["total_in_flight"] == 3       # vllm:num_requests_running
        # cumulative across every model the frontend knows (1200 + 48)
        assert out["total_requests"] == 1248

    async def test_windowed_aggregates(self, central_main, fake_scrape):
        out = await central_main.get_prometheus_stats(window=60)
        assert out["requests_window"] == 200
        assert out["output_tokens_window"] == 54000
        assert out["migrations_window"] == 1
        assert out["rejections_window"] == 1
        assert out["avg_latency_window_s"] == pytest.approx(2.0)
        assert out["avg_ttft_window_s"] == pytest.approx(0.2)
        # 200 requests over the ~54s actually covered by the samples
        assert out["rps_window"] == pytest.approx(200 / fake_scrape, rel=0.05)

    async def test_ttft_percentiles_from_window_bucket_deltas(self, central_main, fake_scrape):
        out = await central_main.get_prometheus_stats(window=60)
        # deltas: le 0.1 -> 100, le 0.25 -> 200, le 0.5 -> 400 (cumulative)
        assert out["ttft_p50_s"] == pytest.approx(0.25)
        assert out["ttft_p95_s"] == pytest.approx(0.475)

    async def test_histograms_are_reported_as_per_bucket_counts(self, central_main, fake_scrape):
        out = await central_main.get_prometheus_stats(window=60)
        assert out["latency_histogram_window"] == [
            {"le": "0.5", "count": 20},
            {"le": "1", "count": 40},
            {"le": "2.5", "count": 40},
        ]
        assert [b["le"] for b in out["ttft_histogram_window"]] == ["0.1", "0.25", "0.5"]

    async def test_per_worker_row(self, central_main, fake_scrape):
        out = await central_main.get_prometheus_stats(window=60)
        (row,) = out["per_worker"]
        assert row["url"] == NODE_URL
        assert row["name"] == "neuron-worker:31001"
        assert row["worker_id"] == "neuron-worker"
        assert row["gpu"] == "0"
        assert row["engine"] == "dynamo"
        assert row["deployment_id"] == "ce5877fe"
        assert row["served_model_name"] == MODEL
        assert row["healthy"] is True
        assert row["running"] == 3
        assert row["waiting"] == 5
        assert row["inflight"] == 3
        assert row["kv_cache_usage_pct"] == 42.0
        assert row["processed"] == 830
        assert row["errors"] == 2
        assert row["processed_window"] == 130
        assert row["errors_window"] == 1

    async def test_filtering_by_served_model_name(self, central_main, fake_scrape):
        out = await central_main.get_prometheus_stats(window=60, served_model_name=MODEL)
        assert len(out["per_worker"]) == 1
        other = await central_main.get_prometheus_stats(
            window=60, served_model_name="not/deployed"
        )
        assert other["per_worker"] == []
        assert other["requests_window"] == 0

    async def test_stopped_deployments_have_no_rows(self, central_main, fake_scrape):
        central_main.manager.deployments = [{**DEPLOYMENT, "status": "stopped"}]
        out = await central_main.get_prometheus_stats(window=60)
        assert out["per_worker"] == []
        assert out["active_workers"] == 0

    async def test_legacy_vllm_node_is_addressed_over_https(self, central_main):
        central_main.manager.deployments = [{**DEPLOYMENT, "engine": "vllm"}]
        out = await central_main.get_prometheus_stats(window=60)
        (row,) = out["per_worker"]
        assert row["url"] == "https://10.0.0.7:61001"
        assert row["engine"] == "vllm"


# ---------------------------------------------------------------------------
# /api/instances
# ---------------------------------------------------------------------------

class TestInstancesEndpoint:
    async def test_rows_match_the_prometheus_stats_table(self, central_main, fake_scrape):
        rows = await central_main.get_instances()
        stats = await central_main.get_prometheus_stats(window=60)
        assert [r["url"] for r in rows] == [r["url"] for r in stats["per_worker"]]
        assert_no_router_era_fields(rows)

    async def test_empty_without_deployments(self, central_main):
        assert await central_main.get_instances() == []


# ---------------------------------------------------------------------------
# /api/rps_history
# ---------------------------------------------------------------------------

class TestRpsHistory:
    def _seed_constant_rate(self, central_main, model, rps, span_s, step_s=5):
        now = time.time()
        total = 0.0
        for ts in range(-span_s, 1, step_s):
            central_main._push_sample(f"fe:req:{model}", now + ts, total)
            total += rps * step_s

    async def test_empty_history_returns_no_samples(self, central_main):
        out = await central_main.get_rps_history(window=60)
        assert out["samples"] == []
        assert out["window_seconds"] == 60
        assert out["allowed"] == central_main._ALLOWED_HISTORY

    async def test_window_clamped_to_allowed(self, central_main):
        out = await central_main.get_rps_history(window=47)
        assert out["window_seconds"] in central_main._ALLOWED_HISTORY

    async def test_constant_rate_reproduced(self, central_main):
        self._seed_constant_rate(central_main, MODEL, rps=2.0, span_s=120)
        out = await central_main.get_rps_history(window=60)
        assert len(out["samples"]) >= 2
        for s in out["samples"]:
            assert s["rps"] == pytest.approx(2.0, abs=0.5)

    async def test_models_are_summed_when_unfiltered(self, central_main):
        self._seed_constant_rate(central_main, MODEL, rps=2.0, span_s=120)
        self._seed_constant_rate(central_main, "other/model", rps=1.0, span_s=120)
        out = await central_main.get_rps_history(window=60)
        assert out["samples"][-1]["rps"] == pytest.approx(3.0, abs=0.5)

    async def test_filtering_by_model(self, central_main):
        self._seed_constant_rate(central_main, MODEL, rps=2.0, span_s=120)
        self._seed_constant_rate(central_main, "other/model", rps=1.0, span_s=120)
        out = await central_main.get_rps_history(window=60, served_model_name="other/model")
        assert out["samples"][-1]["rps"] == pytest.approx(1.0, abs=0.5)

    async def test_stale_series_decays_to_zero(self, central_main):
        """A model whose counters stopped 10 scrape periods ago contributes 0
        to the newest buckets rather than holding its last rate forever."""
        now = time.time()
        for i, ts in enumerate(range(-600, -540, 5)):
            central_main._push_sample(f"fe:req:{MODEL}", now + ts, 10.0 * i)
        out = await central_main.get_rps_history(window=900)
        assert out["samples"][-1]["rps"] == 0.0

    async def test_timestamps_span_the_requested_window(self, central_main):
        self._seed_constant_rate(central_main, MODEL, rps=1.0, span_s=120)
        before = time.time()
        out = await central_main.get_rps_history(window=60)
        assert out["samples"][0]["ts"] == pytest.approx(before - 60, abs=2)
        assert out["samples"][-1]["ts"] < before + 2

    async def test_shape_has_no_router_era_fields(self, central_main):
        self._seed_constant_rate(central_main, MODEL, rps=1.0, span_s=120)
        out = await central_main.get_rps_history(window=60)
        assert set(out) == {"window_seconds", "samples", "allowed"}
        assert_no_router_era_fields(out)


# ---------------------------------------------------------------------------
# module surface: the router era is gone
# ---------------------------------------------------------------------------

class TestRouterEraRemoved:
    @pytest.mark.parametrize(
        "name",
        [
            "_p2c_register", "_p2c_deregister", "sync_p2c_workers",
            "P2C_ROUTER_URL", "ROUTER_METRICS_URL", "ROUTER_WORKERS_URL",
            "_parse_prometheus_full", "_last_live_instances",
        ],
    )
    def test_symbol_is_gone(self, central_main, name):
        assert not hasattr(central_main, name), f"{name} should not exist any more"

    def test_prometheus_parsing_lives_in_dynamo(self, central_main, central_dynamo):
        assert central_main.dynamo is central_dynamo
        assert hasattr(central_dynamo, "parse_prometheus")

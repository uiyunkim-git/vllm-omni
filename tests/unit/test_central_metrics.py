"""Unit tests for the metrics plumbing in central/main.py.

These import the module directly (no server startup, no router) via the
``central_main`` fixture in tests/conftest.py, and exercise the pure(ish)
functions: _parse_prometheus_full, _push_sample, _window_delta, plus the
rps_history bucketing through the endpoint coroutine.
"""

import asyncio
import time

import pytest


# ---------------------------------------------------------------------------
# _parse_prometheus_full
# ---------------------------------------------------------------------------

class TestParsePrometheusFull:
    def test_plain_counter(self, central_main):
        out = central_main._parse_prometheus_full(
            "# HELP foo help\n# TYPE foo counter\nfoo 42\n"
        )
        assert out == {"foo": [{"labels": {}, "value": 42.0}]}

    def test_labeled_series(self, central_main):
        text = (
            'vllm_router_processed_requests_total{worker="https://h1:61001",instance="abc"} 10\n'
            'vllm_router_processed_requests_total{worker="https://h2:61002",instance="def"} 3\n'
        )
        out = central_main._parse_prometheus_full(text)
        entries = out["vllm_router_processed_requests_total"]
        assert len(entries) == 2
        assert entries[0]["labels"] == {"worker": "https://h1:61001", "instance": "abc"}
        assert entries[0]["value"] == 10.0
        assert entries[1]["labels"]["worker"] == "https://h2:61002"

    def test_histogram_buckets(self, central_main):
        text = (
            'gen_seconds_bucket{le="0.5",worker="w"} 1\n'
            'gen_seconds_bucket{le="1",worker="w"} 4\n'
            'gen_seconds_bucket{le="+Inf",worker="w"} 9\n'
            "gen_seconds_sum 12.5\n"
            "gen_seconds_count 9\n"
        )
        out = central_main._parse_prometheus_full(text)
        les = [e["labels"]["le"] for e in out["gen_seconds_bucket"]]
        assert les == ["0.5", "1", "+Inf"]
        assert out["gen_seconds_sum"][0]["value"] == 12.5
        assert out["gen_seconds_count"][0]["value"] == 9.0

    def test_comments_blank_lines_and_junk_skipped(self, central_main):
        text = "# comment\n\nnot a metric line at all !!!\nok_metric 1\n"
        out = central_main._parse_prometheus_full(text)
        assert list(out.keys()) == ["ok_metric"]

    def test_non_numeric_value_skipped(self, central_main):
        out = central_main._parse_prometheus_full("bad_metric not_a_number\n")
        assert "bad_metric" not in out

    def test_escaped_quote_in_label_current_limitation(self, central_main):
        """KNOWN LIMITATION (documented, not fixed here).

        The label regex ``(\\w+)="([^\"]*)"`` does not understand backslash
        escapes, so a label value containing ``\\"`` is truncated at the
        escaped quote. Prometheus label values CAN legally contain escaped
        quotes; if the router ever emits one (e.g. in a route or error label),
        central will silently record a truncated label value. This test pins
        the current behaviour so a future fix flips it consciously.
        """
        text = 'm{route="/gen\\"x"} 3\n'
        out = central_main._parse_prometheus_full(text)
        assert out["m"][0]["value"] == 3.0
        # The correct parse would be '/gen"x'; today we get the truncated '/gen\'.
        assert out["m"][0]["labels"]["route"] == "/gen\\"


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
        t = time.time()
        central_main._push_sample("k", t, 100)
        central_main._push_sample("k", t + 5, 110)
        # Drop => router restarted => whole buffer must be discarded.
        dq = central_main._push_sample("k", t + 10, 5)
        assert list(dq) == [(t + 10, 5)]

    def test_equal_value_is_not_a_reset(self, central_main):
        t = time.time()
        central_main._push_sample("k", t, 7)
        dq = central_main._push_sample("k", t + 5, 7)
        assert len(dq) == 2

    def test_retention_trim_keeps_at_least_one_sample(self, central_main):
        t = time.time()
        max_s = central_main._BUFFER_SECONDS_MAX
        central_main._push_sample("k", t, 1)
        central_main._push_sample("k", t + 10, 2)
        dq = central_main._push_sample("k", t + max_s + 11, 3)
        assert list(dq) == [(t + max_s + 11, 3)]


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
        central_main._push_sample("k", now - 100, 0)     # outside 30s window
        central_main._push_sample("k", now - 20, 50)
        central_main._push_sample("k", now - 0.01, 80)
        delta, duration = central_main._window_delta("k", 30)
        assert delta == 30  # 80 - 50, NOT 80 - 0
        assert 19 <= duration <= 21

    def test_stale_series_returns_none(self, central_main):
        """A series that stopped updating before the window began reports None
        (the doc-comment scenario: router restart leaves frozen counters)."""
        now = time.time()
        central_main._push_sample("k", now - 500, 100)
        central_main._push_sample("k", now - 400, 200)
        assert central_main._window_delta("k", 60) is None

    def test_reset_mid_window(self, central_main):
        """Counter reset inside the window: _push_sample clears the buffer, so
        immediately after the reset there is a single sample (None), and the
        next sample yields only the post-reset delta."""
        now = time.time()
        central_main._push_sample("k", now - 20, 100)
        central_main._push_sample("k", now - 10, 5)      # reset -> buffer cleared
        assert central_main._window_delta("k", 60) is None
        central_main._push_sample("k", now, 8)
        delta, duration = central_main._window_delta("k", 60)
        assert delta == 3
        assert 9 <= duration <= 11


# ---------------------------------------------------------------------------
# rps_history bucketing (via the endpoint coroutine, no HTTP server)
# ---------------------------------------------------------------------------

class TestRpsHistory:
    def _seed_constant_rate(self, central_main, rps: float, span_s: int, step_s: int = 5):
        now = time.time()
        total = 0.0
        for ts in range(-span_s, 1, step_s):
            central_main._push_sample("global:requests", now + ts, total)
            total += rps * step_s

    def test_constant_rate_reproduced(self, central_main):
        self._seed_constant_rate(central_main, rps=2.0, span_s=120)
        result = asyncio.run(central_main.get_rps_history(window=60))
        assert result["window_seconds"] == 60
        samples = result["samples"]
        assert len(samples) >= 2
        # Every bucket should carry ~2.0 rps (rate seeding covers the left edge).
        for s in samples:
            assert s["rps"] == pytest.approx(2.0, abs=0.5)

    def test_empty_history_returns_no_samples(self, central_main):
        result = asyncio.run(central_main.get_rps_history(window=60))
        assert result["samples"] == []

    def test_window_clamped_to_allowed(self, central_main):
        result = asyncio.run(central_main.get_rps_history(window=47))
        assert result["window_seconds"] in central_main._ALLOWED_HISTORY

    def test_timestamps_span_requested_window(self, central_main):
        self._seed_constant_rate(central_main, rps=1.0, span_s=120)
        before = time.time()
        result = asyncio.run(central_main.get_rps_history(window=60))
        samples = result["samples"]
        assert samples[0]["ts"] == pytest.approx(before - 60, abs=2)
        assert samples[-1]["ts"] < before + 2


# ---------------------------------------------------------------------------
# prometheus_stats endpoint edge cases (no scrape yet)
# ---------------------------------------------------------------------------

class TestPrometheusStatsWarmup:
    def test_returns_incomplete_before_first_scrape(self, central_main):
        result = asyncio.run(central_main.get_prometheus_stats(window=60))
        assert result.get("incomplete") is True

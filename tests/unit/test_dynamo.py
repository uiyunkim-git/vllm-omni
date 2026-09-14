"""Unit tests for central/dynamo.py — the data-plane contract.

Everything else in central (deploy plumbing, health checks, the metrics
scraper) derives namespaces, URLs, ports and metric meaning from this module,
so these tests are the load-bearing ones.

No network: the few httpx call sites are exercised through
``httpx.MockTransport`` with recorded-looking payloads.
"""

import base64
import json

import httpx
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# namespace_for
# ---------------------------------------------------------------------------

class TestNamespaceFor:
    def test_slugs_model_id(self, central_dynamo):
        assert central_dynamo.namespace_for("openai/gpt-oss-120b") == (
            "dynamo-openai-gpt-oss-120b"
        )

    def test_lowercases_and_collapses_separators(self, central_dynamo):
        assert central_dynamo.namespace_for("Qwen/Qwen3__Embedding 0.6B") == (
            "dynamo-qwen-qwen3-embedding-0-6b"
        )

    def test_strips_leading_and_trailing_separators(self, central_dynamo):
        ns = central_dynamo.namespace_for("//weird--name__")
        assert ns == "dynamo-weird-name"
        assert not ns.endswith("-")

    def test_uses_the_configured_prefix(self, central_dynamo, monkeypatch):
        monkeypatch.setattr(central_dynamo, "NAMESPACE_PREFIX", "omni")
        assert central_dynamo.namespace_for("m1").startswith("omni-")

    def test_empty_model_falls_back_to_model(self, central_dynamo):
        assert central_dynamo.namespace_for("") == "dynamo-model"
        assert central_dynamo.namespace_for(None) == "dynamo-model"

    def test_capped_at_63_chars(self, central_dynamo):
        long_name = "someorg/" + ("very-long-model-name-" * 10)
        ns = central_dynamo.namespace_for(long_name)
        assert len(ns) == 63
        assert ns.startswith("dynamo-someorg-very-long-model-name")

    def test_replicas_of_one_model_share_a_namespace(self, central_dynamo):
        """Replicas form a single worker pool, so they must land in the same
        namespace; two different models must not."""
        a = central_dynamo.namespace_for("openai/gpt-oss-120b")
        b = central_dynamo.namespace_for("openai/gpt-oss-120b")
        c = central_dynamo.namespace_for("openai/gpt-oss-20b")
        assert a == b
        assert a != c


# ---------------------------------------------------------------------------
# infer_parsers
# ---------------------------------------------------------------------------

class TestInferParsers:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("openai/gpt-oss-120b", ("gpt_oss", "harmony")),
            ("OPENAI/GPT-OSS-20B", ("gpt_oss", "harmony")),
            ("google/gemma-4-27b-it", ("gemma4", "gemma4")),
            ("google/gemma4-27b", ("gemma4", "gemma4")),
            ("Qwen/Qwen3-32B", ("qwen3", "hermes")),
            ("deepseek-ai/deepseek-v4", ("deepseek_v4", "deepseek_v4")),
            ("meta-llama/Llama-3.1-8B-Instruct", (None, None)),
            ("", (None, None)),
            (None, (None, None)),
        ],
    )
    def test_by_model_family(self, central_dynamo, model, expected):
        assert central_dynamo.infer_parsers(model) == expected


# ---------------------------------------------------------------------------
# node_url / node_api_port / is_dynamo
# ---------------------------------------------------------------------------

NODE = {"host": "10.0.0.7", "port": 21001, "name": "dynamo_dep_w_0"}


class TestNodeAddressing:
    def test_dynamo_engine_uses_system_port_over_http(self, central_dynamo):
        dep = {"engine": "dynamo"}
        assert central_dynamo.node_api_port(dep, NODE) == 31001
        assert central_dynamo.node_url(dep, NODE) == "http://10.0.0.7:31001"

    def test_engine_defaults_to_dynamo(self, central_dynamo):
        assert central_dynamo.is_dynamo({}) is True
        assert central_dynamo.is_dynamo(None) is True
        assert central_dynamo.node_url({}, NODE) == "http://10.0.0.7:31001"

    def test_legacy_vllm_engine_uses_api_port_over_https(self, central_dynamo):
        dep = {"engine": "vllm"}
        assert central_dynamo.is_dynamo(dep) is False
        assert central_dynamo.node_api_port(dep, NODE) == 61001
        assert central_dynamo.node_url(dep, NODE) == "https://10.0.0.7:61001"

    def test_offsets_match_the_documented_port_scheme(self, central_dynamo):
        assert central_dynamo.SYSTEM_PORT_OFFSET == 10000
        assert central_dynamo.RPC_PORT_OFFSET == 12000
        assert central_dynamo.RESP_PORT_OFFSET == 42000
        assert central_dynamo.KV_PORT_OFFSET == 44000
        assert central_dynamo.VLLM_API_PORT_OFFSET == 40000

    def test_system_port_stays_below_the_i16_limit(self, central_dynamo):
        """Dynamo parses DYN_SYSTEM_PORT as i16, so base+10000 must fit in
        32767 for the whole allocator range (21001..)."""
        assert 21001 + central_dynamo.SYSTEM_PORT_OFFSET <= 32767


# ---------------------------------------------------------------------------
# parse_prometheus
# ---------------------------------------------------------------------------

class TestParsePrometheus:
    def test_plain_counter(self, central_dynamo):
        out = central_dynamo.parse_prometheus(
            "# HELP foo help\n# TYPE foo counter\nfoo 42\n"
        )
        assert out == {"foo": [{"labels": {}, "value": 42.0}]}

    def test_labeled_series(self, central_dynamo):
        text = (
            'dynamo_frontend_requests_total{model="a",status="ok"} 10\n'
            'dynamo_frontend_requests_total{model="b"} 3\n'
        )
        out = central_dynamo.parse_prometheus(text)
        entries = out["dynamo_frontend_requests_total"]
        assert entries[0]["labels"] == {"model": "a", "status": "ok"}
        assert entries[0]["value"] == 10.0
        assert entries[1]["labels"] == {"model": "b"}

    def test_metric_names_with_colons(self, central_dynamo):
        out = central_dynamo.parse_prometheus(
            'vllm:num_requests_running{model_name="m"} 4\n'
        )
        assert out["vllm:num_requests_running"][0]["value"] == 4.0

    def test_histogram_buckets_keep_order_and_inf(self, central_dynamo):
        text = (
            'h_bucket{le="0.5",model="m"} 1\n'
            'h_bucket{le="1",model="m"} 4\n'
            'h_bucket{le="+Inf",model="m"} 9\n'
            "h_sum 12.5\n"
            "h_count 9\n"
        )
        out = central_dynamo.parse_prometheus(text)
        assert [e["labels"]["le"] for e in out["h_bucket"]] == ["0.5", "1", "+Inf"]
        assert out["h_sum"][0]["value"] == 12.5

    def test_comments_blank_lines_and_junk_skipped(self, central_dynamo):
        text = "# comment\n\n   \nnot a metric line at all !!!\nok_metric 1\n"
        out = central_dynamo.parse_prometheus(text)
        assert list(out.keys()) == ["ok_metric"]

    def test_non_numeric_value_skipped(self, central_dynamo):
        assert central_dynamo.parse_prometheus("bad_metric not_a_number\n") == {}

    def test_scientific_and_negative_values(self, central_dynamo):
        out = central_dynamo.parse_prometheus("a 1.5e3\nb -2\n")
        assert out["a"][0]["value"] == 1500.0
        assert out["b"][0]["value"] == -2.0

    def test_empty_input(self, central_dynamo):
        assert central_dynamo.parse_prometheus("") == {}


# ---------------------------------------------------------------------------
# frontend_summary
# ---------------------------------------------------------------------------

class TestFrontendSummary:
    @pytest.fixture
    def summary(self, central_dynamo, frontend_metrics_text):
        return central_dynamo.frontend_summary(
            central_dynamo.parse_prometheus(frontend_metrics_text)
        )

    def test_models_discovered(self, summary):
        assert set(summary["per_model"]) == {
            "openai/gpt-oss-120b",
            "Qwen/Qwen3-Embedding-0.6B",
        }

    def test_per_model_counters(self, summary):
        m = summary["per_model"]["openai/gpt-oss-120b"]
        assert m["requests_total"] == 1200
        assert m["active_requests"] == 3
        assert m["queued_requests"] == 2
        assert m["output_tokens_total"] == 254000
        assert m["latency_sum"] == pytest.approx(3600.5)
        assert m["latency_count"] == 1200
        assert m["ttft_sum"] == pytest.approx(288.0)
        assert m["ttft_count"] == 1200
        assert m["migrations_total"] == 4
        assert m["rejections_total"] == 1

    def test_buckets_drop_inf(self, summary):
        m = summary["per_model"]["openai/gpt-oss-120b"]
        assert m["latency_buckets"] == {"0.5": 100, "1": 400, "2.5": 900, "5": 1150}
        assert m["ttft_buckets"] == {"0.1": 200, "0.25": 800, "0.5": 1100}
        assert "+Inf" not in m["latency_buckets"]

    def test_model_without_a_metric_reads_zero(self, summary):
        emb = summary["per_model"]["Qwen/Qwen3-Embedding-0.6B"]
        assert emb["requests_total"] == 48
        assert emb["migrations_total"] == 0
        assert emb["ttft_count"] == 0

    def test_ready_is_false_when_frontend_does_not_export_it(self, summary):
        """dynamo 1.4.2 exports no dynamo_frontend_model_ready — central derives
        readiness from etcd instead, so the summary must not claim ready."""
        assert summary["per_model"]["openai/gpt-oss-120b"]["ready"] is False

    def test_ready_reflects_the_metric_when_present(self, central_dynamo):
        raw = central_dynamo.parse_prometheus(
            'dynamo_frontend_model_ready{model="m"} 1\n'
        )
        assert central_dynamo.frontend_summary(raw)["per_model"]["m"]["ready"] is True

    def test_global_totals_sum_across_models(self, summary):
        assert summary["requests_total"] == 1248
        assert summary["active_requests"] == 3
        assert summary["queued_requests"] == 2
        assert summary["output_tokens_total"] == 254000

    def test_empty_scrape(self, central_dynamo):
        out = central_dynamo.frontend_summary({})
        assert out["per_model"] == {}
        assert out["requests_total"] == 0


# ---------------------------------------------------------------------------
# instance_summary
# ---------------------------------------------------------------------------

class TestInstanceSummary:
    @pytest.fixture
    def summary(self, central_dynamo, instance_metrics_text):
        return central_dynamo.instance_summary(
            central_dynamo.parse_prometheus(instance_metrics_text)
        )

    def test_only_the_generate_endpoint_counts(self, summary):
        """`load_metrics`/`clear_kv_blocks` are bookkeeping endpoints; counting
        them would make the served-request numbers meaningless."""
        assert summary["requests_total"] == 830
        assert summary["errors_total"] == 2

    def test_dynamo_component_gauges(self, summary):
        assert summary["inflight"] == 3
        assert summary["duration_sum"] == pytest.approx(2481.25)
        assert summary["duration_count"] == 830
        assert summary["uptime_s"] == 7200

    def test_vllm_engine_gauges(self, summary):
        assert summary["running"] == 3
        assert summary["waiting"] == 5
        assert summary["kv_cache_usage_pct"] == 42.0

    def test_kv_usage_falls_back_to_the_dynamo_component_metric(self, central_dynamo):
        raw = central_dynamo.parse_prometheus(
            "dynamo_component_gpu_cache_usage_percent 37.5\n"
        )
        assert central_dynamo.instance_summary(raw)["kv_cache_usage_pct"] == 37.5

    def test_empty_scrape_is_all_zeroes(self, central_dynamo):
        out = central_dynamo.instance_summary({})
        assert out == {
            "requests_total": 0, "errors_total": 0, "inflight": 0,
            "duration_sum": 0.0, "duration_count": 0, "running": 0,
            "waiting": 0, "kv_cache_usage_pct": 0.0, "uptime_s": 0,
        }


# ---------------------------------------------------------------------------
# percentile_from_buckets
# ---------------------------------------------------------------------------

class TestPercentileFromBuckets:
    def test_empty_returns_none(self, central_dynamo):
        assert central_dynamo.percentile_from_buckets({}, 0.5) is None

    def test_all_zero_counts_returns_none(self, central_dynamo):
        assert central_dynamo.percentile_from_buckets({"1": 0, "2": 0}, 0.5) is None

    def test_single_bucket_interpolates_from_zero(self, central_dynamo):
        # 10 observations, all <= 1.0s, spread uniformly over [0, 1].
        assert central_dynamo.percentile_from_buckets({"1.0": 10}, 0.5) == pytest.approx(0.5)
        assert central_dynamo.percentile_from_buckets({"1.0": 10}, 0.95) == pytest.approx(0.95)

    def test_interpolates_inside_the_matching_bucket(self, central_dynamo):
        buckets = {"0.1": 0, "0.5": 10, "1.0": 20}
        # p50 -> target 10 lands exactly on the 0.5 boundary
        assert central_dynamo.percentile_from_buckets(buckets, 0.5) == pytest.approx(0.5)
        # p25 -> target 5, halfway into [0.1, 0.5]
        assert central_dynamo.percentile_from_buckets(buckets, 0.25) == pytest.approx(0.3)
        # p75 -> target 15, halfway into [0.5, 1.0]
        assert central_dynamo.percentile_from_buckets(buckets, 0.75) == pytest.approx(0.75)

    def test_unsorted_input_is_ordered_numerically(self, central_dynamo):
        """String keys must be compared as floats: '10' < '9' lexically."""
        buckets = {"10": 20, "9": 10, "1": 0}
        assert central_dynamo.percentile_from_buckets(buckets, 0.25) == pytest.approx(5.0)

    def test_window_deltas_work_the_same(self, central_dynamo):
        """The caller feeds per-window deltas, not absolute counters."""
        assert central_dynamo.percentile_from_buckets(
            {"0.25": 2.0, "0.5": 4.0}, 0.5
        ) == pytest.approx(0.25)

    def test_p100_returns_the_largest_bucket_edge(self, central_dynamo):
        assert central_dynamo.percentile_from_buckets({"0.5": 5, "2": 10}, 1.0) == 2.0


# ---------------------------------------------------------------------------
# etcd helpers
# ---------------------------------------------------------------------------

class TestRangeEnd:
    def test_prefix_scan_increments_the_last_byte(self, central_dynamo):
        end = base64.b64decode(central_dynamo._range_end("v1/instances/")).decode()
        assert end == "v1/instances0"  # '/' (0x2f) -> '0' (0x30)

    def test_range_end_sorts_after_every_key_with_the_prefix(self, central_dynamo):
        prefix = "v1/instances/"
        end = base64.b64decode(central_dynamo._range_end(prefix)).decode()
        for key in (prefix, prefix + "a", prefix + "zzz/backend/generate/1"):
            assert key < end

    def test_trailing_0xff_byte_carries_to_the_previous_byte(self, central_dynamo):
        # 'ÿ' is 0xc3 0xbf in utf-8; the last byte increments.
        assert base64.b64decode(central_dynamo._range_end("aÿ")) == b"a\xc3\xc0"


class TestGenerateInstancesByNamespace:
    KEYS = [
        "v1/instances/dynamo-openai-gpt-oss-120b/backend/generate/7587865234",
        "v1/instances/dynamo-openai-gpt-oss-120b/backend/generate/7587865299",
        "v1/instances/dynamo-openai-gpt-oss-120b/backend/load_metrics/7587865234",
        "v1/instances/dynamo-openai-gpt-oss-120b/backend/clear_kv_blocks/7587865234",
        "v1/instances/dynamo-qwen-qwen3-embedding-0-6b/backend/generate/9001",
        "v1/components/dynamo-openai-gpt-oss-120b/backend/generate/7587865234",
        "v1/instances/too/short",
    ]

    def test_groups_generate_endpoints_by_namespace(self, central_dynamo):
        out = central_dynamo.generate_instances_by_namespace(self.KEYS)
        assert out == {
            "dynamo-openai-gpt-oss-120b": ["7587865234", "7587865299"],
            "dynamo-qwen-qwen3-embedding-0-6b": ["9001"],
        }

    def test_replica_count_is_the_instance_count(self, central_dynamo):
        out = central_dynamo.generate_instances_by_namespace(self.KEYS)
        assert len(out["dynamo-openai-gpt-oss-120b"]) == 2

    def test_no_keys(self, central_dynamo):
        assert central_dynamo.generate_instances_by_namespace([]) == {}


# ---------------------------------------------------------------------------
# httpx call sites (mock transport — no sockets)
# ---------------------------------------------------------------------------

def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestHttpCallSites:
    async def test_etcd_instance_keys_posts_a_prefix_range(self, central_dynamo):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            keys = [
                "v1/instances/dynamo-m/backend/generate/1",
                "v1/instances/dynamo-m/backend/generate/2",
            ]
            return httpx.Response(
                200,
                json={"kvs": [{"key": base64.b64encode(k.encode()).decode()} for k in keys]},
            )

        async with _client(handler) as c:
            keys = await central_dynamo.etcd_instance_keys(c)

        assert seen["url"] == f"{central_dynamo.ETCD_ENDPOINT}/v3/kv/range"
        assert base64.b64decode(seen["body"]["key"]).decode() == "v1/instances/"
        assert seen["body"]["range_end"] == central_dynamo._range_end("v1/instances/")
        assert seen["body"]["keys_only"] is True
        assert keys == [
            "v1/instances/dynamo-m/backend/generate/1",
            "v1/instances/dynamo-m/backend/generate/2",
        ]

    async def test_etcd_empty_response(self, central_dynamo):
        async with _client(lambda r: httpx.Response(200, json={"header": {}})) as c:
            assert await central_dynamo.etcd_instance_keys(c) == []

    async def test_frontend_models(self, central_dynamo):
        def handler(request):
            assert request.url.path == "/v1/models"
            return httpx.Response(200, json={"object": "list", "data": [{"id": "m1"}]})

        async with _client(handler) as c:
            assert await central_dynamo.frontend_models(c) == [{"id": "m1"}]

    async def test_frontend_metrics_parses_the_body(
        self, central_dynamo, frontend_metrics_text
    ):
        def handler(request):
            assert request.url.path == "/metrics"
            return httpx.Response(200, text=frontend_metrics_text)

        async with _client(handler) as c:
            raw = await central_dynamo.frontend_metrics(c)
        assert "dynamo_frontend_requests_total" in raw

    async def test_scrape_instance_summarises(self, central_dynamo, instance_metrics_text):
        async with _client(lambda r: httpx.Response(200, text=instance_metrics_text)) as c:
            out = await central_dynamo.scrape_instance(c, "http://10.0.0.7:31001")
        assert out["requests_total"] == 830

    async def test_scrape_instance_returns_none_on_failure(self, central_dynamo):
        async with _client(lambda r: httpx.Response(503, text="nope")) as c:
            assert await central_dynamo.scrape_instance(c, "http://10.0.0.7:31001") is None

        def boom(request):
            raise httpx.ConnectError("refused", request=request)

        async with _client(boom) as c:
            assert await central_dynamo.scrape_instance(c, "http://10.0.0.7:31001") is None

    async def test_instance_health_requires_status_ready(self, central_dynamo):
        """A dynamo worker answers /health long before the engine is loaded;
        only status == "ready" means the generate endpoint is registered."""
        async with _client(lambda r: httpx.Response(200, json={"status": "ready"})) as c:
            assert await central_dynamo.instance_health(c, "http://h:31001") is True

        async with _client(lambda r: httpx.Response(200, json={"status": "notready"})) as c:
            assert await central_dynamo.instance_health(c, "http://h:31001") is False

        async with _client(lambda r: httpx.Response(503, text="")) as c:
            assert await central_dynamo.instance_health(c, "http://h:31001") is False

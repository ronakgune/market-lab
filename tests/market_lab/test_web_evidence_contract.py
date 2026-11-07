from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from market_lab import mlab_ingest
from market_lab.web_evidence import FetchRequest, FetchResponse, ProviderHealth, SearchHit, SearchResponse, utcnow
from market_lab.web_evidence_cli import main as web_evidence_cli_main
from market_lab.web_evidence_providers import DirectHTTPProvider, OptionalProvider
from market_lab.web_evidence_runner import check_health, collect_for_claims, run_benchmark, verify_run
from market_lab.web_evidence_store import append_audit_chain, commit_snapshot, load_audit_chain, read_snapshot_manifest, verify_audit_chain


def _fetch_response(payload: bytes, *, content_type: str = "text/html; charset=utf-8") -> FetchResponse:
    return FetchResponse(
        request_id="fetch-1",
        run_id="run-1",
        claim_ids=["claim-1"],
        query_ids=["query-1"],
        provider_id="direct_http",
        provider_call_id="call-1",
        status="success",
        canonical_url="https://example.com/report",
        redirect_chain=["https://example.com/report"],
        content_type=content_type,
        byte_length=len(payload),
        payload=payload,
        response_headers={"content-type": content_type},
    )


class _FakeDDGS:
    provider_id = "ddgs"

    def health(self):
        return ProviderHealth("ddgs", utcnow(), "ready", ["search"])

    def search(self, request):
        return SearchResponse(
            request_id=request.request_id,
            query_id=request.query_id,
            provider_id="ddgs",
            status="success",
            hits=[
                SearchHit("ddgs", "r1", 1, "https://example.com/report?utm_source=x", "Search title", "SEARCH SNIPPET ONLY"),
                SearchHit("ddgs", "r2", 2, "https://example.com/report", "Duplicate", "SEARCH SNIPPET ONLY"),
            ],
            result_count=2,
            latency_ms=1,
        )


class _FakeDirect:
    provider_id = "direct_http"

    def health(self):
        return ProviderHealth("direct_http", utcnow(), "ready", ["fetch"])

    def fetch(self, request):
        body = b"<html><body><p>Example Corp revenue increased in the filed report.</p></body></html>"
        return FetchResponse(
            request_id=request.request_id,
            run_id=request.run_id,
            claim_ids=request.claim_ids,
            query_ids=request.query_ids,
            provider_id="direct_http",
            provider_call_id="call",
            status="success",
            canonical_url="https://example.com/report",
            redirect_chain=["https://example.com/report"],
            content_type="text/html",
            byte_length=len(body),
            payload=body,
            response_headers={"content-type": "text/html"},
        )


class _FakeOfficial:
    def __init__(self, provider_id: str, calls: list[str]) -> None:
        self.provider_id = provider_id
        self.calls = calls

    def health(self):
        return ProviderHealth(self.provider_id, utcnow(), "ready", ["fetch"])

    def fetch(self, request):
        self.calls.append(self.provider_id)
        body = f"Official adapter {self.provider_id} returned exact record for {request.url}".encode()
        return FetchResponse(
            request_id=request.request_id,
            run_id=request.run_id,
            claim_ids=request.claim_ids,
            query_ids=request.query_ids,
            provider_id=self.provider_id,
            provider_call_id=f"{self.provider_id}-call",
            status="success",
            canonical_url=f"https://official.example/{self.provider_id}",
            redirect_chain=[f"https://official.example/{self.provider_id}"],
            content_type="text/plain",
            byte_length=len(body),
            payload=body,
            response_headers={"content-type": "text/plain"},
        )


class WebEvidenceContractTests(unittest.TestCase):
    def test_audit_chain_uses_one_canonical_hash_and_detects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "audit_log.jsonl").write_text(json.dumps({"event": "legacy"}) + "\n", encoding="utf-8")
            append_audit_chain(run_dir, {"event_type": "first", "run_id": "run"})
            append_audit_chain(run_dir, {"event_type": "second", "run_id": "run"})

            self.assertEqual(verify_audit_chain(load_audit_chain(run_dir)), (True, ""))
            rows = load_audit_chain(run_dir)
            self.assertTrue(rows[1]["previous_event_hash"])

            rows = load_audit_chain(run_dir)
            rows[1]["event_type"] = "tampered"
            self.assertEqual(verify_audit_chain(rows), (False, "event hash mismatch"))

    def test_direct_http_returns_typed_unsafe_for_private_and_redirect(self) -> None:
        provider = DirectHTTPProvider()
        response = provider.fetch(FetchRequest(request_id="r", run_id="run", claim_ids=["c"], query_ids=["q"], url="http://127.0.0.1/"))
        self.assertEqual(response.status, "unsafe_url")

        def fake_fetch(url: str, *, timeout_seconds: int, max_bytes: int, deadline=None):
            return 302, b"", {"location": "http://127.0.0.1/secret"}

        with patch("market_lab.web_evidence_providers._single_pinned_fetch", fake_fetch):
            redirected = provider.fetch(
                FetchRequest(request_id="r2", run_id="run", claim_ids=["c"], query_ids=["q"], url="https://example.com/")
            )
        self.assertEqual(redirected.status, "unsafe_url")

    def test_snapshot_commit_is_immutable_and_extraction_is_typed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            payload = b"<html><head><title>Title</title></head><body><p>Revenue increased in the filed report.</p></body></html>"
            response = _fetch_response(payload)

            snapshot_id, manifest_path = commit_snapshot(
                run_dir,
                provider_id="direct_http",
                request_id="fetch-1",
                claim_ids=["claim-1"],
                query_ids=["query-1"],
                requested_url="https://example.com/report",
                response=response,
                response_body=payload,
                response_headers=response.response_headers,
            )
            extracted_path = manifest_path.parent / "extracted.txt"
            first_mtime = (manifest_path.stat().st_mtime_ns, extracted_path.stat().st_mtime_ns)

            snapshot_id_2, manifest_path_2 = commit_snapshot(
                run_dir,
                provider_id="direct_http",
                request_id="fetch-1",
                claim_ids=["claim-1"],
                query_ids=["query-1"],
                requested_url="https://example.com/report",
                response=response,
                response_body=payload,
                response_headers=response.response_headers,
            )
            self.assertEqual(snapshot_id_2, snapshot_id)
            self.assertEqual(manifest_path_2, manifest_path)
            self.assertEqual(first_mtime, (manifest_path.stat().st_mtime_ns, extracted_path.stat().st_mtime_ns))
            self.assertEqual(read_snapshot_manifest(run_dir, snapshot_id)["extraction_status"], "success")

            media = b"\x89PNG\r\n"
            media_response = _fetch_response(media, content_type="image/png")
            media_id, _ = commit_snapshot(
                run_dir,
                provider_id="direct_http",
                request_id="fetch-2",
                claim_ids=["claim-1"],
                query_ids=["query-1"],
                requested_url="https://example.com/image.png",
                response=media_response,
                response_body=media,
                response_headers=media_response.response_headers,
            )
            self.assertEqual(read_snapshot_manifest(run_dir, media_id)["extraction_status"], "unsupported_media")

    def test_pipeline_links_only_extracted_segments_and_records_dedupe_counterevidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            with patch("market_lab.web_evidence_runner.build_registry", lambda profile, include_optional: [_FakeDDGS(), _FakeDirect()]):
                result = collect_for_claims(
                    run_dir,
                    [{"claim_id": "claim-1", "text": "Example Corp revenue increased"}],
                    mode="live",
                    run_id="run-1",
                )

            self.assertGreaterEqual(result["evidence_added"], 1)
            evidence = [json.loads(line) for line in (run_dir / "evidence.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["source_type"] != "search_result" for row in evidence))
            self.assertTrue(all(row["stance"] == "context" for row in evidence))
            self.assertNotIn("SEARCH SNIPPET ONLY", {row["verbatim_excerpt_or_value"] for row in evidence})
            self.assertTrue(all(row["snapshot_id"].startswith("sha256:") and row["segment_id"].startswith("seg-") for row in evidence))
            queries = [json.loads(line) for line in (run_dir / "web_evidence" / "queries.jsonl").read_text().splitlines()]
            self.assertTrue({"primary_source", "counterevidence", "freshness_supersession"}.issubset({row["lane"] for row in queries}))
            calls = [json.loads(line) for line in (run_dir / "web_evidence" / "provider_calls.jsonl").read_text().splitlines()]
            self.assertTrue(any(row.get("event") == "dedupe_skip" for row in calls))

            checks = verify_run(
                run_dir,
                require_snapshots=True,
                require_counterevidence_coverage=True,
                require_audit_chain=True,
                require_zero_snippet_evidence=True,
                require_zero_execution_side_effects=True,
            )
            self.assertTrue(checks["ok"], checks)

            result_2 = collect_for_claims(
                run_dir,
                [{"claim_id": "claim-1", "text": "Example Corp revenue increased"}],
                mode="live",
                run_id="run-1",
            )
            evidence_2 = [json.loads(line) for line in (run_dir / "evidence.jsonl").read_text().splitlines()]
            queries_2 = [json.loads(line) for line in (run_dir / "web_evidence" / "queries.jsonl").read_text().splitlines()]
            self.assertEqual(len(evidence_2), len({row["evidence_id"] for row in evidence_2}))
            self.assertEqual(len(queries_2), len({row["query_id"] for row in queries_2}))
            self.assertEqual(result_2["evidence_added"], 0)

    def test_optional_provider_unconfigured_and_benchmark_fail_control(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            health = OptionalProvider("tavily", "TAVILY_API_KEY_DOES_NOT_EXIST").health()
            self.assertEqual(health.status, "unconfigured")
            self.assertEqual(health.missing_configuration, ["TAVILY_API_KEY_DOES_NOT_EXIST"])
            with patch.dict("os.environ", {"TAVILY_API_KEY_PRESENT_FOR_TEST": "x"}):
                configured = OptionalProvider("tavily", "TAVILY_API_KEY_PRESENT_FOR_TEST").health()
            self.assertEqual(configured.status, "disabled")
            self.assertNotEqual(configured.status, "ready")

            cases = run_dir / "cases.jsonl"
            cases.write_text(json.dumps({"claim_id": "bad"}) + "\n", encoding="utf-8")
            output = run_benchmark(run_dir / "bench", lane="frozen", cases_path=cases, output_path=run_dir / "out.json")
            self.assertFalse(output["passed"])
            self.assertEqual(output["cases_failed"], 1)

    def test_health_cli_output_and_optional_rows_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "health.json"
            payload = {
                "profile": "keyless_standard",
                "providers": [
                    {"provider_id": "ddgs", "status": "ready"},
                    {"provider_id": "direct_http", "status": "ready"},
                    {"provider_id": "tavily", "status": "unconfigured", "missing_configuration": ["TAVILY_API_KEY"]},
                    {"provider_id": "brave", "status": "unconfigured", "missing_configuration": ["BRAVE_API_KEY"]},
                    {"provider_id": "exa", "status": "unconfigured", "missing_configuration": ["EXA_API_KEY"]},
                    {"provider_id": "firecrawl", "status": "unconfigured", "missing_configuration": ["FIRECRAWL_API_KEY"]},
                    {"provider_id": "parallel", "status": "unconfigured", "missing_configuration": ["PARALLEL_API_KEY"]},
                    {"provider_id": "searxng", "status": "unconfigured", "missing_configuration": ["SEARXNG_BASE_URL"]},
                    {"provider_id": "jina_reader", "status": "unconfigured", "missing_configuration": ["JINA_API_KEY"]},
                ],
            }
            with patch("market_lab.web_evidence_cli.check_health", lambda **kwargs: payload):
                rc = web_evidence_cli_main(["health", "--profile", "keyless_standard", "--output", str(out), "--require-core-ready"])
            self.assertEqual(rc, 0)
            written = json.loads(out.read_text(encoding="utf-8"))
            by_id = {row["provider_id"]: row for row in written["providers"]}
            for name in ("tavily", "brave", "exa", "firecrawl", "parallel", "searxng", "jina_reader"):
                self.assertEqual(by_id[name]["status"], "unconfigured")
                self.assertNotIn("value", by_id[name])

    def test_official_identifier_routes_before_general_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            calls: list[str] = []
            ddgs = _FakeDDGS()

            def search_first(request):
                calls.append("ddgs")
                return _FakeDDGS.search(ddgs, request)

            ddgs.search = search_first
            providers = [
                ddgs,
                _FakeDirect(),
                _FakeOfficial("sec", calls),
                _FakeOfficial("crossref", calls),
                _FakeOfficial("arxiv", calls),
                _FakeOfficial("government_http", calls),
            ]
            with patch("market_lab.web_evidence_runner.build_registry", lambda profile, include_optional: providers):
                collect_for_claims(
                    Path(td),
                    [{"claim_id": "claim-doi", "text": "The DOI 10.1038/nphys1170 identifies a scholarly paper"}],
                    mode="live",
                    run_id="run-official",
                )
            self.assertIn("crossref", calls)
            self.assertLess(calls.index("crossref"), calls.index("ddgs"))

    def test_frozen_and_chaos_benchmark_are_non_empty_zero_network(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = Path("tests/market_lab/fixtures/web_evidence/benchmark_v1.jsonl")
            chaos = Path("tests/market_lab/fixtures/web_evidence/chaos_v1.jsonl")
            frozen = run_benchmark(root / "frozen", lane="frozen", cases_path=fixture, output_path=root / "frozen.json", fail_on_gate=True)
            chaos_result = run_benchmark(root / "chaos", lane="chaos", cases_path=chaos, output_path=root / "chaos.json", fail_on_gate=True)
            self.assertGreater(frozen["cases_total"], 0)
            self.assertEqual(frozen["metrics"]["network_calls"], 0)
            self.assertTrue(frozen["passed"])
            self.assertGreater(chaos_result["cases_total"], 0)
            self.assertEqual(chaos_result["metrics"]["network_calls"], 0)
            self.assertTrue(chaos_result["checks"]["typed_failures"])

            missing = root / "missing.jsonl"
            with self.assertRaises(RuntimeError):
                run_benchmark(root / "missing", lane="frozen", cases_path=missing, output_path=root / "missing.json", fail_on_gate=True)

    def test_frozen_collection_performs_zero_live_provider_calls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with patch("market_lab.web_evidence_providers.DDGSProvider.search", side_effect=AssertionError("network search")):
                with patch("market_lab.web_evidence_providers.DirectHTTPProvider.fetch", side_effect=AssertionError("network fetch")):
                    result = collect_for_claims(
                        Path(td),
                        [{"claim_id": "claim-frozen", "text": "Frozen claim context"}],
                        mode="frozen",
                        run_id="run-frozen",
                    )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["fetches"], 1)

    def test_mlab_ingest_web_evidence_hook_advances_without_disposition(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            (run_dir / "status.json").write_text(
                json.dumps({"run_id": "run", "stage": "claims_extracted", "verdict": "IN_PROGRESS", "owner": "qa", "next_owner": "qa"}),
                encoding="utf-8",
            )
            (run_dir / "claims.json").write_text(
                json.dumps({"run_id": "run", "claims": [{"claim_id": "claim-1", "text": "Example claim"}]}),
                encoding="utf-8",
            )

            result = mlab_ingest.run_web_evidence_research(run_dir, mode="off", owner="qa")
            status = mlab_ingest.read_status(run_dir)
            claims = mlab_ingest.read_claims(run_dir)["claims"]

            self.assertEqual(result["status"], "off")
            self.assertEqual(status["stage"], "research_active")
            self.assertEqual(status["verdict"], "IN_PROGRESS")
            self.assertIsNone(claims[0].get("disposition"))


if __name__ == "__main__":
    unittest.main()

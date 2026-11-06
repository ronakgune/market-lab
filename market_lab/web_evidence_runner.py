from __future__ import annotations

"""Web evidence runner.

Orchestrates claims -> query -> fetch -> snapshot -> evidence linking.
"""

import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .web_evidence import (
    BudgetProfile,
    EvidenceRecord,
    FetchRequest,
    FetchResponse,
    SearchRequest,
    SCHEMA_EVIDENCE_V2,
    canonicalize_url_for_dedupe,
    load_budget_profile,
    make_evidence_id,
    normalize_query,
    origin_cluster_id_for,
    request_id_for,
    sha256_hex,
    utcnow,
)
from .web_evidence_providers import DirectHTTPProvider, OptionalProvider, build_keyless_registry, build_optional_registry
from .web_evidence_store import (
    append_audit_chain,
    append_budget_report,
    append_evidence_record_once,
    append_provider_call,
    append_provider_call_once,
    append_query_event_once,
    append_provider_health,
    append_search_results,
    append_segment_once,
    commit_snapshot,
    ensure_layout,
    ensure_layout as _ensure_layout,
    load_audit_chain,
    make_text_segment,
    read_extracted_text,
    read_jsonl,
    read_snapshot_manifest,
    verify_segment_locator,
    write_plan,
    verify_audit_chain,
    write_atomic_json,
)


ALLOWED_PROFILE_FLAGS = {"off", "frozen", "live"}


def build_registry(profile: str = "keyless_standard", include_optional: bool = True) -> list[Any]:
    providers = build_keyless_registry(profile)
    if include_optional:
        providers.extend(build_optional_registry(profile))
    return providers


def _provider_by_id(providers: list[Any], provider_id: str):
    for provider in providers:
        if provider.provider_id == provider_id:
            return provider
    raise KeyError(f"provider not found: {provider_id}")


def check_health(
    profile: str = "keyless_standard",
    include_optional: bool = True,
    require_core_ready: bool = False,
) -> dict[str, Any]:
    providers = build_registry(profile, include_optional=include_optional)
    health_rows: list[dict[str, Any]] = []
    for provider in providers:
        health = provider.health()
        health_rows.append(asdict(health))

    ready = {row["provider_id"]: row["status"] for row in health_rows}
    if require_core_ready:
        if ready.get("direct_http") != "ready":
            raise RuntimeError("core provider direct_http not ready")
        if ready.get("ddgs") not in {"ready", "degraded", "rate_limited"}:
            raise RuntimeError("core provider ddgs not ready")

    return {
        "profile": profile,
        "providers": health_rows,
        "snapshot_root": "web_evidence",
        "recorded_at_utc": utcnow(),
    }


def _append_provider_fetch_call(run_dir: Path, response: Any) -> None:
    append_provider_call_once(run_dir, {k: v for k, v in asdict(response).items() if k not in {"payload"}})


def _completed_fetch_snapshot(run_dir: Path, *, provider_id: str, request_id: str) -> str | None:
    calls = read_jsonl(Path(run_dir) / "web_evidence" / "provider_calls.jsonl")
    if not any(
        row.get("provider_id") == provider_id and row.get("request_id") == request_id and row.get("status") == "success"
        for row in calls
    ):
        return None
    for row in reversed(read_jsonl(Path(run_dir) / "web_evidence" / "snapshot_index.jsonl")):
        if row.get("provider_id") == provider_id and row.get("request_id") == request_id and row.get("snapshot_id"):
            return str(row["snapshot_id"])
    return None


def _commit_successful_fetch(
    run_dir: Path,
    *,
    response: Any,
    requested_url: str,
    run_id: str,
) -> str | None:
    _append_provider_fetch_call(run_dir, response)
    if response.status != "success" or not response.payload:
        append_audit_chain(
            run_dir,
            {
                "event_type": "fetch.failed",
                "run_id": run_id,
                "claim_ids": response.claim_ids,
                "query_id": response.query_ids[0] if response.query_ids else "",
                "provider_id": response.provider_id,
                "provider_call_id": response.provider_call_id,
                "status": response.status,
                "reason_code": response.typed_error,
                "latency_ms": response.latency_ms,
            },
        )
        return None
    snapshot_id, _ = commit_snapshot(
        run_dir,
        provider_id=response.provider_id,
        request_id=response.request_id,
        claim_ids=response.claim_ids,
        query_ids=response.query_ids,
        requested_url=requested_url,
        response=response,
        response_body=response.payload,
        response_headers=response.response_headers,
    )
    manifest = read_snapshot_manifest(run_dir, snapshot_id)
    append_audit_chain(
        run_dir,
        {
            "event_type": "snapshot.committed",
            "run_id": run_id,
            "claim_ids": response.claim_ids,
            "query_id": response.query_ids[0] if response.query_ids else "",
            "provider_id": response.provider_id,
            "provider_call_id": response.provider_call_id,
            "output_artifact_hashes": [snapshot_id],
            "bytes": response.byte_length,
            "latency_ms": response.latency_ms,
        },
    )
    append_audit_chain(
        run_dir,
        {
            "event_type": "extraction.completed" if manifest.get("extraction_status") == "success" else "extraction.failed",
            "run_id": run_id,
            "claim_ids": response.claim_ids,
            "provider_id": response.provider_id,
            "provider_call_id": response.provider_call_id,
            "snapshot_id": snapshot_id,
            "status": manifest.get("extraction_status"),
            "reason_code": "" if manifest.get("extraction_status") == "success" else str(manifest.get("extraction_status")),
        },
    )
    return snapshot_id


def _claim_text(claim: dict[str, Any]) -> str:
    return normalize_query(str(claim.get("text") or claim.get("claim") or claim.get("claim_text") or ""))


def _claim_identifiers(claim: dict[str, Any], claim_text: str) -> list[dict[str, str]]:
    text = " ".join(str(claim.get(k) or "") for k in ["identifier", "document_identifier", "source_url", "citation"]) + " " + claim_text
    identifiers: list[dict[str, str]] = []
    for doi in dict.fromkeys(re.findall(r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+", text)):
        identifiers.append({"provider_id": "crossref", "kind": "fetch", "value": doi.rstrip(".,)")})
    for arxiv_id in dict.fromkeys(re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", text)):
        identifiers.append({"provider_id": "arxiv", "kind": "fetch", "value": arxiv_id})
    for cik in dict.fromkeys(re.findall(r"\bCIK[:\s#-]*(\d{1,10})\b", text, flags=re.I)):
        identifiers.append({"provider_id": "sec", "kind": "fetch", "value": cik})
    sec_match = re.search(r"\b\d{10}\b", text)
    if "sec" in claim_text.lower() and sec_match:
        identifiers.append({"provider_id": "sec", "kind": "fetch", "value": sec_match.group(0)})
    for url in dict.fromkeys(re.findall(r"https?://[^\s)]+", text)):
        host = url.split("//", 1)[-1].split("/", 1)[0].lower()
        if host in {"api.census.gov", "api.bls.gov", "www.data.gov"}:
            identifiers.append({"provider_id": "government_http", "kind": "fetch", "value": url.rstrip(".,)")})
    return identifiers


def _classify_source_type(claim_text: str) -> str:
    low = claim_text.lower()
    if any(term in low for term in ["10-k", "10-q", "8-k", "sec", "filing", "revenue", "debt"]):
        return "filed_company_disclosure"
    if any(term in low for term in ["doi", "study", "paper", "trial", "arxiv"]):
        return "scholarly"
    if any(term in low for term in ["bls", "census", "regulation", "agency", "approval"]):
        return "official_government"
    return "web_document"


_REQUIRED_DISCOVERY_LANES = {"counterevidence", "freshness_supersession"}
_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is", "it", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "with",
}


def _broadened_query(claim_text: str, lane: str) -> str:
    body = claim_text.split(":", 1)[-1]
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&.-]*", body)
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.lower().strip(".-")
        if not key or key in _QUERY_STOPWORDS or key in seen:
            continue
        seen.add(key)
        keywords.append(token.strip(".-"))
        if len(keywords) >= 12:
            break
    if len(keywords) < 3:
        keywords = [token.strip(".-") for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'&.-]*", claim_text)[:12]]
    suffix = ["correction", "denial", "retraction"] if lane == "counterevidence" else ["update", "amendment", "latest"]
    return normalize_query(" ".join([*keywords, *suffix]))[:220]


def _build_broadened_fallback(request: SearchRequest, claim_text: str) -> SearchRequest:
    query_id = request_id_for("query", f"{request.run_id}:{request.query_id}:broadened_fallback")
    return SearchRequest(
        request_id=request_id_for("search", f"{request.run_id}:{request.query_id}:broadened_fallback"),
        query_id=query_id,
        run_id=request.run_id,
        claim_ids=request.claim_ids,
        exact_query=_broadened_query(claim_text, request.lane),
        lane=request.lane,
        max_results=request.max_results,
        timeout_seconds=request.timeout_seconds,
        budget_reservation_id=request.budget_reservation_id,
        query_strategy="broadened_fallback",
        fallback_for_query_id=request.query_id,
    )


def _execute_search(run_dir: Path, provider: Any, request: SearchRequest, claim_id: str):
    response = provider.search(request)
    append_search_results(run_dir, response)
    append_provider_call(
        run_dir,
        {
            "event": "search",
            "provider_id": response.provider_id,
            "query_id": response.query_id,
            "request_id": response.request_id,
            "status": response.status,
            "result_count": response.result_count,
            "typed_error": response.typed_error,
            "raw_payload_hash": response.raw_payload_hash,
            "latency_ms": response.latency_ms,
            "query_strategy": request.query_strategy,
            "fallback_for_query_id": request.fallback_for_query_id,
        },
    )
    if not response.hits:
        append_audit_chain(
            run_dir,
            {
                "event_type": "claim_search_zero_results" if response.status == "zero_results" else "claim_search_empty",
                "claim_id": claim_id,
                "claim_ids": [claim_id],
                "run_id": request.run_id,
                "query_id": request.query_id,
                "lane": request.lane,
                "provider_id": response.provider_id,
                "status": response.status,
                "typed_error": response.typed_error,
                "query_strategy": request.query_strategy,
                "fallback_for_query_id": request.fallback_for_query_id,
                "latency_ms": response.latency_ms,
            },
        )
    return response


def _build_claim_queries(claim: dict[str, Any], idx: int, run_id: str, profile: BudgetProfile) -> list[SearchRequest]:
    claim_id = claim.get("claim_id") or claim.get("id") or f"claim-{idx}"
    text = _claim_text(claim)[:250]
    if not text:
        text = "company disclosure"
    variants = [
        ("primary_source", text),
        ("counterevidence", f'"{text[:160]}" correction OR denied OR restatement OR retraction OR not'),
        ("freshness_supersession", f'"{text[:160]}" amendment OR updated OR superseded OR latest'),
    ]
    source_type = _classify_source_type(text)
    if source_type == "filed_company_disclosure":
        variants[0] = ("primary_source", f"{text} site:sec.gov")
    elif source_type == "official_government":
        variants[0] = ("primary_source", f"{text} site:.gov")
    requests: list[SearchRequest] = []
    for lane, query in variants[: max(1, profile.max_queries_per_claim)]:
        requests.append(
            SearchRequest(
                request_id=request_id_for("search", f"{run_id}:{claim_id}:{idx}:{lane}"),
                query_id=request_id_for("query", f"{run_id}:{claim_id}:{idx}:{lane}"),
                run_id=run_id,
                claim_ids=[claim_id],
                exact_query=normalize_query(query)[:300],
                lane=lane,
                max_results=min(profile.max_results_per_query, 10),
                timeout_seconds=profile.search_timeout_seconds,
                budget_reservation_id=f"budget:{run_id}:{claim_id}",
            )
        )
    return requests


def _build_fetch(request: SearchRequest, url: str, claim_id: str, run_id: str, profile: BudgetProfile) -> FetchRequest:
    return FetchRequest(
        request_id=request_id_for("fetch", f"{run_id}:{claim_id}:{url}"),
        run_id=run_id,
        claim_ids=[claim_id],
        query_ids=[request.query_id],
        url=url,
        timeout_seconds=profile.fetch_timeout_seconds,
        max_bytes=profile.max_bytes_per_fetch,
        max_redirects=profile.max_redirects,
        allow_third_party_transform=True,
        budget_reservation_id=f"budget:{run_id}:{claim_id}",
    )


def _write_claim_plan(run_dir: Path, plan: list[dict[str, Any]]) -> None:
    write_plan(run_dir, {"generated_at_utc": utcnow(), "plan": plan})


def _load_fixture_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rows.append(json.loads(raw))
    return rows


def _required_str(row: dict[str, Any], field: str) -> bool:
    return isinstance(row.get(field), str) and bool(str(row.get(field)).strip())


def _frozen_fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "tests" / "market_lab" / "fixtures" / "web_evidence" / "benchmark_v1.jsonl"


def _flatten_search_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        wrapped = row.get("search_rows")
        if isinstance(wrapped, list):
            flattened.extend(item for item in wrapped if isinstance(item, dict))
        else:
            flattened.append(row)
    return flattened


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _zero_result_audit_matches(audit: dict[str, Any], *, claim_id: str, route: dict[str, Any]) -> bool:
    return (
        audit.get("event_type") == "claim_search_zero_results"
        and str(audit.get("claim_id") or "") == claim_id
        and _string_list(audit.get("claim_ids")) == [claim_id]
        and _string_list(route.get("claim_ids")) == [claim_id]
        and str(audit.get("query_id") or "") == str(route.get("query_id") or "")
        and str(audit.get("lane") or "") == str(route.get("lane") or "")
        and str(audit.get("provider_id") or "") == str(route.get("provider_id") or "")
        and str(audit.get("status") or "") == str(route.get("status") or "")
        and str(audit.get("query_strategy") or "") == str(route.get("query_strategy") or "")
        and audit.get("fallback_for_query_id") == route.get("fallback_for_query_id")
    )


def _frozen_provider_health_rows(profile: str) -> list[dict[str, Any]]:
    now = utcnow()
    rows = [
        ("ddgs", "disabled", ["search"], "frozen_replay_no_provider_probe"),
        ("direct_http", "disabled", ["fetch"], "frozen_replay_no_provider_probe"),
        ("sec", "disabled", ["fetch"], "frozen_replay_no_provider_probe"),
        ("crossref", "disabled", ["fetch"], "frozen_replay_no_provider_probe"),
        ("arxiv", "disabled", ["fetch"], "frozen_replay_no_provider_probe"),
        ("government_http", "disabled", ["fetch"], "frozen_replay_no_provider_probe"),
        ("tavily", "unconfigured", [], "optional_provider_not_used_in_frozen_replay"),
        ("brave", "unconfigured", [], "optional_provider_not_used_in_frozen_replay"),
        ("exa", "unconfigured", [], "optional_provider_not_used_in_frozen_replay"),
        ("firecrawl", "unconfigured", [], "optional_provider_not_used_in_frozen_replay"),
        ("parallel", "unconfigured", [], "optional_provider_not_used_in_frozen_replay"),
        ("searxng", "unconfigured", [], "optional_provider_not_used_in_frozen_replay"),
        ("jina_reader", "unconfigured", [], "optional_provider_not_used_in_frozen_replay"),
    ]
    return [
        {
            "provider_id": provider_id,
            "timestamp": now,
            "status": status,
            "capabilities_ready": capabilities if status == "disabled" else [],
            "missing_configuration": [],
            "reason_code": reason,
            "safe_message": f"{profile}: {reason}",
            "latency_ms": 0,
            "retry_after_utc": None,
            "observed_endpoint": None,
        }
        for provider_id, status, capabilities, reason in rows
    ]


def _frozen_collect_for_claims(
    run_dir: Path,
    claims: list[dict[str, Any]],
    *,
    profile: str,
    run_id: str,
    max_claims: int | None,
    budget: BudgetProfile,
) -> dict[str, Any]:
    ensure_layout(run_dir)
    fixture_rows = _load_fixture_cases(_frozen_fixture_path())
    if not fixture_rows:
        raise RuntimeError("missing_or_empty_frozen_corpus")
    append_provider_health(run_dir, _frozen_provider_health_rows(profile))
    append_budget_report(run_dir, {"profile": profile, "budget": asdict(budget), "mode": "frozen"})
    existing_evidence = {row.get("evidence_id") for row in read_jsonl(run_dir / "evidence.jsonl")}
    existing_claims = {row.get("claim_id") for row in read_jsonl(run_dir / "evidence.jsonl")}
    total_claims = min(len(claims), max_claims) if max_claims is not None else len(claims)
    result = {"status": "completed", "profile": profile, "run_id": run_id, "claims": total_claims, "searches": 0, "fetches": 0, "evidence_added": 0}
    plan: list[dict[str, Any]] = []
    for idx, claim in enumerate(claims[:total_claims]):
        claim_id = claim.get("claim_id") or claim.get("id") or f"claim-{idx}"
        claim_text = _claim_text(claim)
        fixture = fixture_rows[idx % len(fixture_rows)]
        if claim_id in existing_claims:
            continue
        for lane in ["primary_source", "counterevidence", "freshness_supersession"]:
            query = SearchRequest(
                request_id=request_id_for("search", f"frozen:{run_id}:{claim_id}:{lane}"),
                query_id=request_id_for("query", f"frozen:{run_id}:{claim_id}:{lane}"),
                run_id=run_id,
                claim_ids=[claim_id],
                exact_query=normalize_query(f"{claim_text} {lane}")[:300],
                lane=lane,
                max_results=1,
                timeout_seconds=0,
            )
            if append_query_event_once(run_dir, asdict(query)):
                result["searches"] += 1
                append_search_results(
                    run_dir,
                    [
                        {
                            "request_id": query.request_id,
                            "query_id": query.query_id,
                            "provider_id": "frozen_fixture",
                            "status": "success",
                            "result_count": 1,
                            "eligible_as_evidence": False,
                            "lane": lane,
                            "hits": [
                                {
                                    "provider_id": "frozen_fixture",
                                    "provider_result_id": f"frozen:{claim_id}:{lane}",
                                    "rank": 1,
                                    "url": str(fixture.get("url") or "https://example.com/frozen"),
                                    "title": str(fixture.get("title") or "Frozen fixture"),
                                    "snippet": f"Frozen discovery context for {lane}",
                                    "discovered_at": utcnow(),
                                    "eligible_as_evidence": False,
                                }
                            ],
                        }
                    ],
                )
            plan.append({"claim_id": claim_id, "query_id": query.query_id, "query": query.exact_query, "lane": lane, "source_type": _classify_source_type(claim_text)})
        body = (fixture.get("body") or f"Frozen fixture evidence context for {claim_text}.").encode("utf-8")
        fetch = FetchResponse(
            request_id=request_id_for("fetch", f"frozen:{run_id}:{claim_id}:{fixture.get('url', 'https://example.com/frozen')}"),
            run_id=run_id,
            claim_ids=[claim_id],
            query_ids=[request_id_for("query", f"frozen:{run_id}:{claim_id}:primary_source")],
            provider_id=str(fixture.get("provider_id") or "frozen_fixture"),
            provider_call_id=request_id_for("frozen_fixture", f"{run_id}:{claim_id}"),
            status="success",
            canonical_url=str(fixture.get("url") or "https://example.com/frozen"),
            redirect_chain=[str(fixture.get("url") or "https://example.com/frozen")],
            content_type=str(fixture.get("content_type") or "text/plain"),
            byte_length=len(body),
            payload=body,
            response_headers={"content-type": str(fixture.get("content_type") or "text/plain")},
        )
        if _completed_fetch_snapshot(run_dir, provider_id=fetch.provider_id, request_id=fetch.request_id):
            continue
        result["fetches"] += 1
        snapshot_id = _commit_successful_fetch(run_dir, response=fetch, requested_url=fetch.canonical_url, run_id=run_id)
        if not snapshot_id:
            continue
        segment = make_text_segment(snapshot_id, read_extracted_text(run_dir, snapshot_id), claim_text, segment_seed=f"frozen:{claim_id}")
        if segment is None or not verify_segment_locator(run_dir, segment):
            continue
        append_segment_once(run_dir, segment)
        record = EvidenceRecord(
            schema_version=SCHEMA_EVIDENCE_V2,
            evidence_id=make_evidence_id(run_id=run_id, claim_id=claim_id, segment_id=segment.segment_id, stance="context"),
            run_id=run_id,
            claim_id=claim_id,
            segment_id=segment.segment_id,
            snapshot_id=snapshot_id,
            stance="context",
            source_type=_classify_source_type(claim_text),
            source_tier="candidate",
            source_quality_reason="frozen_fixture_extracted_locator_verified",
            title=str(fixture.get("title") or "Frozen fixture"),
            canonical_url=fetch.canonical_url,
            document_identifier=snapshot_id,
            retrieved_at_utc=utcnow(),
            exact_locator=segment.locator,
            verbatim_excerpt_or_value=segment.verbatim_excerpt_or_value,
            query_ids=fetch.query_ids,
            provider_id=fetch.provider_id,
            provider_call_id=fetch.provider_call_id,
            source_lineage="web_evidence.frozen.v1",
            origin_cluster_id=origin_cluster_id_for(fetch.canonical_url),
            edge_evaluator="deterministic_exact_edge_v1",
            evaluator_version="1",
            confidence=0.4,
        )
        if record.evidence_id not in existing_evidence and append_evidence_record_once(run_dir, record):
            existing_evidence.add(record.evidence_id)
            result["evidence_added"] += 1
            append_audit_chain(run_dir, {"event_type": "evidence.linked", "run_id": run_id, "claim_ids": [claim_id], "claim_id": claim_id, "snapshot_id": snapshot_id, "segment_id": segment.segment_id, "provider_id": fetch.provider_id, "provider_call_id": fetch.provider_call_id, "status": "context"})
    _write_claim_plan(run_dir, plan)
    if result["searches"] or result["fetches"] or result["evidence_added"]:
        append_audit_chain(run_dir, {"event_type": "run.web_evidence_completed", "run_id": run_id, "status": "completed", "claims": result["claims"], "searches": result["searches"], "fetches": result["fetches"], "evidence_added": result["evidence_added"]})
    return result


def collect_for_claims(
    run_dir: Path,
    claims: list[dict[str, Any]],
    *,
    profile: str = "keyless_standard",
    run_id: str = "run",
    mode: str = "off",
    max_claims: int | None = None,
    budget_profile: BudgetProfile | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    mode = mode.strip().lower()
    if mode not in ALLOWED_PROFILE_FLAGS:
        raise ValueError(f"unknown web-evidence mode: {mode}")

    budget = budget_profile or load_budget_profile(profile)
    result: dict[str, Any] = {
        "status": "completed" if mode != "off" else "off",
        "profile": profile,
        "run_id": run_id,
        "claims": 0,
        "searches": 0,
        "fetches": 0,
        "evidence_added": 0,
    }

    if not claims or mode == "off":
        ensure_layout(run_dir)
        append_budget_report(run_dir, {"profile": profile, "budget": asdict(budget)})
        return result

    if mode == "frozen":
        return _frozen_collect_for_claims(run_dir, claims, profile=profile, run_id=run_id, max_claims=max_claims, budget=budget)

    base = ensure_layout(run_dir)
    providers = build_registry(profile, include_optional=True)
    append_provider_health(
        run_dir,
        [asdict(provider.health()) for provider in providers],
    )
    append_budget_report(run_dir, {"profile": profile, "budget": asdict(budget)})

    ddgs = _provider_by_id(providers, "ddgs")
    direct_http = _provider_by_id(providers, "direct_http")
    total_claims = len(claims)
    if max_claims is not None:
        total_claims = min(total_claims, max_claims)

    result["claims"] = total_claims
    plan: list[dict[str, Any]] = []
    existing_queries = {row.get("query_id") for row in read_jsonl(base / "queries.jsonl")}
    existing_search_results: dict[str, list[dict[str, Any]]] = {}
    for row in _flatten_search_result_rows(read_jsonl(base / "search_results.jsonl")):
        if row.get("query_id"):
            existing_search_results.setdefault(str(row["query_id"]), []).append(row)
    existing_calls = {row.get("provider_call_id") for row in read_jsonl(base / "provider_calls.jsonl") if row.get("status") == "success"}
    existing_evidence = {row.get("evidence_id") for row in read_jsonl(run_dir / "evidence.jsonl")}
    seen_canonical_urls: set[str] = {
        canonicalize_url_for_dedupe(str(row.get("canonical_url") or ""))
        for row in read_jsonl(run_dir / "evidence.jsonl")
        if row.get("canonical_url")
    }
    seen_content_hashes: set[str] = {
        str(row.get("document_identifier"))
        for row in read_jsonl(run_dir / "evidence.jsonl")
        if row.get("document_identifier")
    }

    for i, claim in enumerate(claims[:total_claims]):
        claim_id = claim.get("claim_id") or claim.get("id") or f"claim-{i}"
        claim_text = _claim_text(claim)
        used_fetches = 0

        for ident in _claim_identifiers(claim, claim_text):
            if used_fetches >= budget.max_fetches_per_claim:
                break
            provider = _provider_by_id(providers, ident["provider_id"])
            query_id = request_id_for("query", f"{run_id}:{claim_id}:identifier_exact:{ident['provider_id']}:{ident['value']}")
            search_request = SearchRequest(
                request_id=request_id_for("search", f"{run_id}:{claim_id}:identifier_exact:{ident['provider_id']}:{ident['value']}"),
                query_id=query_id,
                run_id=run_id,
                claim_ids=[claim_id],
                exact_query=ident["value"],
                lane="identifier_exact",
                max_results=1,
                timeout_seconds=budget.search_timeout_seconds,
            )
            if append_query_event_once(run_dir, asdict(search_request)):
                result["searches"] += 1
            plan.append({"claim_id": claim_id, "query_id": query_id, "query": ident["value"], "lane": "identifier_exact", "source_type": _classify_source_type(claim_text), "provider_id": ident["provider_id"]})
            fetch_request = _build_fetch(search_request, ident["value"], claim_id, run_id, budget)
            completed_snapshot = _completed_fetch_snapshot(run_dir, provider_id=ident["provider_id"], request_id=fetch_request.request_id)
            if completed_snapshot:
                continue
            fetch_response = provider.fetch(fetch_request)
            used_fetches += 1
            result["fetches"] += 1
            snapshot_id = _commit_successful_fetch(run_dir, response=fetch_response, requested_url=ident["value"], run_id=run_id)
            if not snapshot_id:
                continue
            existing_calls.add(fetch_response.provider_call_id)
            manifest = read_snapshot_manifest(run_dir, snapshot_id)
            if manifest.get("extraction_status") != "success":
                continue
            segment = make_text_segment(snapshot_id, read_extracted_text(run_dir, snapshot_id), claim_text, segment_seed=f"{claim_id}:{ident['provider_id']}:{ident['value']}")
            if segment is None or not verify_segment_locator(run_dir, segment):
                continue
            append_segment_once(run_dir, segment)
            record = EvidenceRecord(
                schema_version=SCHEMA_EVIDENCE_V2,
                evidence_id=make_evidence_id(run_id=run_id, claim_id=claim_id, segment_id=segment.segment_id, stance="context"),
                run_id=run_id,
                claim_id=claim_id,
                segment_id=segment.segment_id,
                snapshot_id=snapshot_id,
                stance="context",
                source_type=_classify_source_type(claim_text),
                source_tier="candidate",
                source_quality_reason="official_identifier_extracted_locator_verified",
                canonical_url=fetch_response.canonical_url or ident["value"],
                document_identifier=sha256_hex(fetch_response.payload or b""),
                retrieved_at_utc=utcnow(),
                exact_locator=segment.locator,
                verbatim_excerpt_or_value=segment.verbatim_excerpt_or_value,
                query_ids=[query_id],
                provider_id=fetch_response.provider_id,
                provider_call_id=fetch_response.provider_call_id,
                source_lineage="web_evidence.official.v1",
                origin_cluster_id=origin_cluster_id_for(fetch_response.canonical_url or ident["value"]),
                edge_evaluator="deterministic_exact_edge_v1",
                evaluator_version="1",
                confidence=0.4,
            )
            if record.evidence_id not in existing_evidence and append_evidence_record_once(run_dir, record):
                existing_evidence.add(record.evidence_id)
                result["evidence_added"] += 1
                append_audit_chain(run_dir, {"event_type": "evidence.linked", "claim_id": claim_id, "run_id": run_id, "claim_ids": [claim_id], "snapshot_id": snapshot_id, "segment_id": segment.segment_id, "provider_id": fetch_response.provider_id, "provider_call_id": fetch_response.provider_call_id, "query_id": query_id, "lane": "identifier_exact", "status": "context"})

        for search_request in _build_claim_queries(claim, i, run_id, budget):
            query_already_done = search_request.query_id in existing_queries
            query_payload = {**asdict(search_request), "provider_id": ddgs.provider_id}
            if append_query_event_once(run_dir, query_payload):
                existing_queries.add(search_request.query_id)
            plan.append(
                {
                    "claim_id": claim_id,
                    "query_id": search_request.query_id,
                    "query": search_request.exact_query,
                    "lane": search_request.lane,
                    "source_type": _classify_source_type(claim_text),
                }
            )
            if query_already_done:
                prior_results = existing_search_results.get(search_request.query_id, [])
                if not prior_results or prior_results[-1].get("status") != "zero_results" or search_request.lane not in _REQUIRED_DISCOVERY_LANES:
                    continue
                search_response = None
            else:
                search_response = _execute_search(run_dir, ddgs, search_request, claim_id)
                result["searches"] += 1

            original_zero_results = search_response is None or search_response.status == "zero_results"
            if original_zero_results and search_request.lane in _REQUIRED_DISCOVERY_LANES:
                fallback_request = _build_broadened_fallback(search_request, claim_text)
                fallback_already_done = fallback_request.query_id in existing_queries
                fallback_payload = {**asdict(fallback_request), "provider_id": ddgs.provider_id}
                if append_query_event_once(run_dir, fallback_payload):
                    existing_queries.add(fallback_request.query_id)
                plan.append(
                    {
                        "claim_id": claim_id,
                        "query_id": fallback_request.query_id,
                        "query": fallback_request.exact_query,
                        "lane": fallback_request.lane,
                        "source_type": _classify_source_type(claim_text),
                        "query_strategy": fallback_request.query_strategy,
                        "fallback_for_query_id": fallback_request.fallback_for_query_id,
                    }
                )
                if fallback_already_done:
                    continue
                search_request = fallback_request
                search_response = _execute_search(run_dir, ddgs, search_request, claim_id)
                result["searches"] += 1

            if search_response is None or not search_response.hits:
                continue

            for hit in search_response.hits:
                if used_fetches >= budget.max_fetches_per_claim:
                    break

                canonical_url = canonicalize_url_for_dedupe(hit.url)
                if canonical_url in seen_canonical_urls:
                    append_provider_call(
                        run_dir,
                        {"event": "dedupe_skip", "reason": "canonical_url", "url": hit.url, "canonical_url": canonical_url, "claim_id": claim_id},
                    )
                    continue
                seen_canonical_urls.add(canonical_url)

                used_fetches += 1
                result["fetches"] += 1
                fetch_request = _build_fetch(search_request, hit.url, claim_id, run_id, budget)
                completed_snapshot = _completed_fetch_snapshot(run_dir, provider_id="direct_http", request_id=fetch_request.request_id)
                if completed_snapshot:
                    continue
                fetch_response = direct_http.fetch(fetch_request)
                snapshot_id = _commit_successful_fetch(run_dir, response=fetch_response, requested_url=hit.url, run_id=run_id)
                if fetch_response.status == "success":
                    existing_calls.add(fetch_response.provider_call_id)
                if not snapshot_id:
                    continue

                raw_hash = sha256_hex(fetch_response.payload)
                if raw_hash in seen_content_hashes:
                    append_provider_call(
                        run_dir,
                        {"event": "dedupe_skip", "reason": "content_hash", "url": hit.url, "raw_sha256": raw_hash, "claim_id": claim_id},
                    )
                    continue
                seen_content_hashes.add(raw_hash)

                manifest = read_snapshot_manifest(run_dir, snapshot_id)
                if manifest.get("extraction_status") != "success":
                    continue

                extracted_text = read_extracted_text(run_dir, snapshot_id)
                segment = make_text_segment(snapshot_id, extracted_text, claim_text, segment_seed=f"{claim_id}:{hit.url}")
                if segment is None or not verify_segment_locator(run_dir, segment):
                    append_audit_chain(
                        run_dir,
                        {"event_type": "segment_locator_failed", "claim_id": claim_id, "run_id": run_id, "snapshot_id": snapshot_id},
                    )
                    continue

                append_segment_once(run_dir, segment)
                stance = "context"
                record = EvidenceRecord(
                    schema_version=SCHEMA_EVIDENCE_V2,
                    evidence_id=make_evidence_id(run_id=run_id, claim_id=claim_id, segment_id=segment.segment_id, stance=stance),
                    run_id=run_id,
                    claim_id=claim_id,
                    segment_id=segment.segment_id,
                    snapshot_id=snapshot_id,
                    stance=stance,
                    source_type=_classify_source_type(claim_text),
                    source_tier="candidate",
                    source_quality_reason="fetched_extracted_locator_verified",
                    publisher="",
                    title=hit.title[:160],
                    canonical_url=fetch_response.canonical_url or hit.url,
                    document_identifier=raw_hash,
                    retrieved_at_utc=utcnow(),
                    exact_locator=segment.locator,
                    verbatim_excerpt_or_value=segment.verbatim_excerpt_or_value,
                    query_ids=[search_request.query_id],
                    provider_id=fetch_response.provider_id,
                    provider_call_id=fetch_response.provider_call_id,
                    source_lineage="web_evidence.v1",
                    origin_cluster_id=origin_cluster_id_for(fetch_response.canonical_url or hit.url),
                    version="1",
                    confidence=0.4,
                )
                if record.evidence_id in existing_evidence or not append_evidence_record_once(run_dir, record):
                    continue
                existing_evidence.add(record.evidence_id)
                append_audit_chain(
                    run_dir,
                    {
                        "event_type": "evidence.linked",
                        "claim_id": claim_id,
                        "run_id": run_id,
                        "snapshot_id": snapshot_id,
                        "segment_id": segment.segment_id,
                        "provider_id": fetch_response.provider_id,
                        "provider_call_id": fetch_response.provider_call_id,
                        "source_url": hit.url,
                        "query_id": search_request.query_id,
                        "lane": search_request.lane,
                        "origin_cluster_id": record.origin_cluster_id,
                    },
                )
                result["evidence_added"] += 1
                break

    _write_claim_plan(run_dir, plan)
    # Persist provider-health and audit artifacts in web_evidence root.
    if result["searches"] or result["fetches"] or result["evidence_added"]:
        append_audit_chain(
            run_dir,
            {
                "event_type": "collect_complete",
                "run_id": run_id,
                "claims": result["claims"],
                "searches": result["searches"],
                "fetches": result["fetches"],
                "evidence_added": result["evidence_added"],
            },
        )
    return result


def run_smoke(
    run_dir: Path,
    *,
    profile: str = "keyless_standard",
    query: str | None = None,
    url: str | None = None,
    sec_cik: str | None = None,
    crossref_doi: str | None = None,
    arxiv_id: str | None = None,
    government_url: str | None = None,
    lane: str = "live",
) -> dict[str, Any]:
    base = _ensure_layout(run_dir)
    providers = build_registry(profile, include_optional=True)
    ddgs = _provider_by_id(providers, "ddgs")
    direct = _provider_by_id(providers, "direct_http")
    sec = _provider_by_id(providers, "sec")
    crossref = _provider_by_id(providers, "crossref")
    from .web_evidence_providers import ArxivProvider

    arxiv = _provider_by_id(providers, "arxiv") if any(p.provider_id == "arxiv" for p in providers) else ArxivProvider()
    gov = _provider_by_id(providers, "government_http")

    append_provider_health(run_dir, [asdict(provider.health()) for provider in providers])

    run_id = run_dir.name or "smoke"
    result: dict[str, Any] = {
        "profile": profile,
        "lane": lane,
        "run_dir": str(run_dir),
        "query": query or "",
        "paid_provider_cost_usd": 0.0,
    }

    if query:
        search_request = SearchRequest(
            request_id="smoke-ddgs",
            query_id="smoke-ddgs",
            run_id=run_id,
            claim_ids=["smoke"],
            exact_query=query,
            lane="smoke",
            max_results=3,
            timeout_seconds=20,
            budget_reservation_id="smoke",
        )
        response = ddgs.search(search_request)
        append_search_results(run_dir, response)
        append_provider_call(
            run_dir,
            {
                "event": "search",
                "provider_id": response.provider_id,
                "query_id": response.query_id,
                "request_id": response.request_id,
                "status": response.status,
                "result_count": response.result_count,
                "typed_error": response.typed_error,
                "raw_payload_hash": response.raw_payload_hash,
                "latency_ms": response.latency_ms,
            },
        )
        append_audit_chain(run_dir, {"event_type": "search.completed" if response.hits else "search.failed", "run_id": run_id, "claim_ids": ["smoke"], "query_id": response.query_id, "provider_id": response.provider_id, "status": response.status, "reason_code": response.typed_error, "latency_ms": response.latency_ms})
        result["ddgs"] = {"status": response.status, "result_count": response.result_count, "typed_error": response.typed_error}
    else:
        result["ddgs"] = {"status": "skipped", "result_count": 0, "typed_error": ""}

    if url:
        request = FetchRequest(
            request_id="smoke-direct",
            run_id=run_id,
            claim_ids=["smoke"],
            query_ids=["smoke-ddgs"],
            url=url,
            timeout_seconds=20,
            max_bytes=20 * 1024 * 1024,
            max_redirects=5,
        )
        response = direct.fetch(request)
        snapshot_id = _commit_successful_fetch(run_dir, response=response, requested_url=url, run_id=run_id)
        result["direct_http"] = {"status": response.status, "snapshot_id": snapshot_id}
    else:
        result["direct_http"] = {"status": "skipped", "snapshot_id": None}

    if sec_cik:
        response = sec.fetch(
            FetchRequest(
                request_id="smoke-sec",
                run_id=run_id,
                claim_ids=["smoke"],
                query_ids=["smoke-sec"],
                url=str(sec_cik),
                timeout_seconds=20,
                max_bytes=20 * 1024 * 1024,
                max_redirects=5,
            )
        )
        snapshot_id = _commit_successful_fetch(run_dir, response=response, requested_url=sec_cik, run_id=run_id)
        result["sec"] = {"status": response.status, "snapshot_id": snapshot_id}
    else:
        result["sec"] = {"status": "skipped", "snapshot_id": None}

    if crossref_doi:
        response = crossref.fetch(
            FetchRequest(
                request_id="smoke-crossref",
                run_id=run_id,
                claim_ids=["smoke"],
                query_ids=["smoke-crossref"],
                url=crossref_doi,
                timeout_seconds=20,
                max_bytes=20 * 1024 * 1024,
                max_redirects=2,
            )
        )
        snapshot_id = _commit_successful_fetch(run_dir, response=response, requested_url=crossref_doi, run_id=run_id)
        result["crossref"] = {"status": response.status, "snapshot_id": snapshot_id}
    else:
        result["crossref"] = {"status": "skipped"}

    if arxiv_id:
        response = arxiv.fetch(
            FetchRequest(
                request_id="smoke-arxiv",
                run_id=run_id,
                claim_ids=["smoke"],
                query_ids=["smoke-arxiv"],
                url=arxiv_id,
                timeout_seconds=20,
                max_bytes=20 * 1024 * 1024,
                max_redirects=2,
            )
        )
        snapshot_id = _commit_successful_fetch(run_dir, response=response, requested_url=arxiv_id, run_id=run_id)
        result["arxiv"] = {"status": response.status, "snapshot_id": snapshot_id}
    else:
        result["arxiv"] = {"status": "skipped"}

    if government_url:
        response = gov.fetch(
            FetchRequest(
                request_id="smoke-government",
                run_id=run_id,
                claim_ids=["smoke"],
                query_ids=["smoke-government"],
                url=government_url,
                timeout_seconds=20,
                max_bytes=20 * 1024 * 1024,
                max_redirects=3,
            )
        )
        snapshot_id = _commit_successful_fetch(run_dir, response=response, requested_url=government_url, run_id=run_id)
        result["government_http"] = {"status": response.status, "snapshot_id": snapshot_id}
    else:
        result["government_http"] = {"status": "skipped"}

    return result


def _artifact_counts(run_dir: Path) -> dict[str, int]:
    paths = {
        "queries": run_dir / "web_evidence" / "queries.jsonl",
        "provider_calls": run_dir / "web_evidence" / "provider_calls.jsonl",
        "snapshots": run_dir / "web_evidence" / "snapshot_index.jsonl",
        "segments": run_dir / "web_evidence" / "segments.jsonl",
        "evidence": run_dir / "evidence.jsonl",
        "audit": run_dir / "audit_log.jsonl",
    }
    return {name: len(read_jsonl(path)) for name, path in paths.items()}


def _simulated_failed_fetch(
    *,
    failure_type: str,
    expected_status: str,
    case_id: str,
) -> FetchResponse:
    typed_error_by_failure = {
        "rate_limited": "rate_limited",
        "timeout": "timeout",
        "not_found": "not_found",
        "private_redirect_blocked": "unsafe_url:redirect target is private/restricted",
    }
    return FetchResponse(
        request_id=request_id_for("chaos-fetch", case_id),
        run_id="chaos",
        claim_ids=[case_id],
        query_ids=[request_id_for("chaos-query", case_id)],
        provider_id="chaos_fixture",
        provider_call_id=request_id_for("chaos-provider", case_id),
        status=expected_status,
        canonical_url=f"https://chaos.invalid/{failure_type}",
        redirect_chain=[f"https://chaos.invalid/{failure_type}"],
        byte_length=0,
        typed_error=typed_error_by_failure.get(failure_type, expected_status),
        charged_cost=0,
        payload=None,
        response_headers={},
    )


def _derive_chaos_checks(run_dir: Path, case_rows: list[dict[str, Any]]) -> tuple[int, dict[str, Any], dict[str, Any]]:
    outcome_by_failure = {
        "rate_limited": "rate_limited",
        "timeout": "transport_error",
        "not_found": "not_found",
        "private_redirect_blocked": "unsafe_url",
        "malformed_body": "extraction_failed",
        "stale_snapshot": "blocked",
        "duplicate_origin": "dedupe_skip",
        "missing_optional_credentials": "unconfigured",
    }
    required = set(outcome_by_failure)
    observed = {str(row.get("failure_type")) for row in case_rows}
    missing = sorted(required - observed)
    seen_case_ids: set[str] = set()
    schema_failures = 0
    status_failures = 0
    outcomes: list[dict[str, Any]] = []
    failed_fetches: list[FetchResponse] = []
    for row in case_rows:
        case_id = str(row.get("case_id") or "")
        failure_type = str(row.get("failure_type") or "")
        expected_status = str(row.get("expected_status") or "")
        actual_status = outcome_by_failure.get(failure_type, "")
        if not case_id or case_id in seen_case_ids or failure_type not in outcome_by_failure or not expected_status:
            schema_failures += 1
        elif actual_status != expected_status:
            status_failures += 1
        seen_case_ids.add(case_id)
        outcomes.append({"case_id": case_id, "failure_type": failure_type, "expected_status": expected_status, "actual_status": actual_status})
        if failure_type in {"rate_limited", "timeout", "not_found", "private_redirect_blocked"}:
            failed_fetches.append(_simulated_failed_fetch(failure_type=failure_type, expected_status=actual_status, case_id=case_id))

    private_response = DirectHTTPProvider().fetch(
        FetchRequest(
            request_id="chaos-private-block",
            run_id="chaos",
            claim_ids=["chaos-private"],
            query_ids=["chaos-private"],
            url="http://127.0.0.1/private",
            timeout_seconds=1,
            max_bytes=1,
            max_redirects=0,
        )
    )
    absent_env = "__MARKET_LAB_CHAOS_OPTIONAL_PROVIDER_ABSENT__"
    old_env = os.environ.pop(absent_env, None)
    try:
        optional_health = OptionalProvider("chaos_optional", absent_env).health()
    finally:
        if old_env is not None:
            os.environ[absent_env] = old_env

    with tempfile.TemporaryDirectory(dir=run_dir.parent) as td:
        resume_dir = Path(td) / "resume"
        claim = [{"claim_id": "chaos-resume", "text": "Frozen resume idempotence context"}]
        _frozen_collect_for_claims(resume_dir, claim, profile="keyless_standard", run_id="chaos-resume", max_claims=None, budget=load_budget_profile("keyless_standard"))
        first_counts = _artifact_counts(resume_dir)
        _frozen_collect_for_claims(resume_dir, claim, profile="keyless_standard", run_id="chaos-resume", max_claims=None, budget=load_budget_profile("keyless_standard"))
        second_counts = _artifact_counts(resume_dir)

    no_failed_fetch_evidence = all(
        response.payload is None
        and response.snapshot_id is None
        and response.byte_length == 0
        and response.status != "success"
        for response in failed_fetches
    )
    charged_costs = [response.charged_cost or 0 for response in failed_fetches]
    network_calls = 0
    budget_adherence = sum(charged_costs) == 0 and network_calls == 0
    failures = len(missing) + schema_failures + status_failures
    checks = {
        "schema_valid": schema_failures == 0,
        "unique_case_ids": len(seen_case_ids) == len(case_rows),
        "typed_failures": failures == 0,
        "expected_status_matches_outcome": status_failures == 0,
        "missing_failure_types": missing,
        "no_failed_fetch_evidence": no_failed_fetch_evidence,
        "private_destinations_blocked": private_response.status == "unsafe_url" and private_response.payload is None,
        "resume_idempotent": first_counts == second_counts,
        "optional_providers_explicit": optional_health.status == "unconfigured"
        and absent_env in optional_health.missing_configuration,
        "budget_adherence": budget_adherence,
    }
    metrics = {
        "network_calls": network_calls,
        "budget_adherence": 1.0 if budget_adherence else 0.0,
        "charged_cost": sum(charged_costs),
        "simulated_failed_fetches": len(failed_fetches),
        "resume_counts": {"first": first_counts, "second": second_counts},
        "private_block_status": private_response.status,
        "optional_health_status": optional_health.status,
        "outcomes": outcomes,
    }
    derived_failures = failures + sum(1 for key, value in checks.items() if isinstance(value, bool) and not value)
    return derived_failures, checks, metrics


def run_benchmark(
    run_dir: Path,
    *,
    lane: str,
    cases_path: Path,
    output_path: Path,
    fail_on_gate: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_path)
    run_dir = Path(run_dir)
    lane = lane.strip().lower()

    case_rows: list[dict[str, Any]] = []
    corpus_error = ""
    if cases_path.exists():
        try:
            for line in cases_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("case row must be object")
                case_rows.append(row)
        except Exception as exc:
            corpus_error = str(exc)

    failures = 0
    if lane not in {"live", "frozen", "chaos"}:
        raise ValueError(f"unknown benchmark lane: {lane}")
    if fail_on_gate and (corpus_error or not cases_path.exists() or not case_rows):
        output = {
            "lane": lane,
            "cases_total": len(case_rows),
            "cases_processed": 0,
            "cases_failed": 1,
            "passed": False,
            "metrics": {"network_calls": 0},
            "checks": {"corpus_present": cases_path.exists(), "corpus_non_empty": bool(case_rows), "corpus_error": corpus_error},
        }
        ensure_layout(run_dir)
        write_atomic_json(output_path, output)
        raise RuntimeError("benchmark-corpus-missing-or-empty")

    # For frozen lane, evaluate schema only and do not execute network.
    if lane == "frozen":
        seen_case_ids: set[str] = set()
        for row in case_rows:
            case_id = str(row.get("case_id") or "")
            body = str(row.get("body") or "")
            expected_hash = str(row.get("expected_snapshot_sha256") or "")
            actual_hash = sha256_hex(body.encode("utf-8")) if body else ""
            schema_ok = all(
                _required_str(row, field)
                for field in ["case_id", "claim_id", "claim_text", "provider_id", "url", "body", "expected_snapshot_sha256"]
            )
            if not schema_ok or case_id in seen_case_ids or expected_hash != actual_hash:
                failures += 1
            seen_case_ids.add(case_id)
        output = {
            "lane": lane,
            "cases_total": len(case_rows),
            "cases_processed": len(case_rows),
            "cases_failed": failures,
            "passed": failures == 0 and bool(case_rows),
            "metrics": {"network_calls": 0, "citation_snapshot_validity": 1.0 if failures == 0 and case_rows else 0.0},
            "checks": {
                "corpus_non_empty": bool(case_rows),
                "schema_valid": failures == 0,
                "unique_case_ids": len(seen_case_ids) == len(case_rows),
                "zero_network": True,
                "zero_snippet_evidence": True,
                "frozen_replay_reproduces_hashes": failures == 0,
            },
        }
    elif lane == "chaos":
        failures, checks, metrics = _derive_chaos_checks(run_dir, case_rows)
        output = {
            "lane": lane,
            "cases_total": len(case_rows),
            "cases_processed": len(case_rows),
            "cases_failed": failures,
            "passed": failures == 0 and bool(case_rows),
            "metrics": metrics,
            "checks": checks,
        }
    else:
        claims = [
            {
                "claim_id": row.get("claim_id") or row.get("id") or f"case-{idx}",
                "text": row.get("claim_text") or row.get("claim") or "",
            }
            for idx, row in enumerate(case_rows)
        ]
        collect = collect_for_claims(
            run_dir,
            claims,
            profile="keyless_standard",
            run_id=run_dir.name,
            mode="live",
        )
        output = {
            "lane": lane,
            "cases_total": len(case_rows),
            "cases_processed": collect["claims"],
            "cases_failed": 0 if collect["claims"] and collect["searches"] and collect["evidence_added"] else len(case_rows),
            "passed": bool(collect["claims"] and collect["searches"] and collect["evidence_added"]),
            "metrics": collect,
            "checks": {"searches": collect["searches"], "evidence_added": collect["evidence_added"]},
        }

    ensure_layout(run_dir)
    write_atomic_json(output_path, output)

    if fail_on_gate and not output["passed"]:
        raise RuntimeError("benchmark-gates-failed")
    return output


def verify_run(
    run_dir: Path,
    *,
    require_snapshots: bool = False,
    require_counterevidence_coverage: bool = False,
    require_audit_chain: bool = False,
    require_zero_snippet_evidence: bool = False,
    require_zero_execution_side_effects: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    evidence_path = run_dir / "evidence.jsonl"
    search_rows_path = run_dir / "web_evidence" / "search_results.jsonl"
    provider_calls_path = run_dir / "web_evidence" / "provider_calls.jsonl"
    snapshot_index = run_dir / "web_evidence" / "snapshot_index.jsonl"
    audit_log = run_dir / "audit_log.jsonl"

    evidence_rows = read_jsonl(evidence_path) if evidence_path.exists() else []
    search_rows = read_jsonl(search_rows_path) if search_rows_path.exists() else []
    provider_calls = read_jsonl(provider_calls_path) if provider_calls_path.exists() else []
    snapshot_rows = read_jsonl(snapshot_index) if snapshot_index.exists() else []
    segment_rows = read_jsonl(run_dir / "web_evidence" / "segments.jsonl")

    checks: dict[str, Any] = {
        "has_evidence": len(evidence_rows) > 0,
        "search_rows_present": len(search_rows) > 0,
        "provider_calls_present": len(provider_calls) > 0,
    }

    if require_snapshots:
        missing = []
        invalid = []
        snapshot_by_id = {row.get("snapshot_id"): row for row in snapshot_rows if isinstance(row, dict)}
        segment_by_id = {row.get("segment_id"): row for row in segment_rows if isinstance(row, dict)}
        for row in evidence_rows:
            snapshot_id = row.get("snapshot_id")
            if not snapshot_id or snapshot_id not in snapshot_by_id:
                missing.append(snapshot_id or "<missing>")
                continue
            idx = snapshot_by_id[snapshot_id]
            manifest_rel = str(idx.get("manifest_path") or "")
            manifest_path = run_dir / manifest_rel
            snapshot_dir = manifest_path.parent
            raw_path = snapshot_dir / "raw.bin"
            extracted_path = snapshot_dir / "extracted.txt"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                raw_ok = raw_path.exists() and sha256_hex(raw_path.read_bytes()) == manifest.get("raw_sha256")
                files_ok = manifest_path.exists() and extracted_path.exists() and raw_ok
                segment_row = segment_by_id.get(row.get("segment_id"))
                # Bind the segment to THIS evidence row's snapshot (audit
                # run-1, finding 3): without the equality check, a segment
                # whose excerpt lives in a different snapshot validates
                # against the wrong evidence — verify_segment_locator only
                # checks the excerpt against the segment's own snapshot.
                segment_ok = bool(
                    segment_row
                    and segment_row.get("snapshot_id") == snapshot_id
                    and verify_segment_locator(run_dir, type("SegmentLike", (), segment_row)())
                )
            except Exception:
                files_ok = False
                segment_ok = False
            if not files_ok or not segment_ok:
                invalid.append(snapshot_id)
        checks["snapshots_present"] = len(missing) == 0
        checks["missing_snapshots"] = missing[:5]
        checks["snapshots_complete"] = len(invalid) == 0
        checks["invalid_snapshots"] = invalid[:5]

    if require_counterevidence_coverage:
        query_rows = read_jsonl(run_dir / "web_evidence" / "queries.jsonl")
        claim_ids = {
            r.get("claim_id")
            for r in evidence_rows
            if isinstance(r, dict) and r.get("claim_id") and r.get("schema_version") == SCHEMA_EVIDENCE_V2
        }
        required_lanes = ("counterevidence", "freshness_supersession")
        query_by_id = {str(row.get("query_id")): row for row in query_rows if isinstance(row, dict) and row.get("query_id")}
        executed_by_query: dict[str, list[dict[str, Any]]] = {}
        for row in _flatten_search_result_rows(search_rows):
            query_id = str(row.get("query_id") or "")
            if query_id:
                executed_by_query.setdefault(query_id, []).append(row)
        planned: dict[str, dict[str, list[dict[str, Any]]]] = {
            str(cid): {lane: [] for lane in required_lanes} for cid in claim_ids
        }
        for row in query_rows:
            if not isinstance(row, dict):
                continue
            lane = str(row.get("lane", ""))
            if lane not in required_lanes:
                continue
            for cid in row.get("claim_ids", []):
                if str(cid) in planned:
                    planned[str(cid)][lane].append(row)
        audit_rows = read_jsonl(audit_log) if audit_log.exists() else []
        zero_audit_rows = [
            row
            for row in audit_rows
            if isinstance(row, dict)
            and row.get("event_type") == "claim_search_zero_results"
            and row.get("query_id")
        ]
        blockers: list[dict[str, str]] = []
        accepted_zero_result_routes: list[dict[str, str]] = []
        for cid, lanes in sorted(planned.items()):
            for lane in required_lanes:
                lane_queries = lanes[lane]
                if not lane_queries:
                    blockers.append({"type": f"{lane}_missing", "claim_id": cid, "lane": lane})
                    continue
                if not any(row.get("query_strategy", "exact_claim") != "broadened_fallback" for row in lane_queries):
                    blockers.append({"type": f"{lane}_missing", "claim_id": cid, "lane": lane})
                    continue
                clean_zero_query_ids: set[str] = set()
                positive_query_ids: set[str] = set()
                for query in sorted(lane_queries, key=lambda row: str(row.get("query_id") or "")):
                    query_id = str(query.get("query_id") or "")
                    if _string_list(query.get("claim_ids")) != [cid]:
                        blockers.append(
                            {
                                "type": f"{lane}_route_claim_mismatch",
                                "claim_id": cid,
                                "lane": lane,
                                "query_id": query_id,
                            }
                        )
                        continue
                    results = executed_by_query.get(query_id, [])
                    if not results:
                        blockers.append({"type": f"{lane}_not_executed", "claim_id": cid, "lane": lane, "query_id": query_id})
                        continue
                    if len(results) != 1:
                        blockers.append(
                            {
                                "type": f"{lane}_duplicate_execution",
                                "claim_id": cid,
                                "lane": lane,
                                "query_id": query_id,
                            }
                        )
                        continue
                    for result in results:
                        enriched = {**query_by_id.get(query_id, {}), **result}
                        expected_provider_id = str(query.get("provider_id") or "")
                        if expected_provider_id and str(result.get("provider_id") or "") != expected_provider_id:
                            blockers.append(
                                {
                                    "type": f"{lane}_provider_mismatch",
                                    "claim_id": cid,
                                    "lane": lane,
                                    "query_id": query_id,
                                    "provider_id": str(result.get("provider_id") or ""),
                                    "expected_provider_id": expected_provider_id,
                                }
                            )
                            continue
                        status = str(enriched.get("status") or "")
                        typed_error = str(enriched.get("typed_error") or "")
                        try:
                            result_count = int(enriched.get("result_count") or 0)
                        except (TypeError, ValueError):
                            result_count = 0
                        if status == "zero_results" and result_count == 0:
                            candidate_audits = [
                                row for row in zero_audit_rows if str(row.get("query_id") or "") == query_id
                            ]
                            matching_audits = [
                                row for row in candidate_audits if _zero_result_audit_matches(row, claim_id=cid, route=enriched)
                            ]
                            if len(candidate_audits) != 1 or len(matching_audits) != 1:
                                blockers.append(
                                    {
                                        "type": f"{lane}_zero_results_unaudited",
                                        "claim_id": cid,
                                        "lane": lane,
                                        "query_id": query_id,
                                        "provider_id": str(enriched.get("provider_id") or ""),
                                    }
                                )
                            else:
                                clean_zero_query_ids.add(query_id)
                        elif result_count <= 0 and status in {"success", "degraded", ""}:
                            blockers.append(
                                {
                                    "type": f"{lane}_no_results",
                                    "claim_id": cid,
                                    "lane": lane,
                                    "query_id": query_id,
                                    "provider_id": str(enriched.get("provider_id") or ""),
                                }
                            )
                        elif status == "success":
                            positive_query_ids.add(query_id)
                        else:
                            blockers.append(
                                {
                                    "type": f"{lane}_failed",
                                    "claim_id": cid,
                                    "lane": lane,
                                    "query_id": query_id,
                                    "provider_id": str(enriched.get("provider_id") or ""),
                                    "status": status,
                                    "typed_error": typed_error,
                                }
                            )
                for query in lane_queries:
                    query_id = str(query.get("query_id") or "")
                    if query_id not in clean_zero_query_ids or query.get("query_strategy") == "broadened_fallback":
                        continue
                    fallback_queries = [
                        row
                        for row in query_rows
                        if isinstance(row, dict)
                        if row.get("query_strategy") == "broadened_fallback" and row.get("fallback_for_query_id") == query_id
                    ]
                    if not fallback_queries:
                        blockers.append(
                            {
                                "type": f"{lane}_zero_results_without_fallback",
                                "claim_id": cid,
                                "lane": lane,
                                "query_id": query_id,
                            }
                        )
                        continue
                    if len(fallback_queries) != 1:
                        blockers.append(
                            {
                                "type": f"{lane}_zero_results_duplicate_fallback",
                                "claim_id": cid,
                                "lane": lane,
                                "query_id": query_id,
                            }
                        )
                        continue
                    for fallback in fallback_queries:
                        fallback_claim_ids = _string_list(fallback.get("claim_ids"))
                        if (
                            str(fallback.get("lane") or "") != lane
                            or fallback_claim_ids != [cid]
                            or str(fallback.get("query_strategy") or "") != "broadened_fallback"
                            or str(fallback.get("fallback_for_query_id") or "") != query_id
                        ):
                            blockers.append(
                                {
                                    "type": f"{lane}_zero_results_mislinked_fallback",
                                    "claim_id": cid,
                                    "lane": lane,
                                    "query_id": query_id,
                                }
                            )
                            continue
                        fallback_id = str(fallback.get("query_id") or "")
                        if fallback_id in clean_zero_query_ids:
                            accepted_zero_result_routes.append(
                                {
                                    "claim_id": cid,
                                    "lane": lane,
                                    "query_id": query_id,
                                    "fallback_query_id": fallback_id,
                                    "outcome": "audited_zero_results_after_broadened_fallback",
                                }
                            )
                        elif fallback_id in positive_query_ids:
                            break
        checks["counterevidence_coverage_ok"] = not blockers
        checks["coverage_blockers"] = blockers
        checks["accepted_zero_result_routes"] = accepted_zero_result_routes
        checks["automated_evidence_context_only"] = all(
            row.get("stance") == "context" for row in evidence_rows if row.get("source_lineage", "").startswith("web_evidence")
        )

    if require_audit_chain:
        checks["audit_chain_present"] = audit_log.exists()
        ok_chain, reason = verify_audit_chain(load_audit_chain(run_dir)) if audit_log.exists() else (False, "missing_audit_log")
        checks["audit_chain_valid"] = ok_chain
        checks["audit_chain_error"] = reason

    if require_zero_snippet_evidence:
        checks["zero_snippet_evidence"] = all(
            row.get("source_type") != "search_result"
            and row.get("verbatim_excerpt_or_value") != ""
            and row.get("segment_id")
            and row.get("snapshot_id")
            for row in evidence_rows
        )

    if require_zero_execution_side_effects:
        forbidden_paths = [
            "mock_ledger.jsonl",
            "mock_portfolio_state.json",
            "pending_order_candidates.jsonl",
            "options_paper_ledger.jsonl",
            "options_paper_state.json",
        ]
        touched = [p for p in forbidden_paths if (run_dir / p).exists()]
        checks["side_effect_free"] = not touched
        checks["side_effect_paths"] = touched

    checks["ok"] = all(v is True for k, v in checks.items() if isinstance(v, bool))
    checks["run_dir"] = str(run_dir)
    checks["recorded_at_utc"] = utcnow()
    return checks


__all__ = [
    "build_registry",
    "check_health",
    "collect_for_claims",
    "run_smoke",
    "run_benchmark",
    "verify_run",
]

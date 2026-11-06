from __future__ import annotations

"""Provider implementations for web-evidence acquisition."""

import hashlib
import ipaddress
import os
import re
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.request import Request, urlopen

from .web_evidence import (
    CostEstimate,
    DomainLookupRequest,
    DomainLookupResponse,
    FetchRequest,
    FetchResponse,
    ProviderHealth,
    ProviderSpec,
    SearchHit,
    SearchRequest,
    SearchResponse,
    PROVIDER_FAMILY_DIRECT_FETCH,
    PROVIDER_FAMILY_DOMAIN_API,
    PROVIDER_FAMILY_WEB_INDEX,
    request_id_for,
    utcnow,
    classify_url_safety,
    validate_fetch_url,
)

_USER_AGENT = "OzLabs Market Lab/1.0 (contact@theozhq.com)"


def sha1_hex(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()


def read_status_json(payload: bytes) -> dict[str, Any] | None:
    try:
        import json

        return json.loads(payload.decode("utf-8", errors="replace"))
    except Exception:
        return None


def _safe_headers() -> dict[str, str]:
    return {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


def _endpoint_health(provider_id: str, capabilities: list[str], endpoint: str) -> ProviderHealth:
    started = time.time()
    ok, _, reason = classify_url_safety(endpoint, enforce_dns=True)
    return ProviderHealth(
        provider_id=provider_id,
        timestamp=utcnow(),
        status="ready" if ok else "unavailable",
        capabilities_ready=capabilities if ok else [],
        reason_code="" if ok else "dns_validation_failed",
        safe_message="ready" if ok else reason,
        latency_ms=int((time.time() - started) * 1000),
        observed_endpoint=endpoint,
    )


def _restricted_ip(ip_txt: str) -> bool:
    ip = ipaddress.ip_address(ip_txt)
    return any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_reserved,
            ip.is_multicast,
            ip.is_unspecified,
            ip.is_global is False,
        )
    )


def _resolve_public(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise RuntimeError(f"unsafe_url:dns resolution failed: {exc}") from exc
    ips: list[str] = []
    for _, _, _, _, sockaddr in infos:
        ip = sockaddr[0]
        try:
            if _restricted_ip(ip):
                raise RuntimeError("unsafe_url:resolved destination is private/restricted")
        except ValueError as exc:
            raise RuntimeError(f"unsafe_url:invalid resolved address: {ip}") from exc
        if ip not in ips:
            ips.append(ip)
    if not ips:
        raise RuntimeError("unsafe_url:dns resolution produced no usable addresses")
    return ips


_MAX_RESPONSE_HEADERS = 100
_MAX_HEADER_BYTES = 64 * 1024


def _read_http_response(sock, *, max_bytes: int) -> tuple[int, bytes, dict[str, str]]:
    fp = sock.makefile("rb")
    status_line = fp.readline(65536).decode("iso-8859-1", errors="replace").strip()
    if not status_line.startswith("HTTP/"):
        raise RuntimeError("transport_error:invalid_http_response")
    parts = status_line.split(" ", 2)
    status = int(parts[1])
    headers: dict[str, str] = {}
    header_bytes = 0
    while True:
        line = fp.readline(65536)
        if line in {b"\r\n", b"\n", b""}:
            break
        # Header budget (audit run-1, finding 4): the body had a byte cap
        # but headers accumulated without bound — a drip-feeding origin
        # could grow memory indefinitely under per-operation timeouts.
        header_bytes += len(line)
        if len(headers) >= _MAX_RESPONSE_HEADERS or header_bytes > _MAX_HEADER_BYTES:
            raise RuntimeError("too_large:response header limit exceeded")
        text = line.decode("iso-8859-1", errors="replace")
        if ":" in text:
            k, v = text.split(":", 1)
            headers[k.lower().strip()] = v.strip()
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = fp.read(min(64 * 1024, max_bytes + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise RuntimeError("too_large:response byte limit exceeded")
    return status, b"".join(chunks), headers


def _single_pinned_fetch(url: str, *, timeout_seconds: int, max_bytes: int, deadline: float | None = None) -> tuple[int, bytes, dict[str, str]]:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    ips = _resolve_public(host, port)
    target = parsed.path or "/"
    if parsed.query:
        target += f"?{parsed.query}"
    last_error: Exception | None = None
    for ip in ips:
        # Aggregate deadline (audit run-1, finding 4): per-operation
        # timeouts alone multiply across IPs and redirect hops, so each
        # attempt gets only the time actually left in the caller's budget.
        effective_timeout = timeout_seconds
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError("transport_error:aggregate deadline exceeded")
            effective_timeout = min(timeout_seconds, remaining)
        try:
            raw = socket.create_connection((ip, port), timeout=effective_timeout)
            raw.settimeout(effective_timeout)
            sock = raw
            if parsed.scheme == "https":
                ctx = ssl.create_default_context()
                sock = ctx.wrap_socket(raw, server_hostname=host)
            try:
                headers = {
                    "Host": host if parsed.port is None else f"{host}:{port}",
                    "User-Agent": _USER_AGENT,
                    "Accept": _safe_headers()["Accept"],
                    "Connection": "close",
                }
                request = f"GET {target} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
                sock.sendall(request.encode("ascii"))
                return _read_http_response(sock, max_bytes=max_bytes)
            finally:
                sock.close()
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"transport_error:{last_error}")


def _http_fetch(
    url: str,
    *,
    timeout_seconds: int = 45,
    max_bytes: int = 20 * 1024 * 1024,
    max_redirects: int = 5,
) -> tuple[int, bytes, dict[str, str], str, list[str]]:
    ok, validated, reason = classify_url_safety(url)
    if not ok:
        raise RuntimeError(f"unsafe_url:{reason}")
    chain = [validated]

    current = validated
    start = time.time()
    deadline = start + timeout_seconds
    redirects_left = max_redirects
    try:
        while True:
            if time.time() >= deadline:
                raise RuntimeError("transport_error:aggregate deadline exceeded")
            status, payload, headers = _single_pinned_fetch(current, timeout_seconds=timeout_seconds, max_bytes=max_bytes, deadline=deadline)
            if status in {301, 302, 303, 307, 308}:
                location = headers.get("location", "")
                if not location:
                    raise RuntimeError(f"transport_error:redirect-without-location:{status}")
                if redirects_left <= 0:
                    raise RuntimeError("transport_error:too_many_redirects")
                redirects_left -= 1
                ok, next_url, reason = classify_url_safety(urljoin(current, location))
                if not ok:
                    raise RuntimeError(f"unsafe_url:{reason}")
                current = next_url
                chain.append(current)
                continue
            if status == 429:
                raise RuntimeError("rate_limited")
            if status in {401, 403}:
                raise RuntimeError("auth_required")
            if status == 404:
                raise RuntimeError("not_found")
            if status >= 400:
                raise RuntimeError(f"transport_error:http_error_{status}")
            return status, payload, headers, current, chain
    finally:
        _ = int((time.time() - start) * 1000)


@dataclass
class ProviderBase:
    provider_id: str
    provider_family: str
    capabilities: list[str]
    implementation_version: str
    configuration_fields: list[str]
    requires_managed_key: bool = False
    supports_keyless: bool = True
    cost_model: str = "zero-per-request"
    retention_note: str = "run-local immutable artifacts"
    rate_policy: str = "default"

    def spec(self) -> ProviderSpec:
        return ProviderSpec(
            provider_id=self.provider_id,
            provider_family=self.provider_family,
            implementation_version=self.implementation_version,
            capabilities=self.capabilities,
            configuration_fields=self.configuration_fields,
            requires_managed_key=self.requires_managed_key,
            supports_keyless=self.supports_keyless,
            cost_model=self.cost_model,
            retention_note=self.retention_note,
            regions=["global"],
            self_hosted=False,
            rate_policy=self.rate_policy,
        )

    def estimate_cost(self, request: SearchRequest | FetchRequest | DomainLookupRequest) -> CostEstimate:
        return CostEstimate(provider_id=self.provider_id, currency="USD", micros=0, requires_hard_budget=False)


class DirectHTTPProvider(ProviderBase):
    """Hardened known-URL fetcher."""

    def __init__(self) -> None:
        super().__init__(
            provider_id="direct_http",
            provider_family=PROVIDER_FAMILY_DIRECT_FETCH,
            capabilities=["fetch"],
            implementation_version="1.0.0",
            configuration_fields=["USER_AGENT"],
        )

    def health(self, context: dict[str, Any] | None = None) -> ProviderHealth:
        started = time.time()
        ok, _, reason = classify_url_safety("https://example.com/", enforce_dns=True)
        return ProviderHealth(
            provider_id=self.provider_id,
            timestamp=utcnow(),
            status="ready" if ok else "unavailable",
            capabilities_ready=["fetch"] if ok else [],
            missing_configuration=[],
            reason_code="" if ok else "dns_validation_failed",
            safe_message="ready" if ok else reason,
            latency_ms=int((time.time() - started) * 1000),
            observed_endpoint="https://",
        )

    def fetch(self, request: FetchRequest, context: dict[str, Any] | None = None) -> FetchResponse:
        started = time.time()
        call_id = request_id_for(self.provider_id, {"request_id": request.request_id, "url": request.url})
        try:
            status, payload, headers, canonical, chain = _http_fetch(
                request.url,
                timeout_seconds=request.timeout_seconds,
                max_bytes=request.max_bytes,
                max_redirects=request.max_redirects,
            )
            return FetchResponse(
                request_id=request.request_id,
                run_id=request.run_id,
                claim_ids=request.claim_ids,
                query_ids=request.query_ids,
                provider_id=self.provider_id,
                provider_call_id=call_id,
                status="success" if status == 200 else "transport_error",
                canonical_url=canonical,
                redirect_chain=chain,
                content_type=str(headers.get("content-type", "")),
                byte_length=len(payload),
                retry_after_utc=None,
                typed_error="" if status == 200 else f"http_{status}",
                latency_ms=int((time.time() - started) * 1000),
                charged_cost=0,
                payload=payload,
                response_headers={k: str(v) for k, v in headers.items()},
            )
        except (RuntimeError, ValueError) as exc:
            reason = str(exc)
            status = "transport_error"
            if reason.startswith("unsafe_url:"):
                status = "unsafe_url"
            elif reason == "rate_limited":
                status = "rate_limited"
            elif reason == "auth_required":
                status = "auth_required"
            elif reason == "not_found":
                status = "not_found"
            elif reason.startswith("too_large:") or reason == "response-too-large":
                status = "too_large"
            return FetchResponse(
                request_id=request.request_id,
                run_id=request.run_id,
                claim_ids=request.claim_ids,
                query_ids=request.query_ids,
                provider_id=self.provider_id,
                provider_call_id=call_id,
                status=status,
                canonical_url=request.url,
                redirect_chain=[request.url],
                byte_length=0,
                typed_error=reason,
                latency_ms=int((time.time() - started) * 1000),
                charged_cost=0,
                payload=None,
                response_headers={},
            )


class DDGSProvider(ProviderBase):
    def __init__(self) -> None:
        super().__init__(
            provider_id="ddgs",
            provider_family=PROVIDER_FAMILY_WEB_INDEX,
            capabilities=["search"],
            implementation_version="1.0.0",
            configuration_fields=[],
        )

    def health(self, context: dict[str, Any] | None = None) -> ProviderHealth:
        started = time.time()
        try:
            from ddgs import DDGS

            with DDGS() as client:
                rows = list(client.text("OpenAI", max_results=1, region="us-en", safesearch="moderate"))
            ok = len(rows) > 0
            return ProviderHealth(
                provider_id=self.provider_id,
                timestamp=utcnow(),
                status="ready" if ok else "degraded",
                capabilities_ready=["search"] if ok else [],
                missing_configuration=[],
                reason_code="" if ok else "zero_results_probe",
                safe_message="ready" if ok else "probe returned zero results",
                latency_ms=int((time.time() - started) * 1000),
                observed_endpoint="ddgs.text",
            )
        except ImportError:
            return ProviderHealth(
                provider_id=self.provider_id,
                timestamp=utcnow(),
                status="unconfigured",
                capabilities_ready=[],
                missing_configuration=["ddgs"],
                reason_code="missing_dependency",
                safe_message="ddgs package is not installed",
                latency_ms=int((time.time() - started) * 1000),
                observed_endpoint="ddgs.text",
            )
        except Exception as exc:
            return ProviderHealth(
                provider_id=self.provider_id,
                timestamp=utcnow(),
                status="unavailable",
                capabilities_ready=[],
                missing_configuration=[],
                reason_code="probe_failed",
                safe_message=str(exc)[:240],
                latency_ms=int((time.time() - started) * 1000),
                observed_endpoint="ddgs.text",
            )

    def search(self, request: SearchRequest, context: dict[str, Any] | None = None) -> SearchResponse:
        started = time.time()
        try:
            from ddgs import DDGS
            from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

            hits: list[SearchHit] = []
            raw_rows: list[dict[str, Any]] = []
            malformed_rows = 0
            with DDGS(timeout=request.timeout_seconds) as client:
                for row in client.text(
                    request.exact_query,
                    max_results=request.max_results,
                    region="us-en" if request.country.upper() == "US" else "wt-wt",
                    safesearch="moderate",
                ):
                    if not isinstance(row, Mapping):
                        malformed_rows += 1
                        raw_rows.append({"malformed_row_type": type(row).__name__})
                        continue
                    raw = dict(row)
                    raw_rows.append(raw)
                    href = str(raw.get("href") or raw.get("url") or "").strip()
                    title = re.sub(r"\s+", " ", str(raw.get("title") or "")).strip()
                    snippet = re.sub(r"\s+", " ", str(raw.get("body") or "")).strip()
                    if not href:
                        malformed_rows += 1
                        continue
                    idx = len(hits) + 1
                    if idx > request.max_results:
                        break
                    hits.append(
                        SearchHit(
                            provider_id=self.provider_id,
                            provider_result_id=f"ddgs:{sha1_hex(href)[:16]}",
                            rank=idx,
                            url=href,
                            title=title[:180],
                            snippet=snippet[:500],
                            published_hint="",
                            raw_result_hash=sha1_hex(str(sorted(raw.items()))),
                            discovered_at=utcnow(),
                        )
                    )
            raw_payload = str(raw_rows)
            if malformed_rows:
                hits = []
                status = "transport_error"
                typed_error = "malformed_results"
            else:
                status = "success" if hits else "zero_results"
                typed_error = "" if hits else "zero_results"
            return SearchResponse(
                request_id=request.request_id,
                query_id=request.query_id,
                provider_id=self.provider_id,
                status=status,
                hits=hits,
                result_count=len(hits),
                latency_ms=int((time.time() - started) * 1000),
                typed_error=typed_error,
                raw_payload_hash=sha1_hex(raw_payload),
            )
        except ImportError as exc:
            return SearchResponse(
                request_id=request.request_id,
                query_id=request.query_id,
                provider_id=self.provider_id,
                status="unconfigured",
                hits=[],
                result_count=0,
                latency_ms=int((time.time() - started) * 1000),
                typed_error=f"missing ddgs dependency: {exc}",
            )
        except RatelimitException as exc:
            return SearchResponse(
                request_id=request.request_id,
                query_id=request.query_id,
                provider_id=self.provider_id,
                status="rate_limited",
                hits=[],
                result_count=0,
                latency_ms=int((time.time() - started) * 1000),
                typed_error=str(exc) or "rate_limited",
            )
        except TimeoutException as exc:
            return SearchResponse(
                request_id=request.request_id,
                query_id=request.query_id,
                provider_id=self.provider_id,
                status="transport_error",
                hits=[],
                result_count=0,
                latency_ms=int((time.time() - started) * 1000),
                typed_error=str(exc) or "timeout",
            )
        except DDGSException as exc:
            clean_zero_results = type(exc) is DDGSException and exc.args == ("No results found.",)
            return SearchResponse(
                request_id=request.request_id,
                query_id=request.query_id,
                provider_id=self.provider_id,
                status="zero_results" if clean_zero_results else "transport_error",
                hits=[],
                result_count=0,
                latency_ms=int((time.time() - started) * 1000),
                typed_error="zero_results" if clean_zero_results else str(exc),
            )
        except Exception as exc:
            return SearchResponse(
                request_id=request.request_id,
                query_id=request.query_id,
                provider_id=self.provider_id,
                status="transport_error",
                hits=[],
                result_count=0,
                latency_ms=int((time.time() - started) * 1000),
                typed_error=str(exc),
            )


class SECProvider(ProviderBase):
    def __init__(self) -> None:
        super().__init__(
            provider_id="sec",
            provider_family=PROVIDER_FAMILY_DOMAIN_API,
            capabilities=["lookup", "fetch"],
            implementation_version="1.0.0",
            configuration_fields=[],
        )

    def health(self, context: dict[str, Any] | None = None) -> ProviderHealth:
        return _endpoint_health(self.provider_id, ["lookup", "fetch"], "https://data.sec.gov/")

    def _sec_cik_url(self, cik: str) -> str:
        digits = re.sub(r"\D", "", cik)
        if not digits:
            raise ValueError("invalid cik")
        cik = digits.zfill(10)
        return f"https://data.sec.gov/submissions/CIK{cik}.json"

    def lookup(self, request: DomainLookupRequest, context: dict[str, Any] | None = None) -> DomainLookupResponse:
        start = time.time()
        try:
            if request.identifier_type not in {"cik", "ticker"}:
                raise ValueError("unsupported identifier_type")
            if request.identifier_type == "ticker":
                url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={quote_plus(request.identifier_value)}&owner=exclude&action=getcompany"
            else:
                url = self._sec_cik_url(request.identifier_value)
            status, payload, _, canonical, _ = _http_fetch(
                url,
                timeout_seconds=request.timeout_seconds,
                max_bytes=20 * 1024 * 1024,
                max_redirects=5,
            )
            if status != 200:
                raise RuntimeError(f"http_{status}")
            return DomainLookupResponse(
                request_id=request.request_id,
                run_id=request.run_id,
                provider_id=self.provider_id,
                identifier_type=request.identifier_type,
                identifier_value=request.identifier_value,
                status="success",
                canonical_url=canonical,
                payload=read_status_json(payload),
                typed_error="",
                latency_ms=int((time.time() - start) * 1000),
            )
        except Exception as exc:
            return DomainLookupResponse(
                request_id=request.request_id,
                run_id=request.run_id,
                provider_id=self.provider_id,
                identifier_type=request.identifier_type,
                identifier_value=request.identifier_value,
                status="unavailable",
                canonical_url=request.identifier_value,
                typed_error=str(exc),
                latency_ms=int((time.time() - start) * 1000),
            )

    def fetch(self, request: FetchRequest, context: dict[str, Any] | None = None) -> FetchResponse:
        url = request.url
        if re.fullmatch(r"\d{1,12}", request.url.strip()):
            try:
                url = self._sec_cik_url(request.url.strip())
            except Exception:
                pass
        return _fetch_json_like(self.provider_id, request, url=url)


class CrossrefProvider(ProviderBase):
    def __init__(self) -> None:
        super().__init__(
            provider_id="crossref",
            provider_family=PROVIDER_FAMILY_WEB_INDEX,
            capabilities=["search", "fetch"],
            implementation_version="1.0.0",
            configuration_fields=[],
        )

    def health(self, context: dict[str, Any] | None = None) -> ProviderHealth:
        return _endpoint_health(self.provider_id, ["search", "fetch"], "https://api.crossref.org/")

    def search(self, request: SearchRequest, context: dict[str, Any] | None = None) -> SearchResponse:
        q = request.exact_query.strip()
        doi_match = re.findall(r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+", q)
        if not doi_match:
            return SearchResponse(
                request_id=request.request_id,
                query_id=request.query_id,
                provider_id=self.provider_id,
                status="success",
                hits=[],
                result_count=0,
                latency_ms=0,
                typed_error="",
                raw_payload_hash="",
            )
        doi = doi_match[0]
        hit = SearchHit(
            provider_id=self.provider_id,
            provider_result_id=f"cr:{sha1_hex(doi)[:16]}",
            rank=1,
            url=f"https://api.crossref.org/works/{quote_plus(doi)}",
            title="Crossref work",
            snippet=doi,
            discovered_at=utcnow(),
        )
        return SearchResponse(
            request_id=request.request_id,
            query_id=request.query_id,
            provider_id=self.provider_id,
            status="success",
            hits=[hit],
            result_count=1,
            latency_ms=0,
            raw_payload_hash=sha1_hex(doi),
        )

    def fetch(self, request: FetchRequest, context: dict[str, Any] | None = None) -> FetchResponse:
        url = request.url
        if request.url.startswith("10."):
            url = f"https://api.crossref.org/works/{quote_plus(request.url)}"
        return _fetch_json_like(self.provider_id, request, url=url)


class ArxivProvider(ProviderBase):
    def __init__(self) -> None:
        super().__init__(
            provider_id="arxiv",
            provider_family=PROVIDER_FAMILY_DOMAIN_API,
            capabilities=["search", "fetch"],
            implementation_version="1.0.0",
            configuration_fields=[],
        )

    def health(self, context: dict[str, Any] | None = None) -> ProviderHealth:
        return _endpoint_health(self.provider_id, ["search", "fetch"], "https://export.arxiv.org/")

    def search(self, request: SearchRequest, context: dict[str, Any] | None = None) -> SearchResponse:
        q = request.exact_query.strip()
        matches = re.findall(r"\d{4}\.\d{4,5}(?:v\d+)?", q)
        if not matches:
            return SearchResponse(
                request_id=request.request_id,
                query_id=request.query_id,
                provider_id=self.provider_id,
                status="success",
                hits=[],
                result_count=0,
                latency_ms=0,
                raw_payload_hash="",
            )
        arxiv_id = matches[0]
        hit = SearchHit(
            provider_id=self.provider_id,
            provider_result_id=f"ax:{sha1_hex(arxiv_id)[:16]}",
            rank=1,
            url=f"https://export.arxiv.org/api/query?search_query=id:{arxiv_id}",
            title="arXiv record",
            snippet=arxiv_id,
            discovered_at=utcnow(),
        )
        return SearchResponse(
            request_id=request.request_id,
            query_id=request.query_id,
            provider_id=self.provider_id,
            status="success",
            hits=[hit],
            result_count=1,
            latency_ms=0,
            raw_payload_hash=sha1_hex(arxiv_id),
        )

    def fetch(self, request: FetchRequest, context: dict[str, Any] | None = None) -> FetchResponse:
        url = request.url
        if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", request.url.strip()):
            url = f"https://export.arxiv.org/api/query?search_query=id:{request.url.strip()}"
        return _fetch_json_like(self.provider_id, request, url=url)


class GovernmentHTTPProvider(ProviderBase):
    def __init__(self, hosts: Iterable[str] | None = None) -> None:
        allow = {"api.census.gov", "api.bls.gov", "www.data.gov"}
        if hosts:
            allow.update(h.lower() for h in hosts)
        self.hosts = sorted(allow)
        super().__init__(
            provider_id="government_http",
            provider_family=PROVIDER_FAMILY_DOMAIN_API,
            capabilities=["fetch"],
            implementation_version="1.0.0",
            configuration_fields=["government_http_hosts"],
        )

    def health(self, context: dict[str, Any] | None = None) -> ProviderHealth:
        rows = [_endpoint_health(self.provider_id, ["fetch"], f"https://{host}/") for host in self.hosts]
        ready = [row for row in rows if row.status == "ready"]
        first = ready[0] if ready else rows[0]
        return ProviderHealth(
            provider_id=self.provider_id,
            timestamp=utcnow(),
            status="ready" if ready else "unavailable",
            capabilities_ready=["fetch"] if ready else [],
            reason_code="" if ready else first.reason_code,
            safe_message="ready" if ready else first.safe_message,
            latency_ms=sum(row.latency_ms for row in rows),
            observed_endpoint=",".join(self.hosts),
        )

    def fetch(self, request: FetchRequest, context: dict[str, Any] | None = None) -> FetchResponse:
        try:
            normalized = validate_fetch_url(request.url)
            host = normalized.split("//", 1)[-1].split("/", 1)[0].lower()
            if host not in self.hosts:
                raise RuntimeError(f"official-host-not-allowed:{host}")
        except (RuntimeError, ValueError) as exc:
            return FetchResponse(
                request_id=request.request_id,
                run_id=request.run_id,
                claim_ids=request.claim_ids,
                query_ids=request.query_ids,
                provider_id=self.provider_id,
                provider_call_id=request_id_for(self.provider_id, request.url),
                status="unsafe_url",
                canonical_url=request.url,
                redirect_chain=[request.url],
                typed_error=str(exc),
                latency_ms=0,
            )
        return _fetch_json_like(self.provider_id, request, url=normalized)


class OptionalProvider(ProviderBase):
    """Template for optional managed providers."""

    def __init__(self, provider_id: str, env_var: str) -> None:
        self.env_var = env_var
        self.has_key = bool(os.environ.get(env_var))
        super().__init__(
            provider_id=provider_id,
            provider_family=PROVIDER_FAMILY_WEB_INDEX,
            capabilities=[],
            implementation_version="0.0.0",
            configuration_fields=[env_var],
            requires_managed_key=True,
            supports_keyless=False,
            cost_model="metered",
        )

    def health(self, context: dict[str, Any] | None = None) -> ProviderHealth:
        if self.has_key:
            return ProviderHealth(
                provider_id=self.provider_id,
                timestamp=utcnow(),
                status="disabled",
                capabilities_ready=[],
                reason_code="probe_not_implemented",
                safe_message="configured but disabled until a safe health probe is implemented",
                observed_endpoint=self.provider_id,
            )
        return ProviderHealth(
            provider_id=self.provider_id,
            timestamp=utcnow(),
            status="unconfigured",
            capabilities_ready=[],
            missing_configuration=[self.env_var],
            reason_code="missing-credential",
            safe_message="credential missing",
            observed_endpoint=self.provider_id,
        )

    def search(self, request: SearchRequest, context: dict[str, Any] | None = None) -> SearchResponse:
        return SearchResponse(
            request_id=request.request_id,
            query_id=request.query_id,
            provider_id=self.provider_id,
            status="unconfigured" if not self.has_key else "disabled",
            hits=[],
            result_count=0,
            latency_ms=0,
            typed_error="missing-managed-key" if not self.has_key else "probe_not_implemented",
        )

    def fetch(self, request: FetchRequest, context: dict[str, Any] | None = None) -> FetchResponse:
        if not self.has_key:
            return FetchResponse(
                request_id=request.request_id,
                run_id=request.run_id,
                claim_ids=request.claim_ids,
                query_ids=request.query_ids,
                provider_id=self.provider_id,
                provider_call_id=request_id_for(self.provider_id, request.url),
                status="unconfigured",
                canonical_url=request.url,
                redirect_chain=[request.url],
                byte_length=0,
                typed_error="missing-managed-key",
            )
        return FetchResponse(
            request_id=request.request_id,
            run_id=request.run_id,
            claim_ids=request.claim_ids,
            query_ids=request.query_ids,
            provider_id=self.provider_id,
            provider_call_id=request_id_for(self.provider_id, request.url),
            status="unsupported_media",
            canonical_url=request.url,
            redirect_chain=[request.url],
            byte_length=0,
            typed_error="optional provider intentionally disabled in v1",
        )


def _fetch_json_like(provider_id: str, request: FetchRequest, url: str) -> FetchResponse:
    start = time.time()
    call_id = request_id_for(provider_id, {"url": url})
    try:
        status, payload, headers, canonical, chain = _http_fetch(
            url,
            timeout_seconds=request.timeout_seconds,
            max_bytes=request.max_bytes,
            max_redirects=request.max_redirects,
        )
        return FetchResponse(
            request_id=request.request_id,
            run_id=request.run_id,
            claim_ids=request.claim_ids,
            query_ids=request.query_ids,
            provider_id=provider_id,
            provider_call_id=call_id,
            status="success" if status == 200 else "transport_error",
            canonical_url=canonical,
            redirect_chain=chain,
            content_type=str(headers.get("content-type", "application/json")),
            byte_length=len(payload),
            latency_ms=int((time.time() - start) * 1000),
            charged_cost=0,
            payload=payload,
            response_headers={k: str(v) for k, v in headers.items()},
        )
    except (RuntimeError, ValueError) as exc:
        reason = str(exc)
        status = "transport_error"
        if reason.startswith("unsafe_url:"):
            status = "unsafe_url"
        elif reason == "rate_limited":
            status = "rate_limited"
        elif reason == "auth_required":
            status = "auth_required"
        elif reason == "not_found":
            status = "not_found"
        elif reason.startswith("too_large:") or reason == "response-too-large":
            status = "too_large"
        return FetchResponse(
            request_id=request.request_id,
            run_id=request.run_id,
            claim_ids=request.claim_ids,
            query_ids=request.query_ids,
            provider_id=provider_id,
            provider_call_id=call_id,
            status=status,
            canonical_url=url,
            redirect_chain=[url],
            typed_error=reason,
            latency_ms=int((time.time() - start) * 1000),
            charged_cost=0,
        )


def build_keyless_registry(profile: str = "keyless_standard") -> list[Any]:
    return [
        DDGSProvider(),
        DirectHTTPProvider(),
        SECProvider(),
        CrossrefProvider(),
        ArxivProvider(),
        GovernmentHTTPProvider(),
    ]


def build_optional_registry(profile: str = "keyless_standard") -> list[Any]:
    return [
        OptionalProvider("jina_reader", "JINA_API_KEY"),
        OptionalProvider("tavily", "TAVILY_API_KEY"),
        OptionalProvider("brave", "BRAVE_API_KEY"),
        OptionalProvider("exa", "EXA_API_KEY"),
        OptionalProvider("firecrawl", "FIRECRAWL_API_KEY"),
        OptionalProvider("parallel", "PARALLEL_API_KEY"),
        OptionalProvider("searxng", "SEARXNG_BASE_URL"),
    ]


__all__ = [
    "DirectHTTPProvider",
    "DDGSProvider",
    "SECProvider",
    "CrossrefProvider",
    "ArxivProvider",
    "GovernmentHTTPProvider",
    "OptionalProvider",
    "build_keyless_registry",
    "build_optional_registry",
]

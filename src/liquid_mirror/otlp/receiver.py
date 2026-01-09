"""
OTLP Receiver - FastAPI routes for OpenTelemetry Protocol ingestion.

Endpoints:
- POST /v1/traces  - Receive trace data (OTLP/HTTP)
- POST /v1/metrics - Receive metric data (OTLP/HTTP)
- POST /v1/logs    - Receive log data (OTLP/HTTP)

All endpoints accept:
- application/json (JSON-encoded OTLP)
- application/x-protobuf (Protobuf-encoded OTLP) [not yet implemented]

Persistence:
- Traces -> enterprise.traces + enterprise.trace_spans
- Logs -> enterprise.structured_logs
- Metrics -> enterprise.request_metrics / enterprise.llm_call_metrics
"""

import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List

from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse
from psycopg2.extras import RealDictCursor

from .mapper import OTelMapper, MappedTrace, MappedSpan, MappedLog, MappedMetric

logger = logging.getLogger(__name__)

# Thread pool for sync DB operations (async routes, sync pool)
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="otlp_persist")

router = APIRouter(prefix="/v1", tags=["otlp"])

# Singleton mapper
_mapper = OTelMapper()


def get_mapper() -> OTelMapper:
    """Dependency injection for mapper."""
    return _mapper


# -----------------------------------------------------------------
# TRACES ENDPOINT
# -----------------------------------------------------------------


@router.post("/traces")
async def receive_traces(
    request: Request,
    mapper: OTelMapper = Depends(get_mapper),
) -> Response:
    """
    Receive OTLP trace data.

    Accepts JSON or Protobuf encoded ExportTraceServiceRequest.
    Maps to enterprise.traces and enterprise.trace_spans.
    """
    content_type = request.headers.get("content-type", "application/json")

    try:
        if "protobuf" in content_type:
            raise HTTPException(501, "Protobuf not yet implemented - use JSON")

        body = await request.json()
        resource_spans = body.get("resourceSpans", [])

        if not resource_spans:
            return _empty_response()

        # Map to enterprise schema
        traces, spans = mapper.map_traces(resource_spans)

        # Persist to database
        await _persist_traces(traces, spans)

        logger.info(
            "Ingested %d traces, %d spans",
            len(traces),
            len(spans),
            extra={"trace_count": len(traces), "span_count": len(spans)},
        )

        return _success_response()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to process traces")
        raise HTTPException(500, f"Failed to process traces: {str(e)}")


# -----------------------------------------------------------------
# METRICS ENDPOINT
# -----------------------------------------------------------------


@router.post("/metrics")
async def receive_metrics(
    request: Request,
    mapper: OTelMapper = Depends(get_mapper),
) -> Response:
    """
    Receive OTLP metric data.

    Routes to enterprise.request_metrics or enterprise.llm_call_metrics
    based on metric name patterns.
    """
    content_type = request.headers.get("content-type", "application/json")

    try:
        if "protobuf" in content_type:
            raise HTTPException(501, "Protobuf not yet implemented - use JSON")

        body = await request.json()
        resource_metrics = body.get("resourceMetrics", [])

        if not resource_metrics:
            return _empty_response()

        # Map to enterprise schema
        metrics = mapper.map_metrics(resource_metrics)

        # Persist to database
        await _persist_metrics(metrics)

        logger.info(
            "Ingested %d metric data points",
            len(metrics),
            extra={"metric_count": len(metrics)},
        )

        return _success_response()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to process metrics")
        raise HTTPException(500, f"Failed to process metrics: {str(e)}")


# -----------------------------------------------------------------
# LOGS ENDPOINT
# -----------------------------------------------------------------


@router.post("/logs")
async def receive_logs(
    request: Request,
    mapper: OTelMapper = Depends(get_mapper),
) -> Response:
    """
    Receive OTLP log data.

    Maps to enterprise.structured_logs.
    """
    content_type = request.headers.get("content-type", "application/json")

    try:
        if "protobuf" in content_type:
            raise HTTPException(501, "Protobuf not yet implemented - use JSON")

        body = await request.json()
        resource_logs = body.get("resourceLogs", [])

        if not resource_logs:
            return _empty_response()

        # Map to enterprise schema
        logs = mapper.map_logs(resource_logs)

        # Persist to database
        await _persist_logs(logs)

        logger.info(
            "Ingested %d log records",
            len(logs),
            extra={"log_count": len(logs)},
        )

        return _success_response()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to process logs")
        raise HTTPException(500, f"Failed to process logs: {str(e)}")


# -----------------------------------------------------------------
# DATABASE CONNECTION (lazy import to avoid circular imports)
# -----------------------------------------------------------------


@contextmanager
def _get_connection():
    """Get connection from pool. Lazy import to avoid circular imports."""
    from ..service import get_pool
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


# -----------------------------------------------------------------
# PERSISTENCE - SYNC IMPLEMENTATIONS
# -----------------------------------------------------------------


def _persist_traces_sync(traces: List[MappedTrace], spans: List[MappedSpan]) -> None:
    """Sync implementation for trace persistence."""
    if not traces and not spans:
        return

    with _get_connection() as conn:
        with conn.cursor() as cur:
            # Upsert traces
            for trace in traces:
                cur.execute("""
                    INSERT INTO enterprise.traces
                    (trace_id, entry_point, endpoint, method, session_id,
                     user_email, department, start_time, end_time, duration_ms,
                     status, error_message, tags)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (trace_id) DO UPDATE SET
                        end_time = EXCLUDED.end_time,
                        duration_ms = EXCLUDED.duration_ms,
                        status = EXCLUDED.status,
                        error_message = EXCLUDED.error_message
                """, (
                    trace.trace_id,
                    trace.entry_point,
                    trace.endpoint,
                    trace.method,
                    trace.session_id,
                    trace.user_email,
                    trace.department,
                    trace.start_time,
                    trace.end_time,
                    trace.duration_ms,
                    trace.status,
                    trace.error_message,
                    json.dumps(trace.tags) if trace.tags else '{}',
                ))

            # Insert spans (no upsert - spans are immutable)
            for span in spans:
                cur.execute("""
                    INSERT INTO enterprise.trace_spans
                    (trace_id, span_id, parent_span_id, operation_name, service_name,
                     start_time, end_time, duration_ms, status, error_message, tags, logs)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (
                    span.trace_id,
                    span.span_id,
                    span.parent_span_id,
                    span.operation_name,
                    span.service_name,
                    span.start_time,
                    span.end_time,
                    span.duration_ms,
                    span.status,
                    span.error_message,
                    json.dumps(span.tags) if span.tags else '{}',
                    json.dumps(span.logs) if span.logs else '[]',
                ))

            conn.commit()

    logger.debug(
        "[OTLP] Persisted %d traces, %d spans",
        len(traces), len(spans)
    )


def _persist_logs_sync(logs: List[MappedLog]) -> None:
    """Sync implementation for log persistence."""
    if not logs:
        return

    with _get_connection() as conn:
        with conn.cursor() as cur:
            for log in logs:
                cur.execute("""
                    INSERT INTO enterprise.structured_logs
                    (timestamp, level, logger_name, message, trace_id, span_id,
                     user_email, department, session_id, endpoint, extra,
                     exception_type, exception_message, exception_traceback)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    log.timestamp,
                    log.level,
                    log.logger_name,
                    log.message,
                    log.trace_id,
                    log.span_id,
                    log.user_email,
                    log.department,
                    log.session_id,
                    log.endpoint,
                    json.dumps(log.extra) if log.extra else '{}',
                    log.exception_type,
                    log.exception_message,
                    log.exception_traceback,
                ))

            conn.commit()

    logger.debug("[OTLP] Persisted %d logs", len(logs))


def _persist_metrics_sync(metrics: List[MappedMetric]) -> None:
    """Sync implementation for metrics persistence."""
    if not metrics:
        return

    # Route metrics to appropriate tables
    request_metrics = [m for m in metrics if m.metric_type == "request"]
    llm_metrics = [m for m in metrics if m.metric_type == "llm"]

    with _get_connection() as conn:
        with conn.cursor() as cur:
            # Insert request metrics
            for m in request_metrics:
                attrs = m.attributes or {}
                cur.execute("""
                    INSERT INTO enterprise.request_metrics
                    (timestamp, endpoint, method, status_code, response_time_ms,
                     user_email, department, request_size_bytes, response_size_bytes, trace_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    m.timestamp,
                    attrs.get("http.route") or attrs.get("http.target"),
                    attrs.get("http.method") or attrs.get("http.request.method"),
                    attrs.get("http.status_code") or attrs.get("http.response.status_code"),
                    m.value if "duration" in m.metric_name.lower() or "latency" in m.metric_name.lower() else None,
                    attrs.get("user.email"),
                    attrs.get("department"),
                    attrs.get("http.request_content_length"),
                    attrs.get("http.response_content_length"),
                    attrs.get("trace_id"),
                ))

            # Insert LLM metrics
            for m in llm_metrics:
                attrs = m.attributes or {}
                cur.execute("""
                    INSERT INTO enterprise.llm_call_metrics
                    (timestamp, model, provider, prompt_tokens, completion_tokens,
                     total_tokens, elapsed_ms, user_email, department, trace_id, success)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    m.timestamp,
                    attrs.get("llm.model") or attrs.get("gen_ai.request.model"),
                    attrs.get("llm.provider") or attrs.get("gen_ai.system"),
                    attrs.get("llm.prompt_tokens") or attrs.get("gen_ai.usage.prompt_tokens"),
                    attrs.get("llm.completion_tokens") or attrs.get("gen_ai.usage.completion_tokens"),
                    attrs.get("llm.total_tokens"),
                    m.value if "duration" in m.metric_name.lower() else None,
                    attrs.get("user.email"),
                    attrs.get("department"),
                    attrs.get("trace_id"),
                    attrs.get("llm.success", True),
                ))

            conn.commit()

    logger.debug(
        "[OTLP] Persisted %d request metrics, %d LLM metrics",
        len(request_metrics), len(llm_metrics)
    )


# -----------------------------------------------------------------
# PERSISTENCE - ASYNC WRAPPERS
# -----------------------------------------------------------------


async def _persist_traces(traces: List[MappedTrace], spans: List[MappedSpan]) -> None:
    """Persist traces and spans to enterprise.traces and enterprise.trace_spans."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _persist_traces_sync, traces, spans)


async def _persist_logs(logs: List[MappedLog]) -> None:
    """Persist logs to enterprise.structured_logs."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _persist_logs_sync, logs)


async def _persist_metrics(metrics: List[MappedMetric]) -> None:
    """Persist metrics to enterprise.request_metrics or enterprise.llm_call_metrics."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _persist_metrics_sync, metrics)


# -----------------------------------------------------------------
# RESPONSE HELPERS
# -----------------------------------------------------------------


def _success_response() -> Response:
    """Standard OTLP success response."""
    # OTLP expects empty JSON object on success
    return JSONResponse(content={}, status_code=200)


def _empty_response() -> Response:
    """Response for empty payload."""
    return JSONResponse(content={}, status_code=200)

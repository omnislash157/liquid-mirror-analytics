"""
OTLP Receiver - FastAPI routes for OpenTelemetry Protocol ingestion.

Endpoints:
- POST /v1/traces  - Receive trace data (OTLP/HTTP)
- POST /v1/metrics - Receive metric data (OTLP/HTTP)
- POST /v1/logs    - Receive log data (OTLP/HTTP)

All endpoints accept:
- application/json (JSON-encoded OTLP)
- application/x-protobuf (Protobuf-encoded OTLP) [not yet implemented]
"""

import logging
from typing import List

from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse

from .mapper import OTelMapper, MappedTrace, MappedSpan, MappedLog, MappedMetric

logger = logging.getLogger(__name__)

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
# PERSISTENCE (implement based on your DB setup)
# -----------------------------------------------------------------


async def _persist_traces(
    traces: List[MappedTrace],
    spans: List[MappedSpan],
) -> None:
    """
    Persist traces and spans to enterprise.traces and enterprise.trace_spans.

    TODO: Replace with actual database calls using existing db module.
    """
    # Example using asyncpg or existing database.py:
    #
    # async with db.pool.acquire() as conn:
    #     for trace in traces:
    #         await conn.execute('''
    #             INSERT INTO enterprise.traces
    #             (trace_id, entry_point, endpoint, method, session_id,
    #              user_email, department, start_time, end_time, duration_ms,
    #              status, error_message, tags)
    #             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
    #             ON CONFLICT (trace_id) DO UPDATE SET
    #                 end_time = EXCLUDED.end_time,
    #                 duration_ms = EXCLUDED.duration_ms,
    #                 status = EXCLUDED.status
    #         ''', trace.trace_id, trace.entry_point, ...)
    #
    #     for span in spans:
    #         await conn.execute('''
    #             INSERT INTO enterprise.trace_spans ...
    #         ''')

    logger.debug("Would persist %d traces, %d spans", len(traces), len(spans))


async def _persist_logs(logs: List[MappedLog]) -> None:
    """
    Persist logs to enterprise.structured_logs.

    TODO: Replace with actual database calls.
    """
    # async with db.pool.acquire() as conn:
    #     await conn.executemany('''
    #         INSERT INTO enterprise.structured_logs
    #         (timestamp, level, logger_name, message, trace_id, span_id,
    #          user_email, department, session_id, endpoint, extra,
    #          exception_type, exception_message, exception_traceback)
    #         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
    #     ''', [(l.timestamp, l.level, ...) for l in logs])

    logger.debug("Would persist %d logs", len(logs))


async def _persist_metrics(metrics: List[MappedMetric]) -> None:
    """
    Persist metrics to enterprise.request_metrics or enterprise.llm_call_metrics.

    TODO: Replace with actual database calls.
    """
    # Route to appropriate table based on metric_type
    # request_metrics = [m for m in metrics if m.metric_type == "request"]
    # llm_metrics = [m for m in metrics if m.metric_type == "llm"]

    logger.debug("Would persist %d metrics", len(metrics))


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

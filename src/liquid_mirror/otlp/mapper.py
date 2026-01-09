"""
OTel -> Enterprise Schema Mapper

Maps OpenTelemetry data structures to existing enterprise.* tables.
This is the core translation layer that makes any OTel client compatible.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .semantic_conventions import ResourceAttributes, SpanAttributes, LogAttributes

logger = logging.getLogger(__name__)


@dataclass
class MappedTrace:
    """Mapped to enterprise.traces"""

    trace_id: str
    entry_point: str
    endpoint: Optional[str]
    method: Optional[str]
    session_id: Optional[str]
    user_email: Optional[str]
    department: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    status: str
    error_message: Optional[str]
    tags: Dict[str, Any]


@dataclass
class MappedSpan:
    """Mapped to enterprise.trace_spans"""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    status: str
    error_message: Optional[str]
    tags: Dict[str, Any]
    logs: List[Dict[str, Any]]


@dataclass
class MappedLog:
    """Mapped to enterprise.structured_logs"""

    timestamp: datetime
    level: str
    logger_name: str
    message: str
    trace_id: Optional[str]
    span_id: Optional[str]
    user_email: Optional[str]
    department: Optional[str]
    session_id: Optional[str]
    endpoint: Optional[str]
    extra: Dict[str, Any]
    exception_type: Optional[str]
    exception_message: Optional[str]
    exception_traceback: Optional[str]


@dataclass
class MappedMetric:
    """Mapped to enterprise.request_metrics or llm_call_metrics"""

    timestamp: datetime
    metric_name: str
    metric_type: str  # 'request' or 'llm'
    value: float
    attributes: Dict[str, Any]


class OTelMapper:
    """
    Maps OpenTelemetry Protocol data to enterprise schema.

    Handles:
    - Traces/Spans -> enterprise.traces + enterprise.trace_spans
    - Logs -> enterprise.structured_logs
    - Metrics -> enterprise.request_metrics / enterprise.llm_call_metrics
    """

    # Severity number to level name mapping (OTel standard)
    SEVERITY_MAP = {
        1: "TRACE",
        2: "TRACE",
        3: "TRACE",
        4: "TRACE",
        5: "DEBUG",
        6: "DEBUG",
        7: "DEBUG",
        8: "DEBUG",
        9: "INFO",
        10: "INFO",
        11: "INFO",
        12: "INFO",
        13: "WARN",
        14: "WARN",
        15: "WARN",
        16: "WARN",
        17: "ERROR",
        18: "ERROR",
        19: "ERROR",
        20: "ERROR",
        21: "FATAL",
        22: "FATAL",
        23: "FATAL",
        24: "FATAL",
    }

    def __init__(self, default_service: str = "unknown"):
        self.default_service = default_service

    # -----------------------------------------------------------------
    # TRACE/SPAN MAPPING
    # -----------------------------------------------------------------

    def map_traces(
        self, resource_spans: List[Dict[str, Any]]
    ) -> Tuple[List[MappedTrace], List[MappedSpan]]:
        """
        Map OTLP TracesData to enterprise schema.

        Args:
            resource_spans: List of ResourceSpans from OTLP payload

        Returns:
            Tuple of (traces, spans) ready for DB insertion
        """
        traces: Dict[str, MappedTrace] = {}
        spans: List[MappedSpan] = []

        for resource_span in resource_spans:
            # Extract resource attributes (service info, k8s, cloud, etc.)
            resource_attrs = self._extract_attributes(
                resource_span.get("resource", {}).get("attributes", [])
            )

            service_name = resource_attrs.get(
                ResourceAttributes.SERVICE_NAME, self.default_service
            )

            # Process each scope (instrumentation library)
            for scope_span in resource_span.get("scopeSpans", []):
                for span in scope_span.get("spans", []):
                    mapped_span = self._map_span(span, service_name, resource_attrs)
                    spans.append(mapped_span)

                    # Build/update trace from root spans
                    if not mapped_span.parent_span_id:
                        trace = self._build_trace_from_root_span(
                            mapped_span, resource_attrs
                        )
                        traces[mapped_span.trace_id] = trace

        return list(traces.values()), spans

    def _map_span(
        self,
        span: Dict[str, Any],
        service_name: str,
        resource_attrs: Dict[str, Any],
    ) -> MappedSpan:
        """Map a single OTel span to MappedSpan."""

        span_attrs = self._extract_attributes(span.get("attributes", []))

        # Merge resource attrs into span tags
        tags = {**resource_attrs, **span_attrs}

        # Parse timestamps (OTel uses nanoseconds)
        start_ns = int(span.get("startTimeUnixNano", 0))
        end_ns = int(span.get("endTimeUnixNano", 0))

        start_time = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)
        end_time = (
            datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc) if end_ns else None
        )

        duration_ms = (end_ns - start_ns) / 1e6 if end_ns else None

        # Map status
        status_obj = span.get("status", {})
        status_code = status_obj.get("code", 0)
        status = {0: "unset", 1: "ok", 2: "error"}.get(status_code, "unset")
        error_message = status_obj.get("message") if status == "error" else None

        # Map events to logs format
        logs = [
            {
                "timestamp": datetime.fromtimestamp(
                    int(e.get("timeUnixNano", 0)) / 1e9, tz=timezone.utc
                ).isoformat(),
                "name": e.get("name", ""),
                "attributes": self._extract_attributes(e.get("attributes", [])),
            }
            for e in span.get("events", [])
        ]

        return MappedSpan(
            trace_id=span.get("traceId", ""),
            span_id=span.get("spanId", ""),
            parent_span_id=span.get("parentSpanId") or None,
            operation_name=span.get("name", "unknown"),
            service_name=service_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            status=status,
            error_message=error_message,
            tags=tags,
            logs=logs,
        )

    def _build_trace_from_root_span(
        self, root_span: MappedSpan, resource_attrs: Dict[str, Any]
    ) -> MappedTrace:
        """Build a trace record from a root span."""

        tags = root_span.tags

        # Determine entry point from span kind or attributes
        http_method = tags.get(SpanAttributes.HTTP_METHOD) or tags.get(
            SpanAttributes.HTTP_REQUEST_METHOD
        )

        if http_method:
            entry_point = "http"
        elif tags.get(SpanAttributes.RPC_SYSTEM):
            entry_point = "rpc"
        elif tags.get(SpanAttributes.MESSAGING_SYSTEM):
            entry_point = "messaging"
        elif tags.get(SpanAttributes.DB_SYSTEM):
            entry_point = "db"
        else:
            entry_point = "internal"

        # Extract endpoint
        endpoint = (
            tags.get(SpanAttributes.HTTP_ROUTE)
            or tags.get(SpanAttributes.HTTP_TARGET)
            or tags.get(SpanAttributes.HTTP_URL)
        )

        return MappedTrace(
            trace_id=root_span.trace_id,
            entry_point=entry_point,
            endpoint=endpoint,
            method=http_method,
            session_id=tags.get(SpanAttributes.SESSION_ID),
            user_email=tags.get(SpanAttributes.USER_EMAIL),
            department=tags.get(SpanAttributes.DEPARTMENT),
            start_time=root_span.start_time,
            end_time=root_span.end_time,
            duration_ms=root_span.duration_ms,
            status=root_span.status,
            error_message=root_span.error_message,
            tags=tags,
        )

    # -----------------------------------------------------------------
    # LOG MAPPING
    # -----------------------------------------------------------------

    def map_logs(self, resource_logs: List[Dict[str, Any]]) -> List[MappedLog]:
        """
        Map OTLP LogsData to enterprise.structured_logs schema.

        Args:
            resource_logs: List of ResourceLogs from OTLP payload

        Returns:
            List of MappedLog ready for DB insertion
        """
        mapped_logs: List[MappedLog] = []

        for resource_log in resource_logs:
            resource_attrs = self._extract_attributes(
                resource_log.get("resource", {}).get("attributes", [])
            )

            service_name = resource_attrs.get(
                ResourceAttributes.SERVICE_NAME, self.default_service
            )

            for scope_log in resource_log.get("scopeLogs", []):
                scope_name = scope_log.get("scope", {}).get("name", service_name)

                for log_record in scope_log.get("logRecords", []):
                    mapped_log = self._map_log_record(
                        log_record, scope_name, resource_attrs
                    )
                    mapped_logs.append(mapped_log)

        return mapped_logs

    def _map_log_record(
        self,
        record: Dict[str, Any],
        logger_name: str,
        resource_attrs: Dict[str, Any],
    ) -> MappedLog:
        """Map a single OTel log record."""

        log_attrs = self._extract_attributes(record.get("attributes", []))
        all_attrs = {**resource_attrs, **log_attrs}

        # Timestamp
        time_ns = int(record.get("timeUnixNano", 0))
        timestamp = datetime.fromtimestamp(time_ns / 1e9, tz=timezone.utc)

        # Severity
        severity_num = record.get("severityNumber", 9)
        level = record.get("severityText") or self.SEVERITY_MAP.get(severity_num, "INFO")

        # Body (message)
        body = record.get("body", {})
        if isinstance(body, dict):
            message = body.get("stringValue", "") or str(body)
        else:
            message = str(body)

        # Exception info
        exception_type = log_attrs.get(LogAttributes.EXCEPTION_TYPE)
        exception_message = log_attrs.get(LogAttributes.EXCEPTION_MESSAGE)
        exception_traceback = log_attrs.get(LogAttributes.EXCEPTION_STACKTRACE)

        # Build extra (everything not mapped to a column)
        extra = {
            k: v
            for k, v in all_attrs.items()
            if k
            not in {
                SpanAttributes.USER_EMAIL,
                SpanAttributes.DEPARTMENT,
                SpanAttributes.SESSION_ID,
                LogAttributes.EXCEPTION_TYPE,
                LogAttributes.EXCEPTION_MESSAGE,
                LogAttributes.EXCEPTION_STACKTRACE,
            }
        }

        return MappedLog(
            timestamp=timestamp,
            level=level.upper(),
            logger_name=logger_name,
            message=message,
            trace_id=record.get("traceId"),
            span_id=record.get("spanId"),
            user_email=all_attrs.get(SpanAttributes.USER_EMAIL),
            department=all_attrs.get(SpanAttributes.DEPARTMENT),
            session_id=all_attrs.get(SpanAttributes.SESSION_ID),
            endpoint=all_attrs.get(SpanAttributes.HTTP_ROUTE),
            extra=extra,
            exception_type=exception_type,
            exception_message=exception_message,
            exception_traceback=exception_traceback,
        )

    # -----------------------------------------------------------------
    # METRICS MAPPING
    # -----------------------------------------------------------------

    def map_metrics(
        self, resource_metrics: List[Dict[str, Any]]
    ) -> List[MappedMetric]:
        """
        Map OTLP MetricsData to enterprise schema.

        Routes to request_metrics or llm_call_metrics based on metric name.
        """
        mapped_metrics: List[MappedMetric] = []

        for resource_metric in resource_metrics:
            resource_attrs = self._extract_attributes(
                resource_metric.get("resource", {}).get("attributes", [])
            )

            for scope_metric in resource_metric.get("scopeMetrics", []):
                for metric in scope_metric.get("metrics", []):
                    metric_name = metric.get("name", "")

                    # Extract data points based on metric type
                    data_points = self._extract_metric_data_points(metric)

                    for dp in data_points:
                        dp_attrs = self._extract_attributes(dp.get("attributes", []))
                        all_attrs = {**resource_attrs, **dp_attrs}

                        # Timestamp
                        time_ns = int(dp.get("timeUnixNano", 0))
                        timestamp = datetime.fromtimestamp(
                            time_ns / 1e9, tz=timezone.utc
                        )

                        # Value
                        value = dp.get("asDouble") or dp.get("asInt") or 0

                        # Determine metric type for routing
                        if "llm" in metric_name.lower() or "token" in metric_name.lower():
                            metric_type = "llm"
                        else:
                            metric_type = "request"

                        mapped_metrics.append(
                            MappedMetric(
                                timestamp=timestamp,
                                metric_name=metric_name,
                                metric_type=metric_type,
                                value=float(value),
                                attributes=all_attrs,
                            )
                        )

        return mapped_metrics

    def _extract_metric_data_points(
        self, metric: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract data points from any metric type."""

        # OTel metrics can be gauge, sum, histogram, etc.
        if "gauge" in metric:
            return metric["gauge"].get("dataPoints", [])
        elif "sum" in metric:
            return metric["sum"].get("dataPoints", [])
        elif "histogram" in metric:
            return metric["histogram"].get("dataPoints", [])
        elif "summary" in metric:
            return metric["summary"].get("dataPoints", [])

        return []

    # -----------------------------------------------------------------
    # UTILITIES
    # -----------------------------------------------------------------

    def _extract_attributes(
        self, attributes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract key-value pairs from OTel attribute format.

        OTel attributes come as:
        [{"key": "foo", "value": {"stringValue": "bar"}}, ...]
        """
        result = {}

        for attr in attributes:
            key = attr.get("key", "")
            value_obj = attr.get("value", {})

            # OTel values are typed
            if "stringValue" in value_obj:
                result[key] = value_obj["stringValue"]
            elif "intValue" in value_obj:
                result[key] = int(value_obj["intValue"])
            elif "doubleValue" in value_obj:
                result[key] = float(value_obj["doubleValue"])
            elif "boolValue" in value_obj:
                result[key] = value_obj["boolValue"]
            elif "arrayValue" in value_obj:
                result[key] = [
                    self._extract_value(v)
                    for v in value_obj["arrayValue"].get("values", [])
                ]
            elif "kvlistValue" in value_obj:
                result[key] = {
                    kv["key"]: self._extract_value(kv["value"])
                    for kv in value_obj["kvlistValue"].get("values", [])
                }

        return result

    def _extract_value(self, value_obj: Dict[str, Any]) -> Any:
        """Extract a single typed value."""
        if "stringValue" in value_obj:
            return value_obj["stringValue"]
        elif "intValue" in value_obj:
            return int(value_obj["intValue"])
        elif "doubleValue" in value_obj:
            return float(value_obj["doubleValue"])
        elif "boolValue" in value_obj:
            return value_obj["boolValue"]
        return None

# OTLP Persistence Wiring - Next Session

**Created:** 2026-01-09
**Status:** Ready to implement
**Prereq:** OTLP receiver module complete (mapper, receiver, semantic_conventions)

---

## What's Done

- `otlp/semantic_conventions.py` - OTel attribute constants
- `otlp/mapper.py` - OTelMapper maps OTLP payloads to dataclasses
- `otlp/receiver.py` - FastAPI routes at `/v1/traces`, `/v1/metrics`, `/v1/logs`
- `main.py` - Router mounted, endpoints documented

## What's Stubbed (TODOs)

Three persistence functions in `receiver.py` need wiring:

```python
async def _persist_traces(traces, spans) -> None  # Line 187
async def _persist_logs(logs) -> None             # Line 220
async def _persist_metrics(metrics) -> None       # Line 238
```

---

## Implementation Plan

### 1. Import Pattern

Use the existing sync pool pattern from `ingest.py`:

```python
from contextlib import contextmanager
from ..service import get_pool
import json

@contextmanager
def _get_connection():
    """Get connection from pool."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)
```

### 2. Async-to-Sync Bridge

Routes are `async def` but pool is sync. Use `run_in_executor`:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=4)

async def _persist_traces(traces, spans):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _persist_traces_sync, traces, spans)
```

### 3. Target Tables (from DATABASE_SCHEMA_MAP.md)

| Dataclass | Target Table | Key Columns |
|-----------|--------------|-------------|
| MappedTrace | enterprise.traces | trace_id (UNIQUE), entry_point, endpoint, method, session_id, user_email, department, start_time, end_time, duration_ms, status, error_message, tags |
| MappedSpan | enterprise.trace_spans | trace_id (FK), span_id, parent_span_id, operation_name, service_name, start_time, end_time, duration_ms, status, error_message, tags, logs |
| MappedLog | enterprise.structured_logs | timestamp, level, logger_name, message, trace_id, span_id, user_email, department, session_id, endpoint, extra, exception_type, exception_message, exception_traceback |
| MappedMetric (request) | enterprise.request_metrics | timestamp, endpoint, method, status_code, response_time_ms, user_email, department, request_size_bytes, response_size_bytes, trace_id |
| MappedMetric (llm) | enterprise.llm_call_metrics | timestamp, model, provider, prompt_tokens, completion_tokens, total_tokens, elapsed_ms, first_token_ms, user_email, department, query_category, trace_id, cost_usd, success, error_message |

### 4. SQL Templates

**Traces (upsert on trace_id):**
```sql
INSERT INTO enterprise.traces
(trace_id, entry_point, endpoint, method, session_id, user_email, department,
 start_time, end_time, duration_ms, status, error_message, tags)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (trace_id) DO UPDATE SET
    end_time = EXCLUDED.end_time,
    duration_ms = EXCLUDED.duration_ms,
    status = EXCLUDED.status,
    error_message = EXCLUDED.error_message
```

**Spans (insert, no upsert needed):**
```sql
INSERT INTO enterprise.trace_spans
(trace_id, span_id, parent_span_id, operation_name, service_name,
 start_time, end_time, duration_ms, status, error_message, tags, logs)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
```

**Logs (insert):**
```sql
INSERT INTO enterprise.structured_logs
(timestamp, level, logger_name, message, trace_id, span_id,
 user_email, department, session_id, endpoint, extra,
 exception_type, exception_message, exception_traceback)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
```

### 5. Metrics Routing Logic

```python
def _persist_metrics_sync(metrics: List[MappedMetric]) -> None:
    request_metrics = [m for m in metrics if m.metric_type == "request"]
    llm_metrics = [m for m in metrics if m.metric_type == "llm"]

    # Insert to respective tables
    # Note: OTel metrics may not map 1:1 to our schema
    # May need to aggregate or transform
```

---

## Gotchas

1. **JSONB columns**: `tags`, `logs`, `extra` need `json.dumps()` before insert
2. **Timestamps**: MappedX dataclasses use `datetime` - psycopg2 handles natively
3. **High volume**: `structured_logs` has 33K+ rows - consider batch inserts
4. **Trigger**: `structured_logs` has `notify_new_log()` trigger - will fire on inserts

---

## Validation Checklist

- [ ] Import `get_pool` from `..service`
- [ ] Add `_get_connection()` context manager
- [ ] Implement `_persist_traces_sync()` with upsert
- [ ] Implement `_persist_logs_sync()` with batch insert
- [ ] Implement `_persist_metrics_sync()` with routing
- [ ] Wrap sync functions with `run_in_executor`
- [ ] Test with curl payload from BUILD_SHEET
- [ ] Verify rows in DB

---

## Test Payload

```bash
curl -X POST http://localhost:8001/v1/traces \
  -H "Content-Type: application/json" \
  -d '{
    "resourceSpans": [{
      "resource": {
        "attributes": [
          {"key": "service.name", "value": {"stringValue": "test-service"}}
        ]
      },
      "scopeSpans": [{
        "spans": [{
          "traceId": "5b8aa5a2d2c872e8321cf37308d69df2",
          "spanId": "051581bf3cb55c13",
          "name": "test-span",
          "kind": 1,
          "startTimeUnixNano": "1704067200000000000",
          "endTimeUnixNano": "1704067200100000000",
          "status": {"code": 1}
        }]
      }]
    }]
  }'
```

---

**END OF PLAN**

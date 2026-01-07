# Liquid Mirror Analytics - Quickstart

## Backend

### Install (development mode)
```bash
cd C:\Users\mthar\projects\liquid_mirror_analytics
pip install -e .
```

### Use in Python
```python
from liquid_mirror import get_analytics_service, get_event_bus, EVENT_QUERY

# Log a query
analytics = get_analytics_service()
query_id = analytics.log_query(
    user_email="user@example.com",
    department="warehouse",
    query_text="How do I process a return?",
    session_id="abc123",
    response_time_ms=250,
    response_length=500,
    tokens_input=50,
    tokens_output=100,
    model_used="grok-beta"
)

# Subscribe to events
def on_query(event):
    print(f"Query logged: {event.query_text}")

bus = get_event_bus()
bus.subscribe(EVENT_QUERY, on_query)
```

### FastAPI Integration
```python
from fastapi import FastAPI
from liquid_mirror import analytics_router

app = FastAPI()
app.include_router(analytics_router, prefix="/api/analytics")
```

### Run Tests
```bash
cd C:\Users\mthar\projects\liquid_mirror_analytics
PYTHONPATH=src pytest tests/ -v
```

---

## Frontend

### Install
```bash
cd frontend
npm install
```

### Dev Server
```bash
npm run dev
```

### Build
```bash
npm run build
```

### Configure API URL
Edit `src/lib/stores/analytics.ts` and update the fetch URLs to point to your backend.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AZURE_PG_USER` | PostgreSQL username |
| `AZURE_PG_PASSWORD` | PostgreSQL password |
| `AZURE_PG_HOST` | PostgreSQL host |
| `AZURE_PG_PORT` | PostgreSQL port (default: 5432) |
| `AZURE_PG_DATABASE` | PostgreSQL database (default: postgres) |

---

## Database Tables

Uses `enterprise` schema:
- `enterprise.query_log` - Query logging with heuristics
- `enterprise.analytics_events` - Session events (login, logout, errors)

# Liquid Mirror Analytics

Enterprise AI observability and analytics engine with event hooks for metacognitive systems.

## Features

- **Query Logging**: Log and classify AI assistant queries with heuristics analysis
- **Event Tracking**: Track session events (login, logout, department switches, errors)
- **Heuristics Engine**: Analyze query complexity, intent, and department context
- **Event Bus**: Decoupled pub/sub for integration with cognitive systems
- **Dashboard API**: FastAPI routes for analytics dashboards

## Installation

```bash
pip install liquid-mirror-analytics
```

Or install from source:

```bash
git clone https://github.com/mhartigan/liquid-mirror-analytics.git
cd liquid-mirror-analytics
pip install -e .
```

## Quick Start

```python
from liquid_mirror import get_analytics_service, analytics_router

# Get the analytics service
analytics = get_analytics_service()

# Log a query
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

# Log an event
event_id = analytics.log_event(
    event_type="login",
    user_email="user@example.com",
    session_id="abc123"
)
```

## Event Bus Integration

The analytics service emits events for integration with cognitive systems:

```python
from liquid_mirror import get_event_bus, EVENT_QUERY, AnalyticsQueryEvent

# Subscribe to query events
bus = get_event_bus()

def on_query(event: AnalyticsQueryEvent):
    print(f"Query logged: {event.query_text}")

bus.subscribe(EVENT_QUERY, on_query)
```

## FastAPI Integration

```python
from fastapi import FastAPI
from liquid_mirror import analytics_router

app = FastAPI()
app.include_router(analytics_router, prefix="/api/analytics")
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_PG_USER` | PostgreSQL username | - |
| `AZURE_PG_PASSWORD` | PostgreSQL password | - |
| `AZURE_PG_HOST` | PostgreSQL host | - |
| `AZURE_PG_PORT` | PostgreSQL port | 5432 |
| `AZURE_PG_DATABASE` | PostgreSQL database | postgres |

## Database Schema

The service uses the `enterprise` schema with two tables:

- `enterprise.query_log` - Query logging with heuristics
- `enterprise.analytics_events` - Session events

## License

MIT

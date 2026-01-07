"""
Liquid Mirror Analytics - Enterprise AI observability and analytics.

A standalone analytics engine for AI assistants with:
- Query logging and classification
- Heuristics-based query analysis
- Event bus for decoupled integration with cognitive systems
- Dashboard API routes
- Metacognitive analysis (cognitive patterns, memory thermodynamics)

Usage:
    from liquid_mirror import get_analytics_service, analytics_router

    # Get the analytics service singleton
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

    # Mount the router in FastAPI
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(analytics_router, prefix="/api/analytics")

    # Use metacognitive mirror for cognitive analysis
    from liquid_mirror.mirror import MetacognitiveMirror
    mirror = MetacognitiveMirror()
"""

from .service import (
    AnalyticsService,
    get_analytics_service,
    get_pool,
)

from .routes import analytics_router

from .events import (
    AnalyticsQueryEvent,
    AnalyticsSessionEvent,
    AnalyticsAccessEvent,
    get_event_bus,
    EVENT_QUERY,
    EVENT_SESSION,
    EVENT_ACCESS,
)

# Metacognitive mirror exports
from .mirror import (
    MetacognitiveMirror,
    CognitivePhase,
    DriftSignal,
    QueryEvent,
    MemoryAccessEvent,
    CognitiveSnapshot,
    QueryArchaeologist,
    MemoryThermodynamics,
    CognitiveSeismograph,
    PredictivePrefetcher,
)

__version__ = "0.2.0"

__all__ = [
    # Service
    "AnalyticsService",
    "get_analytics_service",
    "get_pool",
    # Routes
    "analytics_router",
    # Events
    "AnalyticsQueryEvent",
    "AnalyticsSessionEvent",
    "AnalyticsAccessEvent",
    "get_event_bus",
    "EVENT_QUERY",
    "EVENT_SESSION",
    "EVENT_ACCESS",
    # Mirror (Metacognitive)
    "MetacognitiveMirror",
    "CognitivePhase",
    "DriftSignal",
    "QueryEvent",
    "MemoryAccessEvent",
    "CognitiveSnapshot",
    "QueryArchaeologist",
    "MemoryThermodynamics",
    "CognitiveSeismograph",
    "PredictivePrefetcher",
]

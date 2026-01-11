"""
Liquid Mirror Analytics - FastAPI Application

Multi-tenant AI observability platform with:
- Ingest API (/api/v1/ingest/*) - API key authenticated event ingestion
- Analytics dashboard API (/api/analytics/*) - Dashboard queries
- Metacognitive mirror API (/api/mirror/*) - Cognitive analysis
- OTLP receiver (/v1/traces, /v1/metrics, /v1/logs) - OpenTelemetry ingestion
- Health checks

Deploy as separate Railway service or run locally:
    uvicorn liquid_mirror.main:app --host 0.0.0.0 --port 8001
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .routes import analytics_router
from .ingest import ingest_router
from .otlp import otlp_router
from .service import get_analytics_service
from .logging_config import setup_logging

# Configure structured logging (JSON in production, human-readable in dev)
# Set LM_JSON_LOGS=false for development
json_output = os.getenv("LM_JSON_LOGS", "true").lower() != "false"
setup_logging(json_output=json_output)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("[LiquidMirror] Starting up...")

    # Initialize analytics service (creates DB pool)
    try:
        analytics = get_analytics_service()
        logger.info("[LiquidMirror] Analytics service initialized")
    except Exception as e:
        logger.warning(f"[LiquidMirror] Analytics service init failed: {e}")

    # Wire metacognitive mirror to event bus
    try:
        _setup_mirror_event_subscription()
    except Exception as e:
        logger.warning(f"[LiquidMirror] Mirror event subscription failed: {e}")

    yield

    # Cleanup
    logger.info("[LiquidMirror] Shutting down...")


def _setup_mirror_event_subscription() -> None:
    """Subscribe metacognitive mirror to event bus for cognitive tracking."""
    import numpy as np
    from .events import get_event_bus, EVENT_QUERY
    from .mirror import QueryEvent

    def handle_query_event(event) -> None:
        """Bridge AnalyticsQueryEvent -> QueryEvent for mirror."""
        try:
            mirror = get_mirror()

            # Convert AnalyticsQueryEvent to mirror's QueryEvent format
            query_event = QueryEvent(
                timestamp=event.timestamp,
                query_text=event.query_text,
                query_embedding=event.query_embedding if event.query_embedding is not None else np.zeros(1536, dtype=np.float32),
                retrieved_memory_ids=event.retrieved_memory_ids or [],
                retrieval_scores=event.retrieval_scores or [],
                execution_time_ms=event.execution_time_ms,
                result_count=event.result_count,
                semantic_gate_passed=event.semantic_gate_passed,
            )
            mirror.record_query(query_event)
        except Exception as e:
            logger.error(f"[Mirror] Failed to process query event: {e}")

    bus = get_event_bus()
    bus.subscribe(EVENT_QUERY, handle_query_event)
    logger.info("[LiquidMirror] MetacognitiveMirror subscribed to EVENT_QUERY")


app = FastAPI(
    title="Liquid Mirror Analytics",
    description="Enterprise AI observability and analytics platform",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS - configure for your frontend origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(analytics_router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(ingest_router, prefix="/api/v1/ingest", tags=["Ingest"])
app.include_router(otlp_router, tags=["OTLP"])  # Standard OTLP endpoints at /v1/traces, /v1/metrics, /v1/logs


# =============================================================================
# HEALTH & STATUS
# =============================================================================

@app.get("/health")
def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "service": "liquid-mirror-analytics",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/")
def root():
    """Root endpoint with service info."""
    return {
        "service": "Liquid Mirror Analytics",
        "version": "0.3.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "ingest": "/api/v1/ingest/query",
            "analytics": "/api/analytics/dashboard",
            "mirror": "/api/mirror/insights",
            "otlp": {
                "traces": "/v1/traces",
                "metrics": "/v1/metrics",
                "logs": "/v1/logs"
            }
        }
    }


# =============================================================================
# MIRROR API (Metacognitive endpoints)
# =============================================================================

# Global mirror instance (lazy init)
_mirror_instance = None


def get_mirror():
    """Get or create the MetacognitiveMirror singleton."""
    global _mirror_instance
    if _mirror_instance is None:
        from .mirror import MetacognitiveMirror
        _mirror_instance = MetacognitiveMirror()
        logger.info("[LiquidMirror] MetacognitiveMirror initialized")
    return _mirror_instance


@app.get("/api/mirror/insights", tags=["Mirror"])
def get_mirror_insights():
    """Get real-time metacognitive insights."""
    try:
        mirror = get_mirror()
        return mirror.get_real_time_insights()
    except Exception as e:
        logger.error(f"[Mirror] Error getting insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mirror/health-check", tags=["Mirror"])
def run_mirror_health_check():
    """Run cognitive system health analysis."""
    try:
        mirror = get_mirror()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "insights": mirror.run_health_check()
        }
    except Exception as e:
        logger.error(f"[Mirror] Error running health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mirror/optimizations", tags=["Mirror"])
def get_mirror_optimizations():
    """Get architectural optimization suggestions."""
    try:
        mirror = get_mirror()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "suggestions": mirror.suggest_optimizations()
        }
    except Exception as e:
        logger.error(f"[Mirror] Error getting optimizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mirror/predict", tags=["Mirror"])
def predict_next_memories(memory_ids: list[str], top_k: int = 5):
    """Predict next likely memory accesses."""
    try:
        mirror = get_mirror()
        return mirror.predict_next_access(memory_ids, top_k=top_k)
    except Exception as e:
        logger.error(f"[Mirror] Error predicting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(
        "liquid_mirror.main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )

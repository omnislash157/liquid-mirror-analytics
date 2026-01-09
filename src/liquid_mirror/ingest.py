"""
Liquid Mirror - Multi-Tenant Ingestion API

Handles incoming telemetry from tenants with API key authentication.
Wires data to MetacognitiveMirror for cognitive analysis.

Routes:
    POST /api/v1/ingest/query    - Log a query event
    POST /api/v1/ingest/session  - Log a session event
    POST /api/v1/ingest/batch    - Batch ingest multiple events
"""

import logging
import hashlib
from datetime import datetime, timezone
from typing import Optional, List, Any
from contextlib import contextmanager

from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np

from .service import get_pool

logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class QueryEventPayload(BaseModel):
    """Incoming query event from a tenant."""
    # Required
    query_text: str = Field(..., min_length=1)

    # Identity (optional - anonymous allowed)
    user_email: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None

    # Context
    department: Optional[str] = None

    # Performance metrics
    response_time_ms: Optional[float] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    model_used: Optional[str] = None

    # Pre-computed heuristics (optional - we can compute if missing)
    complexity_score: Optional[float] = None
    intent_type: Optional[str] = None
    specificity_score: Optional[float] = None
    temporal_urgency: Optional[str] = None
    is_multi_part: Optional[bool] = None

    # Memory retrieval context
    memory_ids_retrieved: Optional[List[str]] = None
    retrieval_scores: Optional[List[float]] = None

    # Embedding (optional - 1024 dim vector)
    query_embedding: Optional[List[float]] = None


class SessionEventPayload(BaseModel):
    """Incoming session event from a tenant."""
    event_type: str = Field(..., min_length=1)  # login/logout/error/department_switch/etc

    # Identity
    user_email: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None

    # Context
    department: Optional[str] = None
    event_data: Optional[dict] = None

    # Error details (if event_type == 'error')
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class BatchPayload(BaseModel):
    """Batch of events for bulk ingestion."""
    queries: Optional[List[QueryEventPayload]] = None
    sessions: Optional[List[SessionEventPayload]] = None


class IngestResponse(BaseModel):
    """Response from ingestion endpoint."""
    success: bool
    event_id: Optional[str] = None
    event_ids: Optional[List[str]] = None
    cognitive_phase: Optional[str] = None
    insights: Optional[List[dict]] = None
    message: Optional[str] = None


class TenantContext(BaseModel):
    """Validated tenant context from API key."""
    tenant_id: str
    tenant_source: str  # 'enterprise' or 'liquid_mirror'
    scopes: List[str]


# =============================================================================
# API KEY AUTHENTICATION
# =============================================================================


@contextmanager
def get_connection():
    """Get connection from pool."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def validate_api_key(api_key: str) -> Optional[TenantContext]:
    """
    Validate API key and return tenant context.
    Returns None if invalid.
    """
    if not api_key or not api_key.startswith("lm_"):
        return None

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT tenant_id, tenant_source, scopes, is_active, expires_at
                FROM liquid_mirror.api_keys
                WHERE key_hash = %s
            """, (key_hash,))

            row = cur.fetchone()

            if not row:
                return None

            # Check if active and not expired
            if not row['is_active']:
                return None

            if row['expires_at'] and row['expires_at'] < datetime.now(timezone.utc):
                return None

            # Update last_used_at
            cur.execute("""
                UPDATE liquid_mirror.api_keys
                SET last_used_at = NOW()
                WHERE key_hash = %s
            """, (key_hash,))
            conn.commit()

            return TenantContext(
                tenant_id=str(row['tenant_id']),
                tenant_source=row['tenant_source'],
                scopes=row['scopes'] or ['ingest']
            )


async def get_tenant(x_api_key: str = Header(..., alias="X-API-Key")) -> TenantContext:
    """
    FastAPI dependency to validate API key and get tenant context.
    """
    tenant = validate_api_key(x_api_key)

    if not tenant:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": "API-Key"}
        )

    return tenant


def require_scope(required_scope: str):
    """
    Factory for scope-checking dependencies.

    Usage:
        @router.post("/admin/thing", dependencies=[Depends(require_scope("admin"))])
    """
    async def check_scope(tenant: TenantContext = Depends(get_tenant)) -> TenantContext:
        if required_scope not in tenant.scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Missing required scope: {required_scope}"
            )
        return tenant
    return check_scope


# =============================================================================
# INGESTION LOGIC
# =============================================================================


def set_tenant_context(conn, tenant_id: str) -> None:
    """Set the RLS tenant context for a connection."""
    with conn.cursor() as cur:
        cur.execute("SELECT liquid_mirror.set_tenant_context(%s)", (tenant_id,))


def ingest_query_event(
    conn,
    tenant_id: str,
    payload: QueryEventPayload,
    mirror=None
) -> tuple[str, Optional[str], Optional[List[dict]]]:
    """
    Ingest a query event into liquid_mirror.query_events.

    Returns: (event_id, cognitive_phase, insights)
    """
    now = datetime.now(timezone.utc)
    cognitive_phase = None
    insights = None

    # If mirror is available, run cognitive analysis
    if mirror and payload.query_embedding:
        try:
            from .mirror.metacognitive_mirror import QueryEvent as MirrorQueryEvent
            import numpy as np

            # Create mirror query event
            mirror_event = MirrorQueryEvent(
                timestamp=now,
                query_text=payload.query_text,
                query_embedding=np.array(payload.query_embedding, dtype=np.float32),
                retrieved_memory_ids=payload.memory_ids_retrieved or [],
                retrieval_scores=payload.retrieval_scores or [],
                execution_time_ms=payload.response_time_ms or 0.0,
                result_count=len(payload.memory_ids_retrieved or []),
                semantic_gate_passed=True
            )

            # Record in mirror
            mirror.record_query(mirror_event)

            # Get real-time insights
            rt_insights = mirror.get_real_time_insights()
            cognitive_phase = rt_insights.get('cognitive_phase')

            # Check for critical insights
            health = mirror.run_health_check()
            if health:
                critical = [h for h in health if h.get('severity') == 'critical']
                if critical:
                    insights = critical

        except Exception as e:
            logger.warning(f"[Ingest] Mirror analysis failed: {e}")

    # Insert into database
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            INSERT INTO liquid_mirror.query_events (
                tenant_id, user_email, session_id, trace_id,
                query_text, query_embedding, department,
                response_time_ms, tokens_input, tokens_output, model_used,
                complexity_score, intent_type, specificity_score,
                temporal_urgency, is_multi_part,
                cognitive_phase, memory_ids_retrieved, retrieval_scores,
                created_at
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s
            )
            RETURNING id
        """, (
            tenant_id, payload.user_email, payload.session_id, payload.trace_id,
            payload.query_text,
            payload.query_embedding if payload.query_embedding else None,
            payload.department,
            payload.response_time_ms, payload.tokens_input, payload.tokens_output,
            payload.model_used,
            payload.complexity_score, payload.intent_type, payload.specificity_score,
            payload.temporal_urgency, payload.is_multi_part,
            cognitive_phase, payload.memory_ids_retrieved, payload.retrieval_scores,
            now
        ))

        result = cur.fetchone()
        event_id = str(result['id'])

    conn.commit()
    logger.info(f"[Ingest] Query event {event_id[:8]} for tenant {tenant_id[:8]}")

    return event_id, cognitive_phase, insights


def ingest_session_event(
    conn,
    tenant_id: str,
    payload: SessionEventPayload
) -> str:
    """
    Ingest a session event into liquid_mirror.session_events.

    Returns: event_id
    """
    now = datetime.now(timezone.utc)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            INSERT INTO liquid_mirror.session_events (
                tenant_id, event_type, user_email, session_id, trace_id,
                department, event_data, error_type, error_message,
                created_at
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s
            )
            RETURNING id
        """, (
            tenant_id, payload.event_type, payload.user_email, payload.session_id,
            payload.trace_id, payload.department,
            psycopg2.extras.Json(payload.event_data) if payload.event_data else None,
            payload.error_type, payload.error_message,
            now
        ))

        result = cur.fetchone()
        event_id = str(result['id'])

    conn.commit()
    logger.info(f"[Ingest] Session event {event_id[:8]} ({payload.event_type}) for tenant {tenant_id[:8]}")

    return event_id


# =============================================================================
# ROUTER
# =============================================================================

ingest_router = APIRouter()


@ingest_router.post("/query", response_model=IngestResponse)
async def ingest_query(
    payload: QueryEventPayload,
    tenant: TenantContext = Depends(get_tenant)
):
    """
    Ingest a query event.

    Requires X-API-Key header with valid API key.
    Returns cognitive phase and any critical insights detected.
    """
    if "ingest" not in tenant.scopes:
        raise HTTPException(status_code=403, detail="Missing 'ingest' scope")

    try:
        # Get mirror instance (optional - graceful degradation)
        mirror = None
        try:
            from .main import get_mirror
            mirror = get_mirror()
        except Exception:
            pass

        with get_connection() as conn:
            set_tenant_context(conn, tenant.tenant_id)
            event_id, phase, insights = ingest_query_event(
                conn, tenant.tenant_id, payload, mirror
            )

        return IngestResponse(
            success=True,
            event_id=event_id,
            cognitive_phase=phase,
            insights=insights
        )

    except Exception as e:
        logger.error(f"[Ingest] Query ingest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@ingest_router.post("/session", response_model=IngestResponse)
async def ingest_session(
    payload: SessionEventPayload,
    tenant: TenantContext = Depends(get_tenant)
):
    """
    Ingest a session event (login, logout, error, etc).

    Requires X-API-Key header with valid API key.
    """
    if "ingest" not in tenant.scopes:
        raise HTTPException(status_code=403, detail="Missing 'ingest' scope")

    try:
        with get_connection() as conn:
            set_tenant_context(conn, tenant.tenant_id)
            event_id = ingest_session_event(conn, tenant.tenant_id, payload)

        return IngestResponse(
            success=True,
            event_id=event_id
        )

    except Exception as e:
        logger.error(f"[Ingest] Session ingest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@ingest_router.post("/batch", response_model=IngestResponse)
async def ingest_batch(
    payload: BatchPayload,
    tenant: TenantContext = Depends(get_tenant)
):
    """
    Batch ingest multiple events.

    Requires X-API-Key header with valid API key.
    More efficient for high-volume ingestion.
    """
    if "ingest" not in tenant.scopes:
        raise HTTPException(status_code=403, detail="Missing 'ingest' scope")

    try:
        event_ids = []
        last_phase = None
        all_insights = []

        # Get mirror instance (optional)
        mirror = None
        try:
            from .main import get_mirror
            mirror = get_mirror()
        except Exception:
            pass

        with get_connection() as conn:
            set_tenant_context(conn, tenant.tenant_id)

            # Process queries
            if payload.queries:
                for q in payload.queries:
                    event_id, phase, insights = ingest_query_event(
                        conn, tenant.tenant_id, q, mirror
                    )
                    event_ids.append(event_id)
                    if phase:
                        last_phase = phase
                    if insights:
                        all_insights.extend(insights)

            # Process sessions
            if payload.sessions:
                for s in payload.sessions:
                    event_id = ingest_session_event(conn, tenant.tenant_id, s)
                    event_ids.append(event_id)

        return IngestResponse(
            success=True,
            event_ids=event_ids,
            cognitive_phase=last_phase,
            insights=all_insights if all_insights else None,
            message=f"Ingested {len(event_ids)} events"
        )

    except Exception as e:
        logger.error(f"[Ingest] Batch ingest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@ingest_router.get("/status")
async def ingest_status(tenant: TenantContext = Depends(get_tenant)):
    """
    Check ingestion status and tenant info.

    Useful for testing API key validity.
    """
    return {
        "status": "ok",
        "tenant_id": tenant.tenant_id,
        "tenant_source": tenant.tenant_source,
        "scopes": tenant.scopes,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

"""
Analytics API Routes - Dashboard data endpoints.

All endpoints require admin access (dept_head or super_user).
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from datetime import datetime, timezone
import logging

# Auth dependency - stub for standalone use, replace with your auth system
try:
    from ..observability_auth import get_admin_user, User
except ImportError:
    # Standalone mode - no auth required
    from dataclasses import dataclass
    from typing import Optional
    
    @dataclass
    class User:
        email: str
        role: str = "admin"
        is_super_user: bool = True
    
    async def get_admin_user() -> User:
        """Stub auth - returns admin user. Replace with real auth in production."""
        return User(email="admin@localhost", role="admin", is_super_user=True)

logger = logging.getLogger(__name__)

analytics_router = APIRouter()

# =============================================================================
# DASHBOARD ENDPOINTS
# =============================================================================

@analytics_router.get("/overview")
def get_analytics_overview(
    user: User = Depends(get_admin_user),
    hours: int = Query(24, ge=1, le=168),  # 1 hour to 7 days
):
    """
    Get dashboard overview stats.
    Returns: active_users, total_queries, avg_response_time, error_rate
    """
    try:
        from .service import get_analytics_service
        analytics = get_analytics_service()
        
        # Department filtering for dept heads
        if user.is_super_user:
            result = analytics.get_overview_stats(hours=hours)
        else:
            result = analytics.get_overview_stats(hours=hours, departments=user.dept_head_for)
        
        logger.info(f"[Analytics] {user.email} accessed overview stats")
        return result
    except Exception as e:
        logger.error(f"Error fetching overview: {e}")
        return {
            "active_users": 0,
            "total_queries": 0,
            "avg_response_time_ms": 0,
            "error_rate_percent": 0,
            "period_hours": hours,
            "error": str(e)
        }


@analytics_router.get("/queries")
def get_queries_over_time(
    user: User = Depends(get_admin_user),
    hours: int = Query(24, ge=1, le=168),
):
    """
    Get query counts grouped by hour.
    For the "Queries by Hour" chart.
    """
    try:
        from .service import get_analytics_service
        analytics = get_analytics_service()
        
        # Department filtering for dept heads
        if user.is_super_user:
            data = analytics.get_queries_by_hour(hours=hours)
        else:
            data = analytics.get_queries_by_hour(hours=hours, departments=user.dept_head_for)
        
        return {
            "period_hours": hours,
            "data": data,
            "filtered_by_department": not user.is_super_user
        }
    except Exception as e:
        logger.error(f"Error fetching queries by hour: {e}")
        return {"period_hours": hours, "data": [], "error": str(e)}


@analytics_router.get("/categories")
def get_category_breakdown(
    user: User = Depends(get_admin_user),
    hours: int = Query(24, ge=1, le=168),
):
    """
    Get query category breakdown.
    For the "Query Categories" pie/bar chart.
    """
    try:
        from .service import get_analytics_service
        analytics = get_analytics_service()
        
        # Department filtering for dept heads
        if user.is_super_user:
            data = analytics.get_category_breakdown(hours=hours)
        else:
            data = analytics.get_category_breakdown(hours=hours, departments=user.dept_head_for)
        
        return {
            "period_hours": hours,
            "data": data,
            "filtered_by_department": not user.is_super_user
        }
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        return {"period_hours": hours, "data": [], "error": str(e)}


@analytics_router.get("/departments")
def get_department_stats(
    user: User = Depends(get_admin_user),
    hours: int = Query(24, ge=1, le=168),
):
    """
    Get per-department statistics.
    For the department heatmap/breakdown.
    """
    try:
        from .service import get_analytics_service
        analytics = get_analytics_service()
        
        # Department filtering - only show stats for accessible departments
        if user.is_super_user:
            data = analytics.get_department_stats(hours=hours)
        else:
            data = analytics.get_department_stats(hours=hours, departments=user.dept_head_for)
        
        return {
            "period_hours": hours,
            "data": data,
            "filtered_by_department": not user.is_super_user,
            "accessible_departments": user.dept_head_for if not user.is_super_user else None
        }
    except Exception as e:
        logger.error(f"Error fetching department stats: {e}")
        return {"period_hours": hours, "data": [], "error": str(e)}


@analytics_router.get("/dashboard")
def get_full_dashboard(
    user: User = Depends(get_admin_user),
    hours: int = Query(24, ge=1, le=168),
    include_errors: bool = Query(True),
    include_realtime: bool = Query(True),
):
    """
    Combined dashboard endpoint - all data in ONE request.

    SYNC function - FastAPI runs in threadpool, doesn't block event loop.
    Uses single DB connection for all queries via connection pool.
    """
    try:
        from .service import get_analytics_service
        analytics = get_analytics_service()

        # Pass department filter for dept heads
        if user.is_super_user:
            data = analytics.get_dashboard_data(
                hours=hours,
                include_errors=include_errors,
                include_realtime=include_realtime
            )
        else:
            data = analytics.get_dashboard_data(
                hours=hours,
                include_errors=include_errors,
                include_realtime=include_realtime,
                departments=user.dept_head_for
            )
        
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        data["filtered_by_department"] = not user.is_super_user
        data["accessible_departments"] = user.dept_head_for if not user.is_super_user else None

        logger.info(f"[Analytics] {user.email} accessed full dashboard (dept filter: {not user.is_super_user})")
        return data
    except Exception as e:
        logger.error(f"Error fetching dashboard: {e}")
        return {
            "overview": {
                "active_users": 0,
                "total_queries": 0,
                "avg_response_time_ms": 0,
                "error_rate_percent": 0,
                "period_hours": hours
            },
            "queries_by_hour": [],
            "categories": [],
            "departments": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }

# =============================================================================
# ERROR ANALYTICS (Phase 4: Hardening Integration)
# =============================================================================

@analytics_router.get("/errors/summary")
def get_error_summary(
    user: User = Depends(get_admin_user),
    hours: int = Query(24, ge=1, le=168),
):
    """Get error summary for dashboard."""
    try:
        from .service import get_analytics_service
        analytics = get_analytics_service()
        return analytics.get_error_summary(hours=hours)
    except Exception as e:
        logger.error(f"Error fetching error summary: {e}")
        return {
            "total_errors": 0,
            "affected_users": 0,
            "affected_endpoints": 0,
            "server_error_rate": 0,
            "period_hours": hours,
            "error": "Failed to fetch error summary"
        }


@analytics_router.get("/errors/by-endpoint")
def get_errors_by_endpoint(
    user: User = Depends(get_admin_user),
    hours: int = Query(24, ge=1, le=168),
):
    """Get error breakdown by endpoint."""
    try:
        from .service import get_analytics_service
        analytics = get_analytics_service()
        return {
            "period_hours": hours,
            "data": analytics.get_errors_by_endpoint(hours=hours)
        }
    except Exception as e:
        logger.error(f"Error fetching errors by endpoint: {e}")
        return {"period_hours": hours, "data": [], "error": "Failed to fetch errors"}


@analytics_router.get("/errors/by-type")
def get_errors_by_type(
    user: User = Depends(get_admin_user),
    hours: int = Query(24, ge=1, le=168),
):
    """Get error breakdown by exception type."""
    try:
        from .service import get_analytics_service
        analytics = get_analytics_service()
        return {
            "period_hours": hours,
            "data": analytics.get_errors_by_type(hours=hours)
        }
    except Exception as e:
        logger.error(f"Error fetching errors by type: {e}")
        return {"period_hours": hours, "data": [], "error": "Failed to fetch errors"}


# =============================================================================
# STUB ENDPOINTS - Return empty data for removed features
# These prevent 404s from old frontend builds still calling these endpoints
# =============================================================================

@analytics_router.get("/realtime")
def get_realtime_sessions(user: User = Depends(get_admin_user)):
    """STUB: Realtime sessions endpoint removed - return empty."""
    return {"sessions": []}


@analytics_router.get("/memory-graph-data")
def get_memory_graph_data(
    user: User = Depends(get_admin_user),
    hours: int = Query(168, ge=1, le=720),
):
    """STUB: Memory graph endpoint removed - return empty."""
    return {"nodes": [], "edges": [], "clusters": []}


# =============================================================================
# LIQUID MIRROR INSIGHTS (Multi-tenant cognitive analytics)
# =============================================================================

@analytics_router.get("/insights/recent")
def get_recent_insights(
    user: User = Depends(get_admin_user),
    hours: int = Query(24, ge=1, le=168),
    severity: Optional[str] = Query(None, description="Filter by severity: info, warning, critical"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Get recent insights from the metacognitive mirror.
    Returns system-generated observations and recommendations.
    """
    try:
        from .service import get_pool
        from contextlib import contextmanager
        from psycopg2.extras import RealDictCursor

        @contextmanager
        def get_connection():
            pool = get_pool()
            conn = pool.getconn()
            try:
                yield conn
            finally:
                pool.putconn(conn)

        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build query with optional severity filter
                query = """
                    SELECT id, tenant_id, insight_type, severity, title, description,
                           metrics, suggested_action, estimated_impact,
                           related_user_email, acknowledged_at, created_at
                    FROM liquid_mirror.insights
                    WHERE created_at > NOW() - INTERVAL '1 hour' * %s
                """
                params = [hours]

                if severity:
                    query += " AND severity = %s"
                    params.append(severity)

                query += " ORDER BY created_at DESC LIMIT %s"
                params.append(limit)

                cur.execute(query, params)
                rows = cur.fetchall()

        return {
            "period_hours": hours,
            "severity_filter": severity,
            "count": len(rows),
            "insights": [
                {
                    "id": str(row['id']),
                    "type": row['insight_type'],
                    "severity": row['severity'],
                    "title": row['title'],
                    "description": row['description'],
                    "metrics": row['metrics'],
                    "suggested_action": row['suggested_action'],
                    "estimated_impact": row['estimated_impact'],
                    "related_user": row['related_user_email'],
                    "acknowledged": row['acknowledged_at'] is not None,
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching insights: {e}")
        return {"period_hours": hours, "count": 0, "insights": [], "error": str(e)}


@analytics_router.get("/insights/critical")
def get_critical_insights(
    user: User = Depends(get_admin_user),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Get unacknowledged critical insights.
    These require immediate attention.
    """
    try:
        from .service import get_pool
        from contextlib import contextmanager
        from psycopg2.extras import RealDictCursor

        @contextmanager
        def get_connection():
            pool = get_pool()
            conn = pool.getconn()
            try:
                yield conn
            finally:
                pool.putconn(conn)

        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, tenant_id, insight_type, title, description,
                           metrics, suggested_action, estimated_impact,
                           related_user_email, created_at
                    FROM liquid_mirror.insights
                    WHERE severity = 'critical'
                      AND acknowledged_at IS NULL
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
                rows = cur.fetchall()

        return {
            "count": len(rows),
            "requires_attention": len(rows) > 0,
            "insights": [
                {
                    "id": str(row['id']),
                    "type": row['insight_type'],
                    "title": row['title'],
                    "description": row['description'],
                    "metrics": row['metrics'],
                    "suggested_action": row['suggested_action'],
                    "estimated_impact": row['estimated_impact'],
                    "related_user": row['related_user_email'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching critical insights: {e}")
        return {"count": 0, "requires_attention": False, "insights": [], "error": str(e)}


@analytics_router.post("/insights/{insight_id}/acknowledge")
def acknowledge_insight(
    insight_id: str,
    user: User = Depends(get_admin_user),
):
    """Mark an insight as acknowledged."""
    try:
        from .service import get_pool
        from contextlib import contextmanager
        from psycopg2.extras import RealDictCursor

        @contextmanager
        def get_connection():
            pool = get_pool()
            conn = pool.getconn()
            try:
                yield conn
            finally:
                pool.putconn(conn)

        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE liquid_mirror.insights
                    SET acknowledged_at = NOW(),
                        acknowledged_by = %s
                    WHERE id = %s
                    RETURNING id, acknowledged_at
                """, (user.email, insight_id))

                row = cur.fetchone()
                conn.commit()

                if not row:
                    raise HTTPException(status_code=404, detail="Insight not found")

        logger.info(f"[Insights] {user.email} acknowledged insight {insight_id}")
        return {
            "success": True,
            "insight_id": insight_id,
            "acknowledged_at": row['acknowledged_at'].isoformat() if row['acknowledged_at'] else None,
            "acknowledged_by": user.email
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging insight: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/cognitive/snapshots")
def get_cognitive_snapshots(
    user: User = Depends(get_admin_user),
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(100, ge=1, le=500),
):
    """
    Get cognitive state snapshots over time.
    Shows how the system's cognitive phase evolved.
    """
    try:
        from .service import get_pool
        from contextlib import contextmanager
        from psycopg2.extras import RealDictCursor

        @contextmanager
        def get_connection():
            pool = get_pool()
            conn = pool.getconn()
            try:
                yield conn
            finally:
                pool.putconn(conn)

        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, phase, phase_confidence, temperature, focus_score,
                           stability_score, access_entropy, query_entropy,
                           drift_magnitude, drift_signal, dominant_topics,
                           query_count_window, user_email, session_id, snapshot_time
                    FROM liquid_mirror.cognitive_snapshots
                    WHERE snapshot_time > NOW() - INTERVAL '1 hour' * %s
                    ORDER BY snapshot_time DESC
                    LIMIT %s
                """, (hours, limit))
                rows = cur.fetchall()

        return {
            "period_hours": hours,
            "count": len(rows),
            "snapshots": [
                {
                    "id": str(row['id']),
                    "phase": row['phase'],
                    "phase_confidence": row['phase_confidence'],
                    "temperature": row['temperature'],
                    "focus_score": row['focus_score'],
                    "stability_score": row['stability_score'],
                    "access_entropy": row['access_entropy'],
                    "query_entropy": row['query_entropy'],
                    "drift_magnitude": row['drift_magnitude'],
                    "drift_signal": row['drift_signal'],
                    "dominant_topics": row['dominant_topics'],
                    "query_count": row['query_count_window'],
                    "user_email": row['user_email'],
                    "session_id": row['session_id'],
                    "timestamp": row['snapshot_time'].isoformat() if row['snapshot_time'] else None
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching cognitive snapshots: {e}")
        return {"period_hours": hours, "count": 0, "snapshots": [], "error": str(e)}


@analytics_router.get("/cognitive/current")
def get_current_cognitive_state(
    user: User = Depends(get_admin_user),
):
    """
    Get the current cognitive state from the MetacognitiveMirror.
    Real-time analysis of system behavior.
    """
    try:
        from .main import get_mirror
        mirror = get_mirror()
        return mirror.get_real_time_insights()
    except Exception as e:
        logger.error(f"Error getting current cognitive state: {e}")
        return {
            "cognitive_phase": "unknown",
            "temperature": 0.0,
            "focus_score": 0.0,
            "error": str(e)
        }

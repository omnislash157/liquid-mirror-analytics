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

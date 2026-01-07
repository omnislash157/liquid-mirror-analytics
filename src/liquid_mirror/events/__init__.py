"""
Event infrastructure for analytics -> cognitive system communication.

Provides:
- Event dataclasses (AnalyticsQueryEvent, AnalyticsSessionEvent, AnalyticsAccessEvent)
- InProcessEventBus for pub/sub within a single process
- Event type constants (EVENT_QUERY, EVENT_SESSION, EVENT_ACCESS)
"""

from .events import (
    AnalyticsQueryEvent,
    AnalyticsSessionEvent,
    AnalyticsAccessEvent,
)

from .bus import (
    EventBus,
    InProcessEventBus,
    get_event_bus,
    reset_event_bus,
    EVENT_QUERY,
    EVENT_SESSION,
    EVENT_ACCESS,
)

__all__ = [
    # Event types
    "AnalyticsQueryEvent",
    "AnalyticsSessionEvent",
    "AnalyticsAccessEvent",
    # Event bus
    "EventBus",
    "InProcessEventBus",
    "get_event_bus",
    "reset_event_bus",
    # Constants
    "EVENT_QUERY",
    "EVENT_SESSION",
    "EVENT_ACCESS",
]

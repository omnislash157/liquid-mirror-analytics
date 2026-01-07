"""Tests for event bus functionality."""

import pytest
from datetime import datetime, timezone

from liquid_mirror.events import (
    get_event_bus,
    reset_event_bus,
    EVENT_QUERY,
    EVENT_SESSION,
    AnalyticsQueryEvent,
    AnalyticsSessionEvent,
)


@pytest.fixture(autouse=True)
def fresh_bus():
    """Reset event bus before each test."""
    reset_event_bus()
    yield
    reset_event_bus()


def test_event_bus_singleton():
    """Event bus should be a singleton."""
    bus1 = get_event_bus()
    bus2 = get_event_bus()
    assert bus1 is bus2


def test_subscribe_and_publish():
    """Basic pub/sub should work."""
    bus = get_event_bus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe(EVENT_QUERY, handler)

    event = AnalyticsQueryEvent(
        timestamp=datetime.now(timezone.utc),
        query_text="test query",
        user_email="test@example.com",
        session_id="test-123",
    )
    bus.publish(EVENT_QUERY, event)

    assert len(received) == 1
    assert received[0].query_text == "test query"


def test_multiple_handlers():
    """Multiple handlers should all receive events."""
    bus = get_event_bus()
    received1 = []
    received2 = []

    bus.subscribe(EVENT_QUERY, lambda e: received1.append(e))
    bus.subscribe(EVENT_QUERY, lambda e: received2.append(e))

    event = AnalyticsQueryEvent(
        timestamp=datetime.now(timezone.utc),
        query_text="test",
        user_email="test@example.com",
        session_id="test-123",
    )
    bus.publish(EVENT_QUERY, event)

    assert len(received1) == 1
    assert len(received2) == 1


def test_handler_isolation():
    """Query handlers should not receive session events."""
    bus = get_event_bus()
    query_received = []
    session_received = []

    bus.subscribe(EVENT_QUERY, lambda e: query_received.append(e))
    bus.subscribe(EVENT_SESSION, lambda e: session_received.append(e))

    session_event = AnalyticsSessionEvent(
        timestamp=datetime.now(timezone.utc),
        event_type="login",
        user_email="test@example.com",
        session_id="test-123",
    )
    bus.publish(EVENT_SESSION, session_event)

    assert len(query_received) == 0
    assert len(session_received) == 1


def test_handler_exception_isolation():
    """One failing handler should not break others."""
    bus = get_event_bus()
    received = []

    def bad_handler(event):
        raise ValueError("I'm broken")

    def good_handler(event):
        received.append(event)

    bus.subscribe(EVENT_QUERY, bad_handler)
    bus.subscribe(EVENT_QUERY, good_handler)

    event = AnalyticsQueryEvent(
        timestamp=datetime.now(timezone.utc),
        query_text="test",
        user_email="test@example.com",
        session_id="test-123",
    )
    bus.publish(EVENT_QUERY, event)  # Should not raise

    assert len(received) == 1


def test_unsubscribe():
    """Unsubscribed handlers should not receive events."""
    bus = get_event_bus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe(EVENT_QUERY, handler)
    bus.unsubscribe(EVENT_QUERY, handler)

    event = AnalyticsQueryEvent(
        timestamp=datetime.now(timezone.utc),
        query_text="test",
        user_email="test@example.com",
        session_id="test-123",
    )
    bus.publish(EVENT_QUERY, event)

    assert len(received) == 0

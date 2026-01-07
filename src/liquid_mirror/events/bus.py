"""
Event bus for decoupled analytics -> mirror communication.

Provides a simple pub/sub mechanism that allows analytics to emit events
without knowing about subscribers, and allows the metacognitive mirror
to receive events without importing analytics code.

Usage:
    from shared.event_bus import get_event_bus

    # Publisher (analytics)
    bus = get_event_bus()
    bus.publish("query", AnalyticsQueryEvent(...))

    # Subscriber (mirror)
    bus = get_event_bus()
    bus.subscribe("query", my_handler)
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Any, Union
from functools import wraps

logger = logging.getLogger(__name__)


class EventBus(ABC):
    """Abstract event bus interface."""

    @abstractmethod
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Event type to subscribe to ('query', 'access', 'session')
            handler: Callback function that receives the event
        """
        pass

    @abstractmethod
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """
        Unsubscribe from events.

        Args:
            event_type: Event type to unsubscribe from
            handler: The handler to remove
        """
        pass

    @abstractmethod
    def publish(self, event_type: str, event: Any) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event_type: Type of event being published
            event: The event object
        """
        pass

    @abstractmethod
    async def publish_async(self, event_type: str, event: Any) -> None:
        """
        Publish an event asynchronously.

        Args:
            event_type: Type of event being published
            event: The event object
        """
        pass


class InProcessEventBus(EventBus):
    """
    Simple in-process event bus for monolith deployment.

    Handlers are called synchronously in publish() and asynchronously
    in publish_async(). Exceptions in handlers are caught and logged
    to prevent one bad handler from breaking the chain.

    Thread-safe for basic usage but not optimized for high concurrency.
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._async_handlers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe a sync handler."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
            logger.debug(f"Subscribed {handler.__name__} to '{event_type}' events")

    def subscribe_async(self, event_type: str, handler: Callable) -> None:
        """Subscribe an async handler."""
        if event_type not in self._async_handlers:
            self._async_handlers[event_type] = []

        if handler not in self._async_handlers[event_type]:
            self._async_handlers[event_type].append(handler)
            logger.debug(f"Subscribed async {handler.__name__} to '{event_type}' events")

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Remove a handler."""
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            logger.debug(f"Unsubscribed {handler.__name__} from '{event_type}' events")

        if event_type in self._async_handlers and handler in self._async_handlers[event_type]:
            self._async_handlers[event_type].remove(handler)

    def publish(self, event_type: str, event: Any) -> None:
        """
        Publish event to sync handlers.

        Catches exceptions to prevent handler failures from breaking the chain.
        """
        handlers = self._handlers.get(event_type, [])

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Event handler {handler.__name__} failed for '{event_type}': {e}",
                    exc_info=True
                )

    async def publish_async(self, event_type: str, event: Any) -> None:
        """
        Publish event to both sync and async handlers.

        Sync handlers run first (in order), then async handlers run concurrently.
        """
        # Run sync handlers first
        self.publish(event_type, event)

        # Run async handlers concurrently
        async_handlers = self._async_handlers.get(event_type, [])
        if async_handlers:
            tasks = []
            for handler in async_handlers:
                tasks.append(self._safe_async_call(handler, event, event_type))

            await asyncio.gather(*tasks)

    async def _safe_async_call(
        self,
        handler: Callable,
        event: Any,
        event_type: str
    ) -> None:
        """Wrap async handler call with exception handling."""
        try:
            await handler(event)
        except Exception as e:
            logger.error(
                f"Async event handler {handler.__name__} failed for '{event_type}': {e}",
                exc_info=True
            )

    def get_handler_count(self, event_type: str) -> int:
        """Get number of handlers for an event type."""
        sync_count = len(self._handlers.get(event_type, []))
        async_count = len(self._async_handlers.get(event_type, []))
        return sync_count + async_count

    def clear(self) -> None:
        """Remove all handlers (useful for testing)."""
        self._handlers.clear()
        self._async_handlers.clear()


# Global singleton instance
_event_bus: EventBus = None


def get_event_bus() -> EventBus:
    """
    Get the global event bus singleton.

    Returns the same instance across the application for consistent
    pub/sub behavior.
    """
    global _event_bus
    if _event_bus is None:
        _event_bus = InProcessEventBus()
        logger.info("Event bus initialized (InProcessEventBus)")
    return _event_bus


def reset_event_bus() -> None:
    """
    Reset the event bus (for testing).

    Clears all handlers and resets the singleton.
    """
    global _event_bus
    if _event_bus is not None:
        _event_bus.clear()
    _event_bus = None


# Event type constants
EVENT_QUERY = "query"
EVENT_ACCESS = "access"
EVENT_SESSION = "session"

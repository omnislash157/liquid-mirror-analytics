"""
Structured Logging Configuration for Liquid Mirror Analytics.

Provides JSON-formatted logging with tenant and trace context.
Compatible with existing logger.info() calls.

Usage:
    from liquid_mirror.logging_config import setup_logging

    # Call once at startup
    setup_logging()

    # Then use logging normally - output will be structured JSON
    logger.info("Query processed", extra={"tenant_id": "uuid", "query_count": 5})
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter that outputs structured logs with context.

    Automatically includes:
    - timestamp (ISO 8601)
    - level
    - logger (module name)
    - message
    - Any extra fields passed to the log call
    """

    # Fields to exclude from extra (already handled or internal)
    RESERVED_ATTRS = {
        'name', 'msg', 'args', 'created', 'filename', 'funcName',
        'levelname', 'levelno', 'lineno', 'module', 'msecs',
        'pathname', 'process', 'processName', 'relativeCreated',
        'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
        'taskName', 'message',
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base structure
        log_dict: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add location for errors/warnings
        if record.levelno >= logging.WARNING:
            log_dict["location"] = f"{record.filename}:{record.lineno}"

        # Extract extra fields
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith('_'):
                try:
                    json.dumps(value)  # Check if JSON serializable
                    log_dict[key] = value
                except (TypeError, ValueError):
                    log_dict[key] = str(value)

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_dict["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_dict, default=str)


class DevFormatter(logging.Formatter):
    """
    Human-readable formatter for development.

    Outputs: timestamp | LEVEL | logger | message [extra fields]
    """

    def format(self, record: logging.LogRecord) -> str:
        # Collect extra fields
        extra_parts = []
        for key, value in record.__dict__.items():
            if key not in StructuredFormatter.RESERVED_ATTRS and not key.startswith('_'):
                extra_parts.append(f"{key}={value}")

        # Build message
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        base = f"{timestamp} | {record.levelname:8} | {record.name} | {record.getMessage()}"

        if extra_parts:
            base += f" [{', '.join(extra_parts)}]"

        # Add exception if present
        if record.exc_info:
            base += f"\n{self.formatException(record.exc_info)}"

        return base


def setup_logging(
    level: int = logging.INFO,
    json_output: bool = True,
    module_levels: Optional[Dict[str, int]] = None,
) -> None:
    """
    Configure structured logging for Liquid Mirror.

    Args:
        level: Default log level (INFO)
        json_output: Use JSON format (True for production, False for dev)
        module_levels: Optional dict of module-specific log levels
                       e.g. {"uvicorn": logging.WARNING}
    """
    # Choose formatter based on output mode
    if json_output:
        formatter = StructuredFormatter()
    else:
        formatter = DevFormatter()

    # Configure handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []  # Clear existing handlers
    root.addHandler(handler)

    # Set module-specific levels (quiet noisy loggers)
    default_module_levels = {
        "uvicorn": logging.WARNING,
        "uvicorn.access": logging.WARNING,
        "httpcore": logging.WARNING,
        "httpx": logging.WARNING,
        "asyncio": logging.WARNING,
    }

    if module_levels:
        default_module_levels.update(module_levels)

    for module, mod_level in default_module_levels.items():
        logging.getLogger(module).setLevel(mod_level)

    # Log startup
    logger = logging.getLogger(__name__)
    mode = "JSON" if json_output else "dev"
    logger.info(f"Logging configured", extra={"mode": mode, "level": logging.getLevelName(level)})


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    This is just logging.getLogger but provided for convenience
    and future extensions (e.g., adding default context).
    """
    return logging.getLogger(name)


# Context helpers for adding tenant/trace info to logs
class LogContext:
    """
    Thread-local context for adding tenant_id, trace_id to all logs.

    Usage:
        with LogContext(tenant_id="abc", trace_id="xyz"):
            logger.info("This will include tenant_id and trace_id")
    """

    _context: Dict[str, Any] = {}

    @classmethod
    def set(cls, **kwargs) -> None:
        """Set context values."""
        cls._context.update(kwargs)

    @classmethod
    def clear(cls) -> None:
        """Clear all context."""
        cls._context.clear()

    @classmethod
    def get(cls) -> Dict[str, Any]:
        """Get current context."""
        return cls._context.copy()

    def __init__(self, **kwargs):
        self._saved = {}
        self._new_keys = []
        self._kwargs = kwargs

    def __enter__(self):
        for key, value in self._kwargs.items():
            if key in self._context:
                self._saved[key] = self._context[key]
            else:
                self._new_keys.append(key)
            self._context[key] = value
        return self

    def __exit__(self, *args):
        for key in self._new_keys:
            self._context.pop(key, None)
        self._context.update(self._saved)


class ContextualAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes LogContext in extra.

    Usage:
        logger = ContextualAdapter(logging.getLogger(__name__))
        LogContext.set(tenant_id="abc")
        logger.info("This includes tenant_id automatically")
    """

    def process(self, msg, kwargs):
        extra = kwargs.get('extra', {})
        extra.update(LogContext.get())
        kwargs['extra'] = extra
        return msg, kwargs


def get_contextual_logger(name: str) -> ContextualAdapter:
    """Get a logger that automatically includes LogContext."""
    return ContextualAdapter(logging.getLogger(name), {})

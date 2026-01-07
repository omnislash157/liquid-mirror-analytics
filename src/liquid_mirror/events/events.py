"""
Event contracts for analytics -> metacognitive mirror communication.

These dataclasses define the event schema that flows from the analytics
service to the metacognitive mirror. The mirror consumes these events
to track cognitive patterns without direct coupling to analytics.

Usage:
    from shared.events import AnalyticsQueryEvent

    event = AnalyticsQueryEvent(
        timestamp=datetime.now(),
        query_text="What are the Q4 projections?",
        query_embedding=embedding_vector,
        ...
    )
    event_bus.publish("query", event)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class AnalyticsQueryEvent:
    """
    Emitted by Analytics Service when a query is logged.

    Maps to MetacognitiveMirror.QueryEvent for cognitive tracking.
    Contains both the raw query data and computed heuristics.
    """
    timestamp: datetime
    query_text: str
    user_email: str
    session_id: str

    # Embedding (pre-computed by analytics or retriever)
    query_embedding: Optional[NDArray[np.float32]] = None

    # Department/context
    department: Optional[str] = None

    # Heuristics from analytics engine
    complexity_score: float = 0.0
    intent_type: str = "unknown"
    specificity_score: float = 0.0
    temporal_urgency: Optional[str] = None

    # Retrieval results (populated after memory search)
    retrieved_memory_ids: List[str] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    execution_time_ms: float = 0.0
    result_count: int = 0
    semantic_gate_passed: bool = True

    # Response metadata
    response_time_ms: float = 0.0
    response_length: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    model_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "query_text": self.query_text,
            "user_email": self.user_email,
            "session_id": self.session_id,
            "department": self.department,
            "complexity_score": self.complexity_score,
            "intent_type": self.intent_type,
            "specificity_score": self.specificity_score,
            "temporal_urgency": self.temporal_urgency,
            "retrieved_memory_ids": self.retrieved_memory_ids,
            "retrieval_scores": self.retrieval_scores,
            "execution_time_ms": self.execution_time_ms,
            "result_count": self.result_count,
            "semantic_gate_passed": self.semantic_gate_passed,
            "response_time_ms": self.response_time_ms,
            "response_length": self.response_length,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "model_used": self.model_used,
        }


@dataclass
class AnalyticsAccessEvent:
    """
    Emitted when memories are accessed during retrieval.

    Maps to MetacognitiveMirror.MemoryAccessEvent for thermodynamics tracking.
    Enables memory temperature calculation and co-access graph building.
    """
    timestamp: datetime
    memory_id: str
    access_type: str  # 'retrieval', 'update', 'creation'
    access_score: float  # Retrieval similarity score

    # Context
    query_context: Optional[str] = None
    user_email: Optional[str] = None
    session_id: Optional[str] = None

    # Co-access (other memories retrieved in same query)
    co_accessed_memories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "memory_id": self.memory_id,
            "access_type": self.access_type,
            "access_score": self.access_score,
            "query_context": self.query_context,
            "user_email": self.user_email,
            "session_id": self.session_id,
            "co_accessed_memories": self.co_accessed_memories,
        }


@dataclass
class AnalyticsSessionEvent:
    """
    Emitted for session lifecycle events.

    Tracks login, logout, department switches, and errors for
    session-level analytics and anomaly detection.
    """
    timestamp: datetime
    event_type: str  # 'login', 'logout', 'dept_switch', 'error'
    user_email: str
    session_id: str

    # Optional context
    department: Optional[str] = None
    from_department: Optional[str] = None  # For dept_switch
    to_department: Optional[str] = None    # For dept_switch
    error_type: Optional[str] = None       # For error events
    error_message: Optional[str] = None    # For error events
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "user_email": self.user_email,
            "session_id": self.session_id,
            "department": self.department,
            "from_department": self.from_department,
            "to_department": self.to_department,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }

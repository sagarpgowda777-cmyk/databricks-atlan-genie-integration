from __future__ import annotations

import threading
from dataclasses import dataclass, field

from langchain_core.chat_history import InMemoryChatMessageHistory


@dataclass
class SessionMemory:
    history: InMemoryChatMessageHistory = field(default_factory=InMemoryChatMessageHistory)


class MemoryStore:
    """Process-local session memory; replace with an external store for multi-replica Apps."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemory] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str) -> SessionMemory:
        with self._lock:
            return self._sessions.setdefault(session_id, SessionMemory())

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


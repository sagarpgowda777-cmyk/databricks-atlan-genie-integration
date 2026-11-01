from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

import psycopg
from databricks.sdk import WorkspaceClient

from .config import Settings

logger = logging.getLogger(__name__)
_PRESENTATION_PREFIX = "__KEYSTONE_PRESENTATION_V1__:"


@dataclass
class SessionContext:
    session_id: str
    intent_summary: str = ""
    intent_graph: dict[str, Any] = field(default_factory=lambda: {"nodes": [], "edges": []})
    genie_conversations: dict[str, str] = field(default_factory=dict)
    active_space_ids: list[str] = field(default_factory=list)


@dataclass
class SessionSummary:
    session_id: str
    title: str
    updated_at: datetime


class SessionContextStore:
    """Persists per-session intent/routing state in Lakebase Postgres, next to the space registry."""

    def __init__(self, settings: Settings, connection_factory: Callable[..., Any] = psycopg.connect):
        self.settings = settings
        self._connection_factory = connection_factory
        self._workspace = WorkspaceClient()

    def _generate_password(self) -> str:
        credential = self._workspace.postgres.generate_database_credential(endpoint=self.settings.pg_endpoint)
        return credential.token

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        connection = self._connection_factory(
            host=self.settings.pg_host,
            port=self.settings.pg_port,
            dbname=self.settings.pg_database,
            user=self.settings.pg_user,
            password=self._generate_password(),
            sslmode=self.settings.pg_sslmode,
        )
        try:
            with connection.cursor() as cursor:
                yield cursor
            connection.commit()
        finally:
            connection.close()

    def ensure_table(self) -> None:
        with self._cursor() as cursor:
            cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS {self.settings.session_context_table} (
                    session_id TEXT PRIMARY KEY,
                    intent_summary TEXT NOT NULL DEFAULT '',
                    intent_graph JSONB NOT NULL DEFAULT '{{"nodes": [], "edges": []}}',
                    genie_conversations JSONB NOT NULL DEFAULT '{{}}',
                    active_space_ids JSONB NOT NULL DEFAULT '[]',
                    updated_at TIMESTAMPTZ NOT NULL
                )"""
            )

    def load(self, session_id: str) -> SessionContext:
        try:
            with self._cursor() as cursor:
                cursor.execute(
                    f"SELECT intent_summary, intent_graph, genie_conversations, active_space_ids "
                    f"FROM {self.settings.session_context_table} WHERE session_id = %s",
                    [session_id],
                )
                row = cursor.fetchone()
        except Exception:
            logger.exception("Failed to load session context for %s; continuing without persisted state", session_id)
            return SessionContext(session_id=session_id)
        if not row:
            return SessionContext(session_id=session_id)
        intent_summary, intent_graph, genie_conversations, active_space_ids = row
        return SessionContext(
            session_id=session_id,
            intent_summary=intent_summary or "",
            intent_graph=intent_graph or {"nodes": [], "edges": []},
            genie_conversations=genie_conversations or {},
            active_space_ids=active_space_ids or [],
        )

    def save(self, context: SessionContext) -> None:
        now = datetime.now(timezone.utc)
        try:
            with self._cursor() as cursor:
                cursor.execute(
                    f"""INSERT INTO {self.settings.session_context_table}
                        (session_id, intent_summary, intent_graph, genie_conversations, active_space_ids, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (session_id) DO UPDATE SET
                            intent_summary = EXCLUDED.intent_summary,
                            intent_graph = EXCLUDED.intent_graph,
                            genie_conversations = EXCLUDED.genie_conversations,
                            active_space_ids = EXCLUDED.active_space_ids,
                            updated_at = EXCLUDED.updated_at""",
                    [
                        context.session_id,
                        context.intent_summary,
                        json.dumps(context.intent_graph),
                        json.dumps(context.genie_conversations),
                        json.dumps(context.active_space_ids),
                        now,
                    ],
                )
        except Exception:
            logger.exception("Failed to save session context for %s; state will not persist this turn", context.session_id)

    def ensure_chat_tables(self) -> None:
        with self._cursor() as cursor:
            cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS {self.settings.chat_sessions_table} (
                    session_id TEXT PRIMARY KEY,
                    user_email TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )"""
            )
            cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS {self.settings.chat_messages_table} (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES {self.settings.chat_sessions_table}(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )"""
            )

    def record_turn(
        self, session_id: str, user_email: str, question: str, answer: str, intent_eligible: bool = True,
        presentation: dict[str, Any] | None = None,
    ) -> None:
        """Persists one human/ai message pair, touches the owning session (creating it with a
        title from the first question if new), and prunes that user's sessions down to the
        configured retention limit so the picker never grows unbounded.
        """
        now = datetime.now(timezone.utc)
        stored_answer = answer
        if presentation:
            stored_answer = _PRESENTATION_PREFIX + json.dumps({"content": answer, **presentation})
        try:
            with self._cursor() as cursor:
                cursor.execute(
                    f"""INSERT INTO {self.settings.chat_sessions_table}
                        (session_id, user_email, title, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (session_id) DO UPDATE SET updated_at = EXCLUDED.updated_at""",
                    [session_id, user_email, question[:120], now, now],
                )
                cursor.execute(
                    f"""INSERT INTO {self.settings.chat_messages_table}
                        (session_id, role, content, created_at, intent_eligible)
                        VALUES (%s, %s, %s, %s, %s), (%s, %s, %s, %s, %s)""",
                    [
                        session_id, "human", question, now, intent_eligible,
                        session_id, "ai", stored_answer, now, intent_eligible,
                    ],
                )
                cursor.execute(
                    f"""DELETE FROM {self.settings.chat_sessions_table}
                        WHERE user_email = %s AND session_id NOT IN (
                            SELECT session_id FROM {self.settings.chat_sessions_table}
                            WHERE user_email = %s ORDER BY updated_at DESC LIMIT %s
                        )""",
                    [user_email, user_email, self.settings.max_sessions_per_user],
                )
        except Exception:
            logger.exception("Failed to persist chat turn for session %s; conversation will not be recoverable", session_id)

    def list_sessions(self, user_email: str) -> list[SessionSummary]:
        try:
            with self._cursor() as cursor:
                cursor.execute(
                    f"""SELECT session_id, title, updated_at FROM {self.settings.chat_sessions_table}
                        WHERE user_email = %s ORDER BY updated_at DESC LIMIT %s""",
                    [user_email, self.settings.max_sessions_per_user],
                )
                rows = cursor.fetchall()
        except Exception:
            logger.exception("Failed to list chat sessions for %s", user_email)
            return []
        return [SessionSummary(session_id=row[0], title=row[1], updated_at=row[2]) for row in rows]

    @staticmethod
    def _decode_message(role: str, content: str) -> dict[str, Any]:
        if role == "ai" and content.startswith(_PRESENTATION_PREFIX):
            try:
                payload = json.loads(content[len(_PRESENTATION_PREFIX):])
                if isinstance(payload, dict) and isinstance(payload.get("content"), str):
                    return {"role": role, **payload}
            except (TypeError, ValueError):
                logger.warning("Could not decode stored assistant presentation payload")
        return {"role": role, "content": content}

    def load_message_records(self, session_id: str) -> list[dict[str, Any]]:
        try:
            with self._cursor() as cursor:
                cursor.execute(
                    f"""SELECT role, content FROM {self.settings.chat_messages_table}
                        WHERE session_id = %s ORDER BY id ASC""",
                    [session_id],
                )
                return [self._decode_message(row[0], row[1]) for row in cursor.fetchall()]
        except Exception:
            logger.exception("Failed to load chat messages for session %s", session_id)
            return []

    def load_messages(self, session_id: str, intent_only: bool = False) -> list[tuple[str, str]]:
        try:
            with self._cursor() as cursor:
                intent_filter = " AND intent_eligible = TRUE" if intent_only else ""
                cursor.execute(
                    f"""SELECT role, content FROM {self.settings.chat_messages_table}
                        WHERE session_id = %s{intent_filter} ORDER BY id ASC""",
                    [session_id],
                )
                return [
                    (row[0], self._decode_message(row[0], row[1])["content"])
                    for row in cursor.fetchall()
                ]
        except Exception:
            logger.exception("Failed to load chat messages for session %s", session_id)
            return []

    @staticmethod
    def apply_patch(graph: dict[str, Any], patch: dict[str, Any] | None) -> dict[str, Any]:
        """Merges an incremental {add_nodes, add_edges} patch into the persisted intent graph.

        Nodes are deduped by id: a repeat mention of the same entity/domain/table across turns
        updates `last_seen_turn` and unions list-valued properties (`space_ids`, `aliases`) onto
        the existing node instead of creating a duplicate, so intent summaries from different
        questions in the same session accumulate onto one connected graph.
        """
        if not patch:
            return graph
        nodes = list(graph.get("nodes", []))
        edges = list(graph.get("edges", []))
        nodes_by_id = {node.get("id"): node for node in nodes if isinstance(node, dict)}
        for node in patch.get("add_nodes", []) or []:
            if not isinstance(node, dict) or not node.get("id"):
                continue
            existing = nodes_by_id.get(node["id"])
            if existing is None:
                nodes.append(node)
                nodes_by_id[node["id"]] = node
                continue
            for list_field in ("space_ids", "aliases"):
                if node.get(list_field):
                    merged = list(dict.fromkeys([*existing.get(list_field, []), *node[list_field]]))
                    existing[list_field] = merged
            if node.get("last_seen_turn") is not None:
                existing["last_seen_turn"] = max(existing.get("last_seen_turn") or 0, node["last_seen_turn"])
            for scalar_field in ("confidence", "atlan_guid", "question_text"):
                if node.get(scalar_field) is not None:
                    existing[scalar_field] = node[scalar_field]
        existing_edges = {(edge.get("from"), edge.get("to"), edge.get("relation")) for edge in edges if isinstance(edge, dict)}
        for edge in patch.get("add_edges", []) or []:
            if isinstance(edge, dict):
                key = (edge.get("from"), edge.get("to"), edge.get("relation"))
                if key not in existing_edges:
                    edges.append(edge)
                    existing_edges.add(key)
        return {"nodes": nodes, "edges": edges}

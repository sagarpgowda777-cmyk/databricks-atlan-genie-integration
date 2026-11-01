"""Per-user, unbounded (not pruned like chat_sessions) history of every turn's token/cost usage,
so a user can see their own full run history and totals -- never another user's."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

import psycopg
from databricks.sdk import WorkspaceClient

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class UsageEntry:
    question: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None
    created_at: datetime


@dataclass
class UsageSummary:
    question_count: int
    total_tokens: int
    total_cost_usd: float | None


class UsageHistoryStore:
    """Persists one row per answered turn in Lakebase Postgres, keyed by user_email."""

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
                f"""CREATE TABLE IF NOT EXISTS {self.settings.usage_history_table} (
                    id BIGSERIAL PRIMARY KEY,
                    user_email TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL
                )"""
            )

    def record(
        self,
        user_email: str,
        session_id: str,
        question: str,
        token_usage: dict[str, int] | None,
        estimated_cost_usd: float | None,
    ) -> None:
        if not token_usage:
            return
        try:
            with self._cursor() as cursor:
                cursor.execute(
                    f"""INSERT INTO {self.settings.usage_history_table}
                        (user_email, session_id, question, prompt_tokens, completion_tokens, total_tokens,
                         estimated_cost_usd, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        user_email, session_id, question[:2000],
                        token_usage.get("prompt_tokens", 0), token_usage.get("completion_tokens", 0),
                        token_usage.get("total_tokens", 0), estimated_cost_usd, datetime.now(timezone.utc),
                    ],
                )
        except Exception:
            logger.exception("Failed to record usage history for %s; this turn's usage will be missing", user_email)

    def list_for_user(self, user_email: str, limit: int | None = 200) -> list[UsageEntry]:
        try:
            with self._cursor() as cursor:
                limit_clause = " LIMIT %s" if limit is not None else ""
                parameters: list[Any] = [user_email]
                if limit is not None:
                    parameters.append(limit)
                cursor.execute(
                    f"""SELECT question, prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd, created_at
                        FROM {self.settings.usage_history_table}
                        WHERE user_email = %s ORDER BY created_at DESC{limit_clause}""",
                    parameters,
                )
                rows = cursor.fetchall()
        except Exception:
            logger.exception("Failed to list usage history for %s", user_email)
            return []
        return [
            UsageEntry(
                question=row[0], prompt_tokens=row[1], completion_tokens=row[2], total_tokens=row[3],
                estimated_cost_usd=row[4], created_at=row[5],
            )
            for row in rows
        ]

    def summary_for_user(self, user_email: str) -> UsageSummary:
        try:
            with self._cursor() as cursor:
                cursor.execute(
                    f"""SELECT COUNT(*), COALESCE(SUM(total_tokens), 0), SUM(estimated_cost_usd)
                        FROM {self.settings.usage_history_table} WHERE user_email = %s""",
                    [user_email],
                )
                row = cursor.fetchone()
        except Exception:
            logger.exception("Failed to summarize usage history for %s", user_email)
            return UsageSummary(question_count=0, total_tokens=0, total_cost_usd=None)
        return UsageSummary(question_count=row[0] or 0, total_tokens=row[1] or 0, total_cost_usd=row[2])

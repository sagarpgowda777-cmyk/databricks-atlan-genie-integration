from __future__ import annotations

import json
import logging
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

import psycopg
from databricks.sdk import WorkspaceClient

from .config import Settings
from .models import Skill, SkillCreate, SkillUpdate

logger = logging.getLogger(__name__)


class SkillError(RuntimeError):
    pass


class SkillNotFound(SkillError):
    pass


class SkillLimitExceeded(SkillError):
    pass


class SkillStore:
    """Persists user-defined routing skills (learned question -> Genie space mappings) in
    Lakebase Postgres, so repeat questions can skip the full routing-catalog LLM call entirely.
    """

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
                f"""CREATE TABLE IF NOT EXISTS {self.settings.skills_table} (
                    skill_id TEXT PRIMARY KEY,
                    owner_email TEXT NOT NULL,
                    name TEXT NOT NULL,
                    trigger_phrases JSONB NOT NULL DEFAULT '[]',
                    space_ids JSONB NOT NULL DEFAULT '[]',
                    visibility TEXT NOT NULL DEFAULT 'private',
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )"""
            )

    _COLUMNS = "skill_id, owner_email, name, trigger_phrases, space_ids, visibility, created_at, updated_at"

    @classmethod
    def _to_skill(cls, row: tuple) -> Skill:
        skill_id, owner_email, name, trigger_phrases, space_ids, visibility, created_at, updated_at = row
        return Skill(
            skill_id=skill_id, owner_email=owner_email, name=name,
            trigger_phrases=trigger_phrases or [], space_ids=space_ids or [],
            visibility=visibility, created_at=created_at, updated_at=updated_at,
        )

    def _storage_error(self, action: str, exc: Exception) -> SkillError:
        detail = str(exc).lower()
        if "does not exist" in detail or "undefinedtable" in detail:
            return SkillError(
                f"Cannot {action} skills because {self.settings.skills_table} is not configured. "
                "Create the skills table using the README setup SQL."
            )
        if "permission denied" in detail or "insufficientprivilege" in detail:
            return SkillError(
                f"Cannot {action} skills because the app database role lacks access to "
                f"{self.settings.skills_table}. Grant SELECT, INSERT, UPDATE, and DELETE to the app PGUSER role."
            )
        return SkillError(f"Cannot {action} skills because the skill store is unavailable.")

    def list_for_user(self, user_email: str) -> list[Skill]:
        """Every skill the user owns, plus every shared skill from anyone (promoted skills are
        meant to be reused by the whole team)."""
        try:
            with self._cursor() as cursor:
                cursor.execute(
                    f"""SELECT {self._COLUMNS} FROM {self.settings.skills_table}
                        WHERE owner_email = %s OR visibility = 'shared'
                        ORDER BY updated_at DESC""",
                    [user_email],
                )
                rows = cursor.fetchall()
        except Exception as exc:
            logger.exception("Failed to list skills for %s", user_email)
            raise self._storage_error("load", exc) from exc
        return [self._to_skill(row) for row in rows]

    def _count_owned(self, cursor: Any, user_email: str) -> int:
        cursor.execute(
            f"SELECT COUNT(*) FROM {self.settings.skills_table} WHERE owner_email = %s", [user_email],
        )
        return cursor.fetchone()[0]

    def create(self, user_email: str, payload: SkillCreate) -> Skill:
        now = datetime.now(timezone.utc)
        skill_id = str(uuid.uuid4())
        try:
            with self._cursor() as cursor:
                if self._count_owned(cursor, user_email) >= self.settings.max_skills_per_user:
                    raise SkillLimitExceeded(
                        f"You already have {self.settings.max_skills_per_user} skills; delete one before adding another."
                    )
                cursor.execute(
                    f"""INSERT INTO {self.settings.skills_table}
                        (skill_id, owner_email, name, trigger_phrases, space_ids, visibility, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        skill_id, user_email, payload.name, json.dumps(payload.trigger_phrases),
                        json.dumps(payload.space_ids), payload.visibility, now, now,
                    ],
                )
            return self.get(skill_id)
        except SkillError:
            raise
        except Exception as exc:
            logger.exception("Failed to create skill for %s", user_email)
            raise self._storage_error("create", exc) from exc

    def get(self, skill_id: str) -> Skill:
        try:
            with self._cursor() as cursor:
                cursor.execute(
                    f"SELECT {self._COLUMNS} FROM {self.settings.skills_table} WHERE skill_id = %s", [skill_id],
                )
                row = cursor.fetchone()
                if not row:
                    raise SkillNotFound(skill_id)
                return self._to_skill(row)
        except SkillError:
            raise
        except Exception as exc:
            logger.exception("Failed to load skill %s", skill_id)
            raise self._storage_error("load", exc) from exc

    def update(self, skill_id: str, user_email: str, payload: SkillUpdate) -> Skill:
        try:
            current = self.get(skill_id)
            if current.owner_email != user_email:
                raise SkillError("Only the skill's owner can edit or promote it.")
            changes = payload.model_dump(exclude_unset=True)
            if not changes:
                return current
            assignments, values = [], []
            for field, value in changes.items():
                assignments.append(f"{field} = %s")
                values.append(json.dumps(value) if field in ("trigger_phrases", "space_ids") else value)
            assignments.append("updated_at = %s")
            values.extend([datetime.now(timezone.utc), skill_id])
            with self._cursor() as cursor:
                cursor.execute(
                    f"UPDATE {self.settings.skills_table} SET {', '.join(assignments)} WHERE skill_id = %s", values,
                )
            return self.get(skill_id)
        except SkillError:
            raise
        except Exception as exc:
            logger.exception("Failed to update skill %s for %s", skill_id, user_email)
            raise self._storage_error("update", exc) from exc

    def delete(self, skill_id: str, user_email: str) -> None:
        try:
            current = self.get(skill_id)
            if current.owner_email != user_email:
                raise SkillError("Only the skill's owner can delete it.")
            with self._cursor() as cursor:
                cursor.execute(f"DELETE FROM {self.settings.skills_table} WHERE skill_id = %s", [skill_id])
        except SkillError:
            raise
        except Exception as exc:
            logger.exception("Failed to delete skill %s for %s", skill_id, user_email)
            raise self._storage_error("delete", exc) from exc

    _STOPWORDS = {
        "a", "an", "the", "of", "for", "me", "you", "can", "could", "please", "provide", "give",
        "show", "get", "tell", "about", "data", "and", "or", "to", "in", "on", "with", "this",
        "that", "it", "its", "what", "which", "who", "is", "are", "from",
    }

    @classmethod
    def _words(cls, text: str) -> set[str]:
        return {
            word for word in re.findall(r"[A-Za-z0-9_]+", text.lower())
            if len(word) > 2 and word not in cls._STOPWORDS
        }

    @staticmethod
    def command_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    @classmethod
    def question_for_skill(cls, question: str, skill: Skill) -> str:
        command = re.escape(cls.command_name(skill.name))
        remaining = re.sub(rf"^/{command}(?:\s+|$)", "", question.strip(), count=1, flags=re.IGNORECASE)
        return remaining.strip() or skill.trigger_phrases[0]

    def match(self, question: str, user_email: str) -> Skill | None:
        """Best keyword-overlap match against the user's own + shared skills, or None. Pure
        string matching -- no LLM call -- so this never adds routing latency/token cost.
        """
        skills = self.list_for_user(user_email)
        command_match = re.match(r"^/([a-z0-9-]+)(?:\s|$)", question.strip(), re.IGNORECASE)
        if command_match:
            command = command_match.group(1).lower()
            for skill in skills:
                if self.command_name(skill.name) == command:
                    return skill

        question_words = self._words(question)
        if not question_words:
            return None
        best: tuple[float, Skill] | None = None
        for skill in skills:
            for phrase in skill.trigger_phrases:
                phrase_words = self._words(phrase)
                if not phrase_words:
                    continue
                overlap = question_words & phrase_words
                if not overlap:
                    continue
                # Each saved phrase is an independent example. Combining every phrase would
                # dilute a complete match whenever the skill has several unrelated examples.
                score = len(overlap) / len(phrase_words)
                if score >= 0.5 and (best is None or score > best[0]):
                    best = (score, skill)
        return best[1] if best else None

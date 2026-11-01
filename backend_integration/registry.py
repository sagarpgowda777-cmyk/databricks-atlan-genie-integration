from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

import psycopg
from databricks.sdk import WorkspaceClient

from .config import Settings
from .models import (
    SPACE_DESCRIPTION_MAX_LENGTH,
    SPACE_DISPLAY_NAME_MAX_LENGTH,
    SPACE_QUALIFIED_NAME_MAX_LENGTH,
    Space,
    SpaceUpdate,
)


class RegistryError(RuntimeError):
    pass


class SpaceNotFound(RegistryError):
    pass


_ATLAN_TEXT_LIMITS = {
    "display_name": SPACE_DISPLAY_NAME_MAX_LENGTH,
    "description": SPACE_DESCRIPTION_MAX_LENGTH,
    "atlan_qualified_name": SPACE_QUALIFIED_NAME_MAX_LENGTH,
}


def _bounded_text(value: Any, max_length: int) -> str:
    return str(value or "")[:max_length]


class PostgresRegistry:
    """CRUD repository backed by a Lakebase Postgres database."""

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
                f"""CREATE TABLE IF NOT EXISTS {self.settings.registry_table} (
                    space_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    description TEXT,
                    instructions TEXT,
                    connection_params TEXT,
                    enabled BOOLEAN NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual',
                    atlan_qualified_name TEXT,
                    last_synced_at TIMESTAMPTZ,
                    deployment_environment TEXT,
                    workspace_name TEXT,
                    agent_owner TEXT,
                    certificate_status TEXT,
                    domains TEXT NOT NULL DEFAULT '[]',
                    tags TEXT,
                    connector TEXT,
                    readme TEXT
                )"""
            )

    _COLUMNS = (
        "space_id, display_name, description, instructions, connection_params, enabled, "
        "created_by, created_at, updated_at, source, atlan_qualified_name, last_synced_at, "
        "deployment_environment, workspace_name, agent_owner, certificate_status, domains, tags, connector, readme"
    )

    def list_spaces(self, enabled_only: bool = False) -> list[Space]:
        where = " WHERE enabled = TRUE" if enabled_only else ""
        with self._cursor() as cursor:
            cursor.execute(
                f"SELECT {self._COLUMNS} FROM {self.settings.registry_table}{where} "
                "ORDER BY display_name"
            )
            columns = [item[0] for item in cursor.description]
            return [self._to_space(dict(zip(columns, row))) for row in cursor.fetchall()]

    def get(self, space_id: str) -> Space:
        with self._cursor() as cursor:
            cursor.execute(
                f"SELECT {self._COLUMNS} FROM {self.settings.registry_table} "
                "WHERE space_id = %s", [space_id]
            )
            row = cursor.fetchone()
            if not row:
                raise SpaceNotFound(space_id)
            return self._to_space(dict(zip([item[0] for item in cursor.description], row)))

    def update(self, space_id: str, payload: SpaceUpdate) -> Space:
        current = self.get(space_id)
        changes = payload.model_dump(exclude_unset=True)
        if not changes:
            return current
        assignments, values = [], []
        for field, value in changes.items():
            assignments.append(f"{field} = %s")
            values.append(json.dumps(value) if field == "connection_params" else value)
        assignments.append("updated_at = %s")
        values.extend([datetime.now(timezone.utc), space_id])
        with self._cursor() as cursor:
            cursor.execute(
                f"UPDATE {self.settings.registry_table} SET {', '.join(assignments)} WHERE space_id = %s",
                values,
            )
        return self.get(space_id)

    def delete(self, space_id: str) -> None:
        self.get(space_id)
        with self._cursor() as cursor:
            cursor.execute(f"DELETE FROM {self.settings.registry_table} WHERE space_id = %s", [space_id])

    def create_genie_space(
        self,
        space_id: str,
        display_name: str,
        description: str,
        actor: str,
        warehouse_id: str,
        table_names: list[str],
    ) -> Space:
        now = datetime.now(timezone.utc)
        with self._cursor() as cursor:
            cursor.execute(
                f"""INSERT INTO {self.settings.registry_table}
                    (space_id, display_name, description, instructions, connection_params,
                     enabled, created_by, created_at, updated_at, source, deployment_environment,
                    workspace_name, agent_owner, certificate_status, domains, tags, connector, readme)
                    VALUES (%s, %s, %s, '', %s, FALSE, %s, %s, %s, 'manual', 'DEV', %s, %s,
                        'PENDING', '[]', '[]', 'databricks', '')
                    ON CONFLICT (space_id) DO NOTHING""",
                [
                    space_id,
                    display_name,
                    description,
                    json.dumps({"warehouse_id": warehouse_id, "tables": table_names}),
                    actor,
                    now,
                    now,
                    self._workspace.config.host,
                    actor,
                ],
            )
        return self.get(space_id)

    def sync_from_atlan(self, apps: list[dict], actor: str) -> dict[str, int]:
        """Upsert AIApplication assets discovered from Atlan into the registry.

        Only Atlan-owned fields (display_name, description, atlan_qualified_name) are
        overwritten on an existing row -- instructions/enabled/connection_params set by an
        admin are left untouched. Apps with no resolvable space_id are skipped (can't be
        routed to). Returns counts: added, updated, skipped_no_space_id, total_seen.
        """
        now = datetime.now(timezone.utc)
        candidates = [app for app in apps if app.get("space_id")]
        with self._cursor() as cursor:
            cursor.execute(f"SELECT space_id FROM {self.settings.registry_table}")
            existing_ids = {row[0] for row in cursor.fetchall()}
            for app in candidates:
                cursor.execute(
                    f"""INSERT INTO {self.settings.registry_table}
                        (space_id, display_name, description, instructions, connection_params,
                         enabled, created_by, created_at, updated_at, source, atlan_qualified_name,
                         last_synced_at, deployment_environment, workspace_name, agent_owner,
                         certificate_status, domains, tags, connector, readme)
                        VALUES (%s, %s, %s, '', '{{}}', TRUE, %s, %s, %s, 'atlan', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (space_id) DO UPDATE SET
                            display_name = EXCLUDED.display_name,
                            description = EXCLUDED.description,
                            atlan_qualified_name = EXCLUDED.atlan_qualified_name,
                            source = 'atlan',
                            updated_at = EXCLUDED.updated_at,
                            last_synced_at = EXCLUDED.last_synced_at,
                            deployment_environment = EXCLUDED.deployment_environment,
                            workspace_name = EXCLUDED.workspace_name,
                            agent_owner = EXCLUDED.agent_owner,
                            certificate_status = EXCLUDED.certificate_status,
                            domains = EXCLUDED.domains,
                            tags = EXCLUDED.tags,
                            connector = EXCLUDED.connector,
                            readme = EXCLUDED.readme""",
                    [app["space_id"],
                     _bounded_text(app["name"], SPACE_DISPLAY_NAME_MAX_LENGTH),
                     _bounded_text(app.get("description"), SPACE_DESCRIPTION_MAX_LENGTH),
                     actor, now, now,
                     _bounded_text(app.get("qualified_name"), SPACE_QUALIFIED_NAME_MAX_LENGTH),
                     now, app.get("deployment_environment"),
                     app.get("workspace_name"), app.get("agent_owner"), app.get("certificate"),
                     json.dumps(app.get("domains") or []), json.dumps(app.get("tags") or []),
                     app.get("connector"), app.get("readme") or ""],
                )
        added = sum(1 for app in candidates if app["space_id"] not in existing_ids)
        return {
            "added": added,
            "updated": len(candidates) - added,
            "skipped_no_space_id": len(apps) - len(candidates),
            "total_seen": len(apps),
        }

    @staticmethod
    def _to_space(row: dict[str, Any]) -> Space:
        for field, max_length in _ATLAN_TEXT_LIMITS.items():
            if isinstance(row.get(field), str):
                row[field] = row[field][:max_length]
        raw = row.get("connection_params")
        row["connection_params"] = json.loads(raw) if raw else {}
        raw_tags = row.get("tags")
        row["tags"] = json.loads(raw_tags) if raw_tags else []
        raw_domains = row.get("domains")
        row["domains"] = json.loads(raw_domains) if raw_domains else []
        return Space.model_validate(row)

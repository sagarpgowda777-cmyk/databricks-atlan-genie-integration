from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
from typing import Any, Callable

from databricks.sdk import WorkspaceClient

from .atlan_client import AtlanClient
from .config import Settings
from .models import GenieSpaceDraft, GenieSpacePreview, GenieTableCandidate, GenieTableDiscovery

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "analysis", "analyze", "for", "from", "in", "of", "on", "or",
    "the", "to", "with", "data", "genie", "space",
}


class GenieBuilderError(RuntimeError):
    pass


class GenieBuilder:
    """Discovers user-visible tables and creates a confirmed Genie space as that user."""

    def __init__(
        self,
        settings: Settings,
        atlan: AtlanClient | None = None,
        workspace_factory: Callable[..., WorkspaceClient] = WorkspaceClient,
    ):
        self.settings = settings
        self.atlan = atlan
        self._workspace_factory = workspace_factory
        self._host = workspace_factory().config.host.rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.settings.genie_builder_warehouse_id and self.settings.genie_builder_signing_secret)

    def workspace_for_user(self, user_token: str) -> WorkspaceClient:
        if not user_token:
            raise GenieBuilderError("Missing signed-in user access token.")
        return self._workspace_factory(host=self._host, token=user_token, auth_type="pat")

    def genie_url(self, space_id: str) -> str:
        return f"{self._host}/genie/rooms/{space_id}"

    def discover_tables(
        self,
        user_token: str,
        purpose: str = "",
        search: str = "",
        offset: int = 0,
        limit: int = 50,
    ) -> GenieTableDiscovery:
        workspace = self.workspace_for_user(user_token)
        page_size = min(max(limit, 1), self.settings.genie_builder_max_tables)
        rows, total_count = self._visible_tables(workspace, search, max(offset, 0), page_size)
        candidates = [self._enrich_table(row, purpose) for row in rows]
        candidates.sort(
            key=lambda item: (
                not item.recommended,
                -item.match_score,
                item.full_name,
            )
        )
        return GenieTableDiscovery(
            tables=candidates,
            accessible_count=total_count,
            offset=max(offset, 0),
            next_offset=(max(offset, 0) + len(candidates)) if max(offset, 0) + len(candidates) < total_count else None,
            warehouse_id=self.settings.genie_builder_warehouse_id,
        )

    def preview(self, user_token: str, draft: GenieSpaceDraft) -> GenieSpacePreview:
        workspace = self.workspace_for_user(user_token)
        tables = [self._validate_selected_table(workspace, name, draft.description) for name in draft.table_names]
        token = self._confirmation_token(draft)
        return GenieSpacePreview(
            title=draft.title,
            description=draft.description,
            warehouse_id=self.settings.genie_builder_warehouse_id,
            parent_path=self.settings.genie_builder_parent_path or None,
            tables=tables,
            confirmation_token=token,
        )

    def create(self, user_token: str, draft: GenieSpaceDraft, confirmation_token: str) -> tuple[Any, list[GenieTableCandidate]]:
        if not hmac.compare_digest(self._confirmation_token(draft), confirmation_token):
            raise GenieBuilderError("The confirmed configuration changed. Review the table list again.")
        workspace = self.workspace_for_user(user_token)
        tables = [self._validate_selected_table(workspace, name, draft.description) for name in draft.table_names]
        serialized_space = json.dumps(
            {
                "version": 2,
                "data_sources": {
                    "tables": [
                        {"identifier": table.full_name, "column_configs": []}
                        for table in tables
                    ]
                },
            },
            separators=(",", ":"),
        )
        created = workspace.genie.create_space(
            warehouse_id=self.settings.genie_builder_warehouse_id,
            serialized_space=serialized_space,
            title=draft.title,
            description=draft.description,
            parent_path=self.settings.genie_builder_parent_path or None,
        )
        if not created.space_id:
            raise GenieBuilderError("Databricks created the space without returning a space ID.")
        return created, tables

    def _visible_tables(
        self, workspace: WorkspaceClient, search: str, offset: int, limit: int
    ) -> tuple[list[dict[str, str]], int]:
        # Unity Catalog information_schema is permission-aware for the caller. Exclude system and
        # information-schema rows, and cap the response so one request cannot flood the app.
        search_filter = ""
        parameters = None
        if search.strip():
            from databricks.sdk.service.sql import StatementParameterListItem

            search_filter = """AND (
                LOWER(CONCAT(table_catalog, '.', table_schema, '.', table_name)) LIKE :search
                OR LOWER(COALESCE(comment, '')) LIKE :search
            )"""
            parameters = [
                StatementParameterListItem(name="search", value=f"%{search.strip().lower()}%", type="STRING")
            ]
        statement = f"""
            SELECT table_catalog, table_schema, table_name, table_type,
                   COALESCE(comment, '') AS comment, COUNT(*) OVER() AS total_count
            FROM system.information_schema.tables
            WHERE table_catalog <> 'system'
                            AND LEFT(table_catalog, 2) <> '__'
              AND table_schema <> 'information_schema'
              {search_filter}
            ORDER BY table_catalog, table_schema, table_name
            LIMIT {int(limit)} OFFSET {int(offset)}
        """
        response = workspace.statement_execution.execute_statement(
            statement=statement,
            warehouse_id=self.settings.genie_builder_warehouse_id,
            parameters=parameters,
            wait_timeout="50s",
            row_limit=limit,
        )
        response = self._wait_for_statement(workspace, response)
        data = getattr(getattr(response, "result", None), "data_array", None) or []
        rows = [
            {
                "full_name": f"{row[0]}.{row[1]}.{row[2]}",
                "table_type": str(row[3] or ""),
                "comment": str(row[4] or ""),
            }
            for row in data
            if len(row) >= 6
        ]
        total_count = int(data[0][5]) if data else 0
        return rows, total_count

    def _validate_selected_table(
        self, workspace: WorkspaceClient, full_name: str, purpose: str
    ) -> GenieTableCandidate:
        try:
            table = workspace.tables.get(full_name)
        except Exception as exc:
            raise GenieBuilderError(f"Table '{full_name}' is unavailable to the signed-in user: {exc}") from exc
        actual_name = getattr(table, "full_name", None) or full_name
        quoted_name = ".".join(f"`{part}`" for part in actual_name.split("."))
        try:
            response = workspace.statement_execution.execute_statement(
                statement=f"SELECT * FROM {quoted_name} LIMIT 0",
                warehouse_id=self.settings.genie_builder_warehouse_id,
                wait_timeout="50s",
                row_limit=1,
            )
            self._wait_for_statement(workspace, response)
        except GenieBuilderError as exc:
            raise GenieBuilderError(
                f"The signed-in user cannot query table '{actual_name}': {exc}"
            ) from exc
        return self._enrich_table(
            {
                "full_name": actual_name,
                "table_type": str(getattr(table, "table_type", "") or ""),
                "comment": str(getattr(table, "comment", "") or ""),
            },
            purpose,
        )

    @staticmethod
    def _wait_for_statement(workspace: WorkspaceClient, response: Any) -> Any:
        for _ in range(20):
            status = getattr(response, "status", None)
            state = str(getattr(status, "state", "")).split(".")[-1]
            if state == "SUCCEEDED":
                return response
            if state in {"FAILED", "CANCELED", "CLOSED"}:
                message = getattr(getattr(status, "error", None), "message", "SQL validation failed.")
                raise GenieBuilderError(message)
            statement_id = getattr(response, "statement_id", None)
            if not statement_id:
                raise GenieBuilderError("SQL validation did not return a statement ID.")
            time.sleep(0.5)
            response = workspace.statement_execution.get_statement(statement_id)
        raise GenieBuilderError("SQL validation timed out.")

    def _enrich_table(self, row: dict[str, str], purpose: str) -> GenieTableCandidate:
        full_name = row["full_name"]
        catalog, schema, table = full_name.split(".", 2)
        asset: dict[str, Any] | None = None
        if self.atlan and self.atlan.enabled:
            try:
                asset = self.atlan.find_table_asset(catalog, schema, table)
            except Exception:
                logger.warning("Atlan enrichment failed for %s", full_name, exc_info=True)
        description = (asset or {}).get("description") or row.get("comment") or ""
        owners = (asset or {}).get("owners") or []
        certificate = str((asset or {}).get("certificate_status") or "")
        match_terms, score = self._match(description, full_name, purpose, certificate)
        return GenieTableCandidate(
            full_name=full_name,
            table_type=row.get("table_type") or "",
            databricks_comment=row.get("comment") or "",
            atlan_description=(asset or {}).get("description") or "",
            atlan_owners=owners,
            certificate_status=certificate,
            atlan_found=bool(asset),
            match_score=score,
            match_terms=match_terms,
            recommended=certificate.upper() == "VERIFIED" and len(match_terms) >= 2,
        )

    @staticmethod
    def _match(description: str, full_name: str, purpose: str, certificate: str) -> tuple[list[str], int]:
        purpose_words = {
            word for word in _WORD_RE.findall(purpose.lower())
            if len(word) >= 3 and word not in _STOPWORDS
        }
        table_words = set(_WORD_RE.findall(f"{full_name} {description}".lower()))
        matches = sorted(purpose_words & table_words)
        boost = 20 if certificate.upper() == "VERIFIED" else 6 if certificate else 0
        return matches, min(99, 32 + len(matches) * 13 + boost)

    def _confirmation_token(self, draft: GenieSpaceDraft) -> str:
        canonical = json.dumps(
            {
                "title": draft.title,
                "description": draft.description,
                "table_names": sorted(draft.table_names),
                "warehouse_id": self.settings.genie_builder_warehouse_id,
                "parent_path": self.settings.genie_builder_parent_path,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hmac.new(
            self.settings.genie_builder_signing_secret.encode(),
            canonical.encode(),
            hashlib.sha256,
        ).hexdigest()

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient

from .atlan_client import extract_table_refs
from .config import Settings
from .models import QueryResult, QueryResultColumn, SourceAnswer, Space

logger = logging.getLogger(__name__)

_PENDING_STATUSES = {
    "PENDING", "IN_PROGRESS", "SUBMITTED", "RUNNING", "FETCHING_METADATA", "EXECUTING_QUERY",
    "ASKING_AI", "EXECUTING", "FETCHING_DATA", "PENDING_WAREHOUSE",
    "FILTERING_CONTEXT", "WAITING_FOR_EXTERNAL_EXECUTION",
}
# 3s then 5s x 17 ≈ 88s of polling, matching typical Genie warm-up/query latency.
_POLL_DELAYS = [3.0] + [5.0] * 17
# Some Genie MCP servers don't return structured JSON from the ask call; instead they reply with
# plain text naming a poll tool (e.g. "Use the poll_response_0f... tool ... to poll until the
# message reaches a completed state").
_POLL_TOOL_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*poll[A-Za-z0-9_]*)\b", re.IGNORECASE)
_STILL_PROCESSING_RE = re.compile(r"still (processing|running)|not yet complete|poll (until|the)", re.IGNORECASE)
_MAX_RESULT_ROWS = 100


class GenieClient:
    """Calls each Genie space through its per-space Genie Agent MCP server, on behalf of the requesting user."""

    def __init__(self, settings: Settings):
        self.settings = settings
        # Tool schemas are per-space, not per-user, so they're safe to cache across requests.
        self._tool_specs: dict[str, dict[str, Any]] = {}

    def _mcp_client(self, space_id: str, workspace: WorkspaceClient) -> DatabricksMCPClient:
        host = workspace.config.host.rstrip("/")
        return DatabricksMCPClient(
            server_url=f"{host}/api/2.0/mcp/genie/{space_id}",
            workspace_client=workspace,
        )

    def _tools(self, client: DatabricksMCPClient, space_id: str) -> dict[str, Any]:
        cached = self._tool_specs.get(space_id)
        if cached is None:
            tools = client.list_tools()
            if not tools:
                raise RuntimeError(f"Genie MCP server for space {space_id} exposed no tools")
            poll_tool = next((tool for tool in tools if "poll" in tool.name.lower()), None)
            # Prefer a tool whose name signals it's the actual question/query entrypoint; falling
            # back to "first non-poll tool" risks picking an unrelated tool when a space exposes
            # more than just {ask, poll} (e.g. metadata/list-tables tools).
            ask_tool = next(
                (tool for tool in tools if tool is not poll_tool and any(
                    keyword in tool.name.lower() for keyword in ("query", "ask", "genie", "question")
                )),
                None,
            ) or next((tool for tool in tools if tool is not poll_tool), tools[0])
            cached = {"ask": ask_tool, "poll": poll_tool}
            self._tool_specs[space_id] = cached
        return cached

    @staticmethod
    def _primary_arg(tool: Any) -> str:
        schema = tool.inputSchema or {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        return next((name for name in required if name in properties), None) or next(iter(properties), "question")

    @staticmethod
    def _poll_args(tool: Any) -> tuple[str, str]:
        properties = (tool.inputSchema or {}).get("properties", {})
        conversation_arg = next((name for name in properties if "conversation" in name.lower()), "conversation_id")
        message_arg = next((name for name in properties if "message" in name.lower()), "message_id")
        return conversation_arg, message_arg

    @staticmethod
    def _parse_payload(result: Any) -> dict[str, Any] | None:
        for item in getattr(result, "content", None) or []:
            text = getattr(item, "text", None)
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _lookup(payload: dict[str, Any], *keys: str) -> Any:
        # Different Genie MCP server versions use camelCase or snake_case keys.
        for key in keys:
            if key in payload:
                return payload[key]
        return None

    def _poll_until_complete(
        self,
        client: DatabricksMCPClient,
        poll_tool: Any | None,
        payload: dict[str, Any] | None,
        raw_result: Any,
    ) -> tuple[dict[str, Any] | None, Any]:
        if not payload or poll_tool is None:
            return payload, raw_result
        conversation_id = self._lookup(payload, "conversationId", "conversation_id")
        message_id = self._lookup(payload, "messageId", "message_id")
        if not conversation_id or not message_id:
            return payload, raw_result
        conversation_arg, message_arg = self._poll_args(poll_tool)
        deadline = time.monotonic() + self.settings.genie_timeout_seconds
        status = str(payload.get("status", "")).upper()
        attempt = 0
        while status in _PENDING_STATUSES and time.monotonic() < deadline:
            time.sleep(_POLL_DELAYS[min(attempt, len(_POLL_DELAYS) - 1)])
            attempt += 1
            raw_result = client.call_tool(
                poll_tool.name,
                {conversation_arg: conversation_id, message_arg: message_id},
            )
            payload = self._parse_payload(raw_result) or payload
            status = str(payload.get("status", "")).upper()
        return payload, raw_result

    def _poll_dynamic_tool(
        self,
        client: DatabricksMCPClient,
        poll_tool: Any | None,
        conversation_id: str | None,
        raw_result: Any,
    ) -> tuple[dict[str, Any] | None, Any]:
        """Handle Genie MCP servers that don't return JSON from the ask call, but instead reply
        with plain text naming a poll tool (e.g. "poll_response_<message_id>"). Prefer the poll
        tool already discovered in the space's tool list (matches how other working Genie MCP
        integrations find it); fall back to calling the exact name mentioned in the text directly
        if no poll tool was cached, since it may not appear in list_tools() either way.
        """
        text = self._combined_text(raw_result)
        if not _STILL_PROCESSING_RE.search(text):
            # Plain text that isn't a "still processing" notice is already the final answer.
            return None, raw_result

        match = _POLL_TOOL_NAME_RE.search(text)
        tool_name = poll_tool.name if poll_tool is not None else (match.group(1) if match else None)
        if not tool_name:
            logger.info("Genie dynamic poll: no poll tool available (cached or in text): %r", text[:300])
            return None, raw_result

        message_id = self._message_id_from_tool_name(match.group(1)) if match else None
        if not message_id and poll_tool is not None:
            message_id = self._message_id_from_tool_name(poll_tool.name)
        conversation_arg, message_arg = self._poll_args(poll_tool) if poll_tool is not None else ("conversation_id", "message_id")
        # Try progressively simpler arg shapes since we don't always know this tool's exact schema.
        arg_candidates = []
        if conversation_id and message_id:
            arg_candidates.append({conversation_arg: conversation_id, message_arg: message_id})
        if message_id:
            arg_candidates.append({message_arg: message_id})
        arg_candidates.append({})
        logger.info("Genie dynamic poll: tool=%s message_id=%s conversation_id=%s", tool_name, message_id, conversation_id)

        deadline = time.monotonic() + self.settings.genie_timeout_seconds
        payload: dict[str, Any] | None = None
        attempt = 0
        while time.monotonic() < deadline:
            args = arg_candidates[min(attempt, len(arg_candidates) - 1)]
            try:
                raw_result = client.call_tool(tool_name, args)
            except Exception as exc:
                logger.warning("Genie dynamic poll: call_tool(%s, %s) failed: %s", tool_name, args, exc)
                if attempt >= len(arg_candidates) - 1:
                    return None, raw_result
                attempt += 1
                continue
            payload = self._parse_payload(raw_result)
            if payload:
                status = str(payload.get("status", "")).upper()
                logger.info("Genie dynamic poll: got JSON payload, status=%s", status)
                if status not in _PENDING_STATUSES:
                    return payload, raw_result
            else:
                text = self._combined_text(raw_result)
                logger.info("Genie dynamic poll: plain-text response: %r", text[:300])
                if not _STILL_PROCESSING_RE.search(text):
                    # Plain text with a real answer, not another "still processing" notice.
                    return None, raw_result
            time.sleep(_POLL_DELAYS[min(attempt, len(_POLL_DELAYS) - 1)])
            attempt += 1
        logger.warning("Genie dynamic poll: timed out after %ss for tool %s", self.settings.genie_timeout_seconds, tool_name)
        return payload, raw_result

    @staticmethod
    def _message_id_from_tool_name(tool_name: str) -> str | None:
        # e.g. "poll_response_01f1421ce78814189a2db778e3264ff0" -> "01f1421ce78814189a2db778e3264ff0"
        match = re.search(r"([0-9a-fA-F]{16,})$", tool_name)
        return match.group(1) if match else None

    @staticmethod
    def _combined_text(result: Any) -> str:
        return "\n".join(str(getattr(item, "text", "") or "") for item in getattr(result, "content", None) or [])

    @staticmethod
    def _answer_from_payload(payload: dict[str, Any] | None) -> str | None:
        if not payload:
            return None
        content = payload.get("content") or {}
        sections = [str(text) for text in content.get("textAttachments") or [] if text]
        for attachment in content.get("queryAttachments") or []:
            if isinstance(attachment, dict):
                description = attachment.get("description")
                if description:
                    sections.append(str(description))
                # The description is Genie's NL explanation; the actual generated SQL is a
                # separate field and was previously dropped, so users never saw the query it ran.
                query = attachment.get("query")
                if query:
                    sections.append(f"```sql\n{str(query).strip()}\n```")
        if str(payload.get("status", "")).upper() in {"FAILED", "ERROR", "CANCELED"}:
            sections.append(str(payload.get("error") or "Genie query failed."))
        return "\n\n".join(sections) if sections else None

    @staticmethod
    def _referenced_tables_from_payload(payload: dict[str, Any] | None) -> list[str]:
        if not payload:
            return []
        content = payload.get("content") or {}
        refs: dict[str, None] = {}
        for attachment in content.get("queryAttachments") or []:
            if not isinstance(attachment, dict):
                continue
            sql_text = attachment.get("query") or ""
            for catalog, schema, table in extract_table_refs(sql_text):
                refs.setdefault(f"{catalog}.{schema}.{table}", None)
        return list(refs)

    def _query_results_from_payload(
        self,
        payload: dict[str, Any] | None,
        space_id: str,
        workspace: WorkspaceClient,
    ) -> list[QueryResult]:
        if not payload:
            return []
        conversation_id = self._lookup(payload, "conversationId", "conversation_id")
        message_id = self._lookup(payload, "messageId", "message_id")
        if not conversation_id or not message_id:
            return []

        attachments = [
            attachment for attachment in (payload.get("content") or {}).get("queryAttachments") or []
            if isinstance(attachment, dict)
        ]
        attachment_ids = [
            self._lookup(attachment, "attachmentId", "attachment_id", "id")
            for attachment in attachments
        ]
        if not any(attachment_ids):
            try:
                message = workspace.genie.get_message(space_id, conversation_id, message_id)
                query_attachments = [item for item in message.attachments or [] if item.query]
                attachment_ids = [item.attachment_id for item in query_attachments]
                attachments = [item.query.as_dict() for item in query_attachments]
            except Exception:
                logger.warning(
                    "Could not discover Genie query attachments for message %s in space %s",
                    message_id, space_id, exc_info=True,
                )
                return []

        results: list[QueryResult] = []
        for attachment, attachment_id in zip(attachments, attachment_ids):
            if not attachment_id:
                continue
            try:
                response = workspace.genie.get_message_attachment_query_result(
                    space_id, conversation_id, message_id, attachment_id,
                )
                statement = response.statement_response
                schema = statement.manifest.schema if statement and statement.manifest else None
                result = statement.result if statement else None
                columns = schema.columns if schema else None
                if not columns or not result or not result.data_array:
                    continue
                rows = result.data_array[:_MAX_RESULT_ROWS]
                results.append(QueryResult(
                    title=self._lookup(attachment, "title", "description"),
                    columns=[QueryResultColumn(
                        name=column.name or f"Column {index + 1}",
                        type_name=column.type_name.value if column.type_name else column.type_text,
                    ) for index, column in enumerate(columns)],
                    rows=rows,
                    truncated=bool(
                        (statement.manifest and statement.manifest.truncated)
                        or len(result.data_array) > _MAX_RESULT_ROWS
                    ),
                ))
            except Exception:
                logger.warning(
                    "Could not fetch Genie query result for attachment %s in space %s",
                    attachment_id, space_id, exc_info=True,
                )
        return results

    def ask(
        self,
        space: Space,
        question: str,
        conversation_id: str | None,
        workspace: WorkspaceClient,
    ) -> SourceAnswer:
        # `question` is already fully enriched by the orchestrator (intent + recap); Genie
        # never sees the space's admin-authored routing instructions.
        client = self._mcp_client(space.space_id, workspace)
        try:
            tools = self._tools(client, space.space_id)
            ask_tool = tools["ask"]
            arg_name = self._primary_arg(ask_tool)
            result = client.call_tool(ask_tool.name, {arg_name: question})
            payload = self._parse_payload(result)
            logger.info(
                "Genie ask payload for space %s: %s | raw_text=%r",
                space.space_id, payload, self._combined_text(result)[:300],
            )
            payload, result = self._poll_until_complete(client, tools.get("poll"), payload, result)
            if not payload:
                payload, result = self._poll_dynamic_tool(client, tools.get("poll"), conversation_id, result)
            answer = self._answer_from_payload(payload) or self._extract_answer(result)
            new_conversation_id = self._lookup(payload or {}, "conversationId", "conversation_id") or conversation_id
            return SourceAnswer(
                space_id=space.space_id,
                display_name=space.display_name,
                answer=answer,
                conversation_id=new_conversation_id,
                referenced_tables=self._referenced_tables_from_payload(payload),
                query_results=self._query_results_from_payload(payload, space.space_id, workspace),
            )
        except Exception as exc:  # MCP/SDK errors vary by workspace/API version.
            logger.exception("Genie MCP call failed for space %s", space.space_id)
            return SourceAnswer(
                space_id=space.space_id,
                display_name=space.display_name,
                answer="",
                conversation_id=conversation_id,
                error=self._describe_error(exc),
            )

    @classmethod
    def _describe_error(cls, exc: BaseException) -> str:
        # anyio/httpx transports raise ExceptionGroup, which hides the real cause.
        sub_exceptions = getattr(exc, "exceptions", None)
        if sub_exceptions:
            return "; ".join(cls._describe_error(sub) for sub in sub_exceptions)
        return f"{type(exc).__name__}: {exc}"

    @classmethod
    def _extract_answer(cls, result: Any) -> str:
        sections: list[str] = []
        for item in getattr(result, "content", None) or []:
            text = getattr(item, "text", None)
            if text:
                sections.append(str(text))
        return "\n\n".join(sections) or "Genie completed without a text response."

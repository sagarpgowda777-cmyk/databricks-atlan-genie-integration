from __future__ import annotations

import logging
from typing import Any

import requests
from databricks.sdk import WorkspaceClient

from .config import Settings
from .models import SourceAnswer

logger = logging.getLogger(__name__)


class KnowledgeClient:
    """Invokes a Databricks agent/v1/responses endpoint as the signed-in user."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(self.settings.knowledge_endpoint)

    def ask(self, question: str, workspace: WorkspaceClient) -> SourceAnswer:
        endpoint = self.settings.knowledge_endpoint
        if not endpoint:
            return SourceAnswer(
                space_id="__knowledge__", display_name="Knowledge assistant", answer="",
                error="Knowledge endpoint is not configured.",
            )
        try:
            response = requests.post(
                f"{workspace.config.host.rstrip('/')}/serving-endpoints/{endpoint}/invocations",
                headers=workspace.config.authenticate(),
                json={
                    "input": [{
                        "role": "user",
                        "content": question,
                    }],
                },
                timeout=self.settings.knowledge_timeout_seconds,
            )
            response.raise_for_status()
            answer = self._answer_from_response(response.json())
            return SourceAnswer(
                space_id="__knowledge__", display_name="Knowledge assistant", answer=answer,
            )
        except Exception as exc:
            logger.exception("Knowledge endpoint invocation failed for %s", endpoint)
            return SourceAnswer(
                space_id="__knowledge__", display_name="Knowledge assistant", answer="",
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _answer_from_response(payload: dict[str, Any]) -> str:
        """Extract text from the standard OpenAI/Databricks Responses API result shape."""
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        sections: list[str] = []
        for item in payload.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                text = content.get("text") or content.get("output_text")
                if isinstance(text, str) and text.strip():
                    sections.append(text.strip())
        if sections:
            return "\n\n".join(sections)
        raise ValueError("Knowledge endpoint returned no text output")
from __future__ import annotations

import asyncio
import json
import queue
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings, get_settings
from .genie_builder import GenieBuilder, GenieBuilderError
from .genie_client import GenieClient
from .knowledge_client import KnowledgeClient
from .memory import MemoryStore
from .models import (
    ChatRequest, ChatResponse, GenieSpaceCreate, GenieSpaceCreated, GenieSpaceDraft,
    GenieSpacePreview, GenieTableDiscovery, Skill, SkillCreate, SkillUpdate, Space,
    SpaceUpdate, UsageEntryOut, UsageSummaryOut,
)
from .orchestrator import Orchestrator
from .registry import PostgresRegistry, RegistryError, SpaceNotFound
from .session_store import SessionContextStore
from .skills_store import SkillError, SkillNotFound, SkillStore
from .usage_store import UsageHistoryStore
from .atlan_client import AtlanClient
import mlflow


def create_app(settings: Settings | None = None, registry: PostgresRegistry | None = None) -> FastAPI:
    resolved = settings or get_settings()
    if resolved.mlflow_experiment_id:
        mlflow.set_tracking_uri(resolved.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_id=resolved.mlflow_experiment_id)
        mlflow.langchain.autolog()

    repository = registry or PostgresRegistry(resolved)
    session_store = SessionContextStore(resolved)
    skills_store = SkillStore(resolved)
    usage_store = UsageHistoryStore(resolved)
    knowledge = KnowledgeClient(resolved)
    memory = MemoryStore()
    atlan = AtlanClient(resolved.atlan_base_url, resolved.atlan_client_id, resolved.atlan_client_secret)
    genie_builder = GenieBuilder(resolved, atlan)
    orchestrator = Orchestrator(
        resolved, repository, GenieClient(resolved), memory, session_store, atlan=atlan, skills=skills_store,
        usage_history=usage_store, knowledge=knowledge,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # session_context table is created out-of-band; app role lacks CREATE TABLE on schema public
        await asyncio.to_thread(repository.list_spaces)
        yield

    app = FastAPI(title="Keystone Genie", version="0.1.0", lifespan=lifespan)
    app.state.registry = repository
    app.state.orchestrator = orchestrator
    app.state.memory = memory
    app.state.session_store = session_store
    app.state.skills_store = skills_store
    app.state.usage_store = usage_store
    app.state.genie_builder = genie_builder

    def repo() -> PostgresRegistry:
        return app.state.registry

    @app.exception_handler(SpaceNotFound)
    async def not_found_handler(_, exc: SpaceNotFound):
        return JSONResponse({"detail": f"Space {exc} was not found"}, status_code=404)

    @app.exception_handler(RegistryError)
    async def registry_handler(_, exc: RegistryError):
        return JSONResponse({"detail": f"Registry unavailable: {exc}"}, status_code=503)

    @app.exception_handler(SkillNotFound)
    async def skill_not_found_handler(_, exc: SkillNotFound):
        return JSONResponse({"detail": f"Skill {exc} was not found"}, status_code=404)

    @app.exception_handler(SkillError)
    async def skill_error_handler(_, exc: SkillError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(GenieBuilderError)
    async def genie_builder_error_handler(_, exc: GenieBuilderError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "llm_endpoint_configured": bool(resolved.llm_endpoint)}

    @app.get("/api/me")
    async def me(
        x_forwarded_email: str | None = Header(default=None),
        x_forwarded_preferred_username: str | None = Header(default=None),
        x_forwarded_user: str | None = Header(default=None),
    ) -> dict[str, Any]:
        return {
            "email": x_forwarded_email or x_forwarded_preferred_username,
            "user_id": x_forwarded_user,
            "signed_in": bool(x_forwarded_email or x_forwarded_preferred_username or x_forwarded_user),
        }

    @app.get("/api/spaces", response_model=list[Space])
    async def list_spaces(repository: PostgresRegistry = Depends(repo)):
        return await asyncio.to_thread(repository.list_spaces)

    @app.patch("/api/spaces/{space_id}", response_model=Space)
    async def update_space(space_id: str, payload: SpaceUpdate, repository: PostgresRegistry = Depends(repo)):
        return await asyncio.to_thread(repository.update, space_id, payload)

    @app.delete("/api/spaces/{space_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_space(space_id: str, repository: PostgresRegistry = Depends(repo)):
        await asyncio.to_thread(repository.delete, space_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/api/spaces/sync-atlan")
    async def sync_atlan_spaces(
        repository: PostgresRegistry = Depends(repo),
        x_forwarded_email: str | None = Header(default=None),
        x_databricks_user: str | None = Header(default=None),
    ) -> dict[str, int]:
        if not atlan.enabled:
            raise HTTPException(status_code=503, detail="Atlan is not configured (missing base URL/credentials)")
        actor = x_forwarded_email or x_databricks_user or "atlan-sync"
        apps = await asyncio.to_thread(atlan.discover_ai_applications)
        return await asyncio.to_thread(repository.sync_from_atlan, apps, actor)

    def require_builder_token(token: str | None) -> str:
        if not genie_builder.enabled:
            raise HTTPException(
                status_code=503,
                detail="Genie builder is not configured (warehouse or signing secret is missing).",
            )
        if not token:
            raise HTTPException(
                status_code=401,
                detail="Missing X-Forwarded-Access-Token; enable User authorization for this app.",
            )
        return token

    @app.get("/api/spaces/builder/tables", response_model=GenieTableDiscovery)
    async def discover_builder_tables(
        purpose: str = "",
        search: str = "",
        offset: int = 0,
        limit: int = 50,
        x_forwarded_access_token: str | None = Header(default=None, alias="X-Forwarded-Access-Token"),
    ):
        token = require_builder_token(x_forwarded_access_token)
        return await asyncio.to_thread(
            genie_builder.discover_tables, token, purpose, search, offset, limit,
        )

    @app.post("/api/spaces/builder/preview", response_model=GenieSpacePreview)
    async def preview_genie_space(
        payload: GenieSpaceDraft,
        x_forwarded_access_token: str | None = Header(default=None, alias="X-Forwarded-Access-Token"),
    ):
        token = require_builder_token(x_forwarded_access_token)
        return await asyncio.to_thread(genie_builder.preview, token, payload)

    @app.post("/api/spaces/builder/create", response_model=GenieSpaceCreated, status_code=status.HTTP_201_CREATED)
    async def create_genie_space(
        payload: GenieSpaceCreate,
        x_forwarded_access_token: str | None = Header(default=None, alias="X-Forwarded-Access-Token"),
        x_forwarded_email: str | None = Header(default=None),
        x_forwarded_user: str | None = Header(default=None),
        repository: PostgresRegistry = Depends(repo),
    ):
        token = require_builder_token(x_forwarded_access_token)
        actor = x_forwarded_email or x_forwarded_user or "unknown"
        draft = GenieSpaceDraft(
            title=payload.title,
            description=payload.description,
            table_names=payload.table_names,
        )
        created, tables = await asyncio.to_thread(
            genie_builder.create, token, draft, payload.confirmation_token,
        )
        space = await asyncio.to_thread(
            repository.create_genie_space,
            created.space_id,
            created.title or payload.title,
            created.description or payload.description,
            actor,
            created.warehouse_id or resolved.genie_builder_warehouse_id,
            [table.full_name for table in tables],
        )
        return GenieSpaceCreated(
            space=space,
            genie_url=genie_builder.genie_url(created.space_id),
            tables=[table.full_name for table in tables],
        )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest,
        x_forwarded_access_token: str | None = Header(default=None, alias="X-Forwarded-Access-Token"),
        x_forwarded_email: str | None = Header(default=None),
        x_forwarded_user: str | None = Header(default=None),
    ):
        if not x_forwarded_access_token:
            raise HTTPException(
                status_code=401,
                detail="Missing X-Forwarded-Access-Token; enable User authorization for this app.",
            )
        try:
            return await asyncio.to_thread(
                app.state.orchestrator.chat,
                payload.session_id,
                payload.message,
                x_forwarded_access_token,
                x_forwarded_email or x_forwarded_user or "unknown",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/chat/stream")
    async def chat_stream(
        payload: ChatRequest,
        x_forwarded_access_token: str | None = Header(default=None, alias="X-Forwarded-Access-Token"),
        x_forwarded_email: str | None = Header(default=None),
        x_forwarded_user: str | None = Header(default=None),
    ) -> StreamingResponse:
        if not x_forwarded_access_token:
            raise HTTPException(
                status_code=401,
                detail="Missing X-Forwarded-Access-Token; enable User authorization for this app.",
            )
        user_email = x_forwarded_email or x_forwarded_user or "unknown"

        async def event_source():
            # orchestrator.chat_stream() is a plain sync generator (LangGraph/HTTP calls aren't
            # async); run it on a worker thread and bridge each event to this async generator via
            # a thread-safe queue, so reasoning steps can be flushed to the client as they occur
            # instead of only once the whole request finishes.
            q: "queue.Queue[dict[str, Any] | None]" = queue.Queue()

            def worker() -> None:
                try:
                    for event in app.state.orchestrator.chat_stream(
                        payload.session_id, payload.message, x_forwarded_access_token, user_email,
                    ):
                        if event["type"] == "final":
                            event = {"type": "final", "response": event["response"].model_dump(mode="json")}
                        q.put(event)
                except Exception as exc:  # surfaced to the client as an error event, not a 500 mid-stream
                    q.put({"type": "error", "detail": str(exc)})
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            while True:
                item = await asyncio.to_thread(q.get)
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.get("/api/sessions")
    async def list_sessions(
        x_forwarded_email: str | None = Header(default=None),
        x_forwarded_user: str | None = Header(default=None),
    ) -> list[dict[str, Any]]:
        user_email = x_forwarded_email or x_forwarded_user or "unknown"
        summaries = await asyncio.to_thread(app.state.session_store.list_sessions, user_email)
        return [
            {"session_id": item.session_id, "title": item.title, "updated_at": item.updated_at.isoformat()}
            for item in summaries
        ]

    @app.get("/api/sessions/{session_id}/messages")
    async def get_session_messages(session_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(app.state.session_store.load_message_records, session_id)

    @app.delete("/api/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def clear_session(session_id: str):
        app.state.memory.clear(session_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/usage")
    async def get_usage(
        x_forwarded_email: str | None = Header(default=None),
        x_forwarded_user: str | None = Header(default=None),
    ) -> dict[str, Any]:
        # Scoped to the caller's own forwarded identity only -- never accepts a user_email query
        # param, so one user can't read another user's usage history.
        user_email = x_forwarded_email or x_forwarded_user or "unknown"
        entries = await asyncio.to_thread(app.state.usage_store.list_for_user, user_email)
        summary = await asyncio.to_thread(app.state.usage_store.summary_for_user, user_email)
        return {
            "summary": UsageSummaryOut(
                question_count=summary.question_count, total_tokens=summary.total_tokens,
                total_cost_usd=summary.total_cost_usd,
            ).model_dump(mode="json"),
            "entries": [
                UsageEntryOut(
                    question=entry.question, prompt_tokens=entry.prompt_tokens,
                    completion_tokens=entry.completion_tokens, total_tokens=entry.total_tokens,
                    estimated_cost_usd=entry.estimated_cost_usd, created_at=entry.created_at,
                ).model_dump(mode="json")
                for entry in entries
            ],
        }

    @app.get("/api/usage/export")
    async def export_usage(
        x_forwarded_email: str | None = Header(default=None),
        x_forwarded_user: str | None = Header(default=None),
    ) -> dict[str, Any]:
        # Full, unpruned history for the caller only. The dashboard endpoint above remains capped
        # for fast rendering, while XLSX export intentionally includes every retained usage row.
        user_email = x_forwarded_email or x_forwarded_user or "unknown"
        entries = await asyncio.to_thread(app.state.usage_store.list_for_user, user_email, None)
        summary = await asyncio.to_thread(app.state.usage_store.summary_for_user, user_email)
        return {
            "summary": UsageSummaryOut(
                question_count=summary.question_count, total_tokens=summary.total_tokens,
                total_cost_usd=summary.total_cost_usd,
            ).model_dump(mode="json"),
            "entries": [
                UsageEntryOut(
                    question=entry.question, prompt_tokens=entry.prompt_tokens,
                    completion_tokens=entry.completion_tokens, total_tokens=entry.total_tokens,
                    estimated_cost_usd=entry.estimated_cost_usd, created_at=entry.created_at,
                ).model_dump(mode="json")
                for entry in entries
            ],
        }

    @app.get("/api/skills", response_model=list[Skill])
    async def list_skills(
        x_forwarded_email: str | None = Header(default=None),
        x_forwarded_preferred_username: str | None = Header(default=None),
        x_forwarded_user: str | None = Header(default=None),
    ):
        user_email = x_forwarded_email or x_forwarded_preferred_username or x_forwarded_user or "unknown"
        return await asyncio.to_thread(app.state.skills_store.list_for_user, user_email)

    @app.post("/api/skills", response_model=Skill, status_code=status.HTTP_201_CREATED)
    async def create_skill(
        payload: SkillCreate,
        x_forwarded_email: str | None = Header(default=None),
        x_forwarded_preferred_username: str | None = Header(default=None),
        x_forwarded_user: str | None = Header(default=None),
    ):
        user_email = x_forwarded_email or x_forwarded_preferred_username or x_forwarded_user or "unknown"
        return await asyncio.to_thread(app.state.skills_store.create, user_email, payload)

    @app.patch("/api/skills/{skill_id}", response_model=Skill)
    async def update_skill(
        skill_id: str,
        payload: SkillUpdate,
        x_forwarded_email: str | None = Header(default=None),
        x_forwarded_preferred_username: str | None = Header(default=None),
        x_forwarded_user: str | None = Header(default=None),
    ):
        user_email = x_forwarded_email or x_forwarded_preferred_username or x_forwarded_user or "unknown"
        return await asyncio.to_thread(app.state.skills_store.update, skill_id, user_email, payload)

    @app.delete("/api/skills/{skill_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_skill(
        skill_id: str,
        x_forwarded_email: str | None = Header(default=None),
        x_forwarded_preferred_username: str | None = Header(default=None),
        x_forwarded_user: str | None = Header(default=None),
    ):
        user_email = x_forwarded_email or x_forwarded_preferred_username or x_forwarded_user or "unknown"
        await asyncio.to_thread(app.state.skills_store.delete, skill_id, user_email)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app

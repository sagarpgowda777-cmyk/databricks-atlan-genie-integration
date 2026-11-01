from __future__ import annotations

import operator
import logging
import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from databricks.sdk import WorkspaceClient

from .config import Settings
from .genie_client import GenieClient
from .knowledge_client import KnowledgeClient
from .memory import MemoryStore
from .models import ChatResponse, GraphEdge, GraphNode, SkillSuggestion, SourceAnswer, Space, TableLineage
from .registry import PostgresRegistry
from .session_store import SessionContext, SessionContextStore
from .skills_store import SkillStore
from .usage_store import UsageHistoryStore
from .atlan_client import (
    AtlanClient,
    extract_bare_table_candidates,
    extract_data_product_candidates,
    extract_glossary_term_candidates,
    extract_table_refs,
    format_context_markdown,
    format_data_product_markdown,
    format_glossary_markdown,
)


logger = logging.getLogger(__name__)

_LINEAGE_KEYWORDS = (
    "lineage", "upstream", "downstream", "where does", "comes from", "feeds into",
    "built from", "derived from", "source of", "sources for", "traces back", "data flow",
)
_GLOSSARY_KEYWORDS = (
    "glossary", "business term", "business terms", "term definition", "what does", "what is the meaning of",
    "definition of", "meaning of",
)
_DATA_PRODUCT_KEYWORDS = (
    "data product", "data products",
)
_AI_APPLICATION_METADATA_KEYWORDS = (
    "metadata", "details", "owner", "ownership", "certification", "certificate", "certified",
    "tags", "qualified name", "environment", "workspace", "registered", "registration",
)
_KNOWLEDGE_KEYWORDS = (
    "faq", "onboarding", "onboard", "how do i", "how to", "setup", "set up", "configure",
    "guide", "documentation", "databricks app", "genie app", "registration",
)
_GENIE_CATALOG_KEYWORDS = (
    "available", "list", "show", "which", "what each", "can answer", "capabilities",
)


class GraphState(TypedDict):
    question: str
    history: str
    intent_summary: str
    intent_graph: dict[str, Any]
    turn_id: str
    context_recap: str
    selected_space_ids: list[str]
    responses: Annotated[list[SourceAnswer], operator.add]
    final_answer: str
    reasoning: Annotated[list[str], operator.add]
    # Populated only by the direct-to-Atlan lineage node (bypasses Genie entirely), so its
    # upstream/downstream graph data still reaches ChatResponse.lineage like the Genie-sourced path.
    direct_lineage: Annotated[list[TableLineage], operator.add]
    # Set by route() only when it fell through to full LLM/heuristic routing (not a skill match or
    # follow-up), so clients can offer to save it as a skill for next time.
    skill_suggestion: dict[str, Any] | None


class RouteDecision(BaseModel):
    space_ids: list[str] = Field(
        default_factory=list,
        description="Exact space IDs whose description genuinely matches the question's intent. "
        "Empty list if none of the spaces plausibly answer it -- never guess a space just to return one.",
    )
    reasoning: str = Field(
        default="",
        description="One or two sentences explaining, in terms a user can follow, why these exact "
        "space(s) were chosen (what in their description matched) or why none were chosen.",
    )


class IntentResult(BaseModel):
    intent_summary: str = Field(
        description="Self-contained restatement of the user's intent, resolving pronouns/follow-up references"
    )
    context_recap: str = Field(
        default="",
        description="Short recap of prior context relevant to a space seeing this conversation for the "
        "first time; empty on the first turn",
    )
    add_nodes: list[GraphNode] = Field(
        default_factory=list,
        description="New business entities/domains worth remembering (type must be 'entity' or 'domain' only "
        "-- turn/space/table nodes are managed by the system). Reuse an existing id from the graph context "
        "instead of minting a new one when the same concept recurs.",
    )
    add_edges: list[GraphEdge] = Field(
        default_factory=list,
        description="New relations between entities/domains (relation must be 'belongs_to_domain' or "
        "'resolves_to' only -- 'mentions'/'answered_by'/'refines' edges are managed by the system).",
    )


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        registry: PostgresRegistry,
        genie: GenieClient,
        memory: MemoryStore,
        session_store: SessionContextStore,
        llm: Any | None = None,
        atlan: AtlanClient | None = None,
        skills: SkillStore | None = None,
        usage_history: UsageHistoryStore | None = None,
        knowledge: KnowledgeClient | None = None,
    ):
        self.settings = settings
        self.registry = registry
        self.genie = genie
        self.memory = memory
        self.session_store = session_store
        self.atlan = atlan
        self.skills = skills
        self.usage_history = usage_history
        self.knowledge = knowledge
        # Explicit override (e.g. tests) bypasses OBO and is used as-is for every request.
        self._llm_override = llm
        # Used both to read the workspace host and to call the LLM as the app's own service
        # principal -- routing/merge prompts only see space descriptions and conversation text
        # (no row-level user data), so they don't need per-user OBO auth. This also means the
        # "Can query" grant on the llm_endpoint app resource is sufficient for every user, instead
        # of requiring each individual signed-in user to separately have serving-endpoint access.
        self._app_workspace = WorkspaceClient()
        self._host = self._app_workspace.config.host

    def _make_llm(self, workspace: WorkspaceClient) -> Any | None:
        if not self.settings.llm_endpoint:
            return None
        from databricks_langchain import ChatDatabricks

        return ChatDatabricks(endpoint=self.settings.llm_endpoint, temperature=0, workspace_client=workspace)

    def chat(self, session_id: str, question: str, user_token: str, user_email: str = "unknown") -> ChatResponse:
        for event in self._run(session_id, question, user_token, user_email):
            if event["type"] == "final":
                return event["response"]
        raise RuntimeError("Orchestrator run finished without producing a final response")

    def chat_stream(self, session_id: str, question: str, user_token: str, user_email: str = "unknown"):
        """Same as chat(), but yields {'type': 'step', 'text': ...} events as each graph node
        completes (in real time, during inference), followed by a final {'type': 'final', 'response': ChatResponse}.
        """
        yield from self._run(session_id, question, user_token, user_email)

    def _run(self, session_id: str, question: str, user_token: str, user_email: str):
        if not user_token:
            raise ValueError("Missing on-behalf-of-user access token (X-Forwarded-Access-Token)")
        # auth_type="pat" forces token-only auth; otherwise the SDK also picks up the
        # app's DATABRICKS_CLIENT_ID/SECRET env vars and errors on conflicting auth methods.
        obo_workspace = WorkspaceClient(host=self._host, token=user_token, auth_type="pat")
        llm = self._llm_override if self._llm_override is not None else self._make_llm(self._app_workspace)
        spaces = self.registry.list_spaces(enabled_only=True)
        if not spaces:
            yield {
                "type": "final",
                "response": ChatResponse(
                    answer="No enabled Genie spaces are configured. Add one in Manage spaces.",
                    answered_by=[], sources=[],
                ),
            }
            return
        pre_matched_skill = None
        is_direct_source_question = (
            self._is_genie_catalog_question(question)
            or self._is_ai_application_metadata_question(question)
            or self._is_lineage_question(question)
            or self._is_glossary_question(question)
            or self._is_data_product_question(question)
            or (self.knowledge and self.knowledge.enabled and self._is_knowledge_question(question))
        )
        if self.skills and not is_direct_source_question:
            pre_matched_skill = self.skills.match(question, user_email)
            if pre_matched_skill and question.lstrip().startswith("/"):
                question = self.skills.question_for_skill(question, pre_matched_skill)
        session = self.memory.get(session_id)
        # A fresh MemoryStore (process restart, new replica, or first hit on this instance) has no
        # in-memory transcript yet -- rehydrate it from Lakebase so history/routing continuity survives.
        if not session.history.messages:
            for role, content in self.session_store.load_messages(session_id, intent_only=True):
                if role == "human":
                    session.history.add_user_message(content)
                else:
                    session.history.add_ai_message(content)
        context = self.session_store.load(session_id)
        history = "\n".join(
            f"{message.type}: {message.content}" for message in session.history.messages[-6:]
        )
        graph = self._build_graph(
            spaces, context, obo_workspace, llm, user_email, pre_matched_skill=pre_matched_skill,
        )
        initial_state: GraphState = {
            "question": question,
            "history": history,
            "intent_summary": "",
            "intent_graph": context.intent_graph,
            "turn_id": "",
            "context_recap": "",
            "selected_space_ids": [],
            "responses": [],
            "final_answer": "",
            "reasoning": [],
            "direct_lineage": [],
            "skill_suggestion": None,
        }
        result = initial_state
        seen_reasoning = 0
        # stream_mode="values" yields the full (already-reduced) state after each node finishes,
        # so each new "reasoning" entry can be surfaced to the caller the moment it's produced,
        # instead of only after the whole graph run completes.
        for state in graph.stream(initial_state, stream_mode="values"):
            result = state
            reasoning = state.get("reasoning") or []
            for line in reasoning[seen_reasoning:]:
                yield {"type": "step", "text": line}
            seen_reasoning = len(reasoning)

        responses = result["responses"]
        successful = [item for item in responses if not item.error]
        genie_space_ids = {space.space_id for space in spaces}
        is_genie_turn = any(space_id in genie_space_ids for space_id in result["selected_space_ids"])
        successful_genie = [item for item in successful if item.space_id in genie_space_ids]
        if is_genie_turn:
            context.active_space_ids = [item.space_id for item in successful_genie]
            for item in successful_genie:
                if item.conversation_id:
                    context.genie_conversations[item.space_id] = item.conversation_id
            context.intent_summary = result["intent_summary"] or context.intent_summary
            context.intent_graph = result["intent_graph"] or context.intent_graph
        final_answer = result["final_answer"]
        turn_number = self._turn_number_from_id(result["turn_id"]) if is_genie_turn else 0
        answered_by_patch = self._answered_by_patch(result["turn_id"], turn_number, successful_genie) if is_genie_turn else {}
        context.intent_graph = SessionContextStore.apply_patch(context.intent_graph, answered_by_patch)
        lineage, lineage_patch = self._enrich_lineage(successful_genie, turn_number) if is_genie_turn else ([], {})
        lineage = lineage + (result.get("direct_lineage") or [])
        if is_genie_turn:
            context.intent_graph = SessionContextStore.apply_patch(context.intent_graph, lineage_patch)
            self.session_store.save(context)
        query_results = [
            query_result.model_dump(mode="json")
            for response in successful_genie
            for query_result in response.query_results
        ]
        self.session_store.record_turn(
            session_id, user_email, question, final_answer, intent_eligible=is_genie_turn,
            presentation={"query_results": query_results} if query_results else None,
        )
        # Lineage stays structured (ChatResponse.lineage) rather than being string-appended onto
        # the answer, so API clients can present it separately.
        if is_genie_turn:
            session.history.add_messages([HumanMessage(content=question), AIMessage(content=final_answer)])
        skill_suggestion_data = (
            result.get("skill_suggestion")
            if is_genie_turn and successful_genie and turn_number == 1
            else None
        )
        token_usage, estimated_cost = self._capture_token_usage()
        if self.usage_history:
            self.usage_history.record(user_email, session_id, question, token_usage, estimated_cost)
        yield {
            "type": "final",
            "response": ChatResponse(
                answer=final_answer,
                answered_by=[item.display_name for item in successful],
                sources=responses,
                lineage=lineage,
                reasoning=result["reasoning"],
                skill_suggestion=SkillSuggestion(**skill_suggestion_data) if skill_suggestion_data else None,
                token_usage=token_usage,
                estimated_cost_usd=estimated_cost,
            ),
        }

    def _capture_token_usage(self) -> tuple[dict[str, int] | None, float | None]:
        """Reads token usage from the MLflow trace autologged for this turn's LLM calls (mlflow.langchain.autolog()
        in api.py), so the end user can see roughly what a turn cost without needing MLflow UI access. Best-effort
        only -- the exact trace attribute shape has changed across MLflow versions, so any failure just means no
        usage/cost is shown, rather than breaking the turn."""
        if not self.settings.mlflow_experiment_id:
            return None, None
        try:
            import mlflow

            trace_id = mlflow.get_last_active_trace_id()
            trace = mlflow.get_trace(trace_id, flush=True) if trace_id else None
            usage = self._extract_token_usage(trace) if trace is not None else None
            if not usage:
                logger.info("No token usage found on MLflow trace %s for this turn", trace_id)
                return None, None
            trace_cost = usage.pop("cost", None)
            cost = trace_cost if trace_cost is not None else self._estimate_cost(usage)
            return usage, cost
        except Exception:
            logger.warning("Could not read MLflow trace token usage for this turn", exc_info=True)
            return None, None

    @staticmethod
    def _extract_token_usage(trace: Any) -> dict[str, Any] | None:
        # MLflow >=2.21 aggregates usage directly on trace.info; older versions only expose it per-span.
        # Databricks-hosted endpoints also attach a platform-computed dollar cost per span, which we prefer
        # over our own price-per-token estimate when it's available.
        prompt = completion = total = 0
        cost = 0.0
        found_cost = False
        info_usage = getattr(getattr(trace, "info", None), "token_usage", None)
        if isinstance(info_usage, dict) and info_usage:
            prompt = int(info_usage.get("input_tokens") or info_usage.get("prompt_tokens") or 0)
            completion = int(info_usage.get("output_tokens") or info_usage.get("completion_tokens") or 0)
            total = int(info_usage.get("total_tokens") or 0)
        for span in getattr(getattr(trace, "data", None), "spans", None) or []:
            get_attribute = getattr(span, "get_attribute", None)
            attributes = getattr(span, "attributes", None) or {}

            def _attr(key: str) -> Any:
                if callable(get_attribute):
                    value = get_attribute(key)
                    if value is not None:
                        return value
                return attributes.get(key)

            if not (prompt or completion or total):
                span_usage = _attr("mlflow.chat.tokenUsage")
                if isinstance(span_usage, dict):
                    prompt += int(span_usage.get("input_tokens") or span_usage.get("prompt_tokens") or 0)
                    completion += int(span_usage.get("output_tokens") or span_usage.get("completion_tokens") or 0)
                    total += int(span_usage.get("total_tokens") or 0)
            for key, value in attributes.items():
                if "cost" in key.lower() and isinstance(value, (int, float)):
                    cost += float(value)
                    found_cost = True
        if not (prompt or completion or total):
            return None
        usage: dict[str, Any] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total or (prompt + completion),
        }
        if found_cost:
            usage["cost"] = round(cost, 6)
        return usage

    def _estimate_cost(self, usage: dict[str, int]) -> float | None:
        if not (self.settings.llm_input_price_per_million or self.settings.llm_output_price_per_million):
            return None
        input_cost = usage.get("prompt_tokens", 0) / 1_000_000 * self.settings.llm_input_price_per_million
        output_cost = usage.get("completion_tokens", 0) / 1_000_000 * self.settings.llm_output_price_per_million
        return round(input_cost + output_cost, 6)

    @staticmethod
    def _turn_number_from_id(turn_id: str) -> int:
        try:
            return int(turn_id.split(":", 1)[1])
        except (IndexError, ValueError):
            return 0

    def _answered_by_patch(self, turn_id: str, turn_number: int, successful: list[SourceAnswer]) -> dict[str, Any]:
        """Codes each space that actually answered this turn into the graph, since routing/answering
        is a system fact -- not something the LLM should have to assert itself."""
        if not turn_id or not successful:
            return {}
        add_nodes: list[dict[str, Any]] = []
        add_edges: list[dict[str, Any]] = []
        for item in successful:
            space_node_id = f"space:{item.space_id}"
            add_nodes.append(
                GraphNode(
                    id=space_node_id, type="space", label=item.display_name,
                    first_seen_turn=turn_number, last_seen_turn=turn_number, source="genie",
                ).model_dump()
            )
            add_edges.append(
                GraphEdge(
                    from_=turn_id, to=space_node_id, relation="answered_by",
                    first_seen_turn=turn_number, source="genie",
                ).model_dump(by_alias=True)
            )
        return {"add_nodes": add_nodes, "add_edges": add_edges}

    def _enrich_lineage(
        self, successful: list[SourceAnswer], turn_number: int
    ) -> tuple[list[TableLineage], dict[str, Any]]:
        if not self.atlan or not self.atlan.enabled:
            return [], {}
        # A table can be referenced by more than one space in the same turn; tag the intent-graph
        # node with every space that surfaced it so future intent summaries know where it came from.
        table_space_ids: dict[str, list[str]] = {}
        for item in successful:
            for fqtn in item.referenced_tables:
                table_space_ids.setdefault(fqtn, [])
                if item.space_id not in table_space_ids[fqtn]:
                    table_space_ids[fqtn].append(item.space_id)

        lineage: list[TableLineage] = []
        add_nodes: list[dict[str, Any]] = []
        add_edges: list[dict[str, Any]] = []
        for fqtn, space_ids in table_space_ids.items():
            parts = fqtn.split(".")
            if len(parts) != 3:
                continue
            catalog, schema, table = parts
            try:
                context = self.atlan.get_table_context(catalog, schema, table)
            except Exception as exc:
                logger.warning("Atlan enrichment failed for %s", fqtn, exc_info=exc)
                continue
            upstream_fqtns = [
                f"{u['catalog']}.{u['schema']}.{u['table']}"
                for u in context.get("upstream") or [] if u.get("catalog") and u.get("schema") and u.get("table")
            ]
            downstream_fqtns = [
                f"{d['catalog']}.{d['schema']}.{d['table']}"
                for d in context.get("downstream") or [] if d.get("catalog") and d.get("schema") and d.get("table")
            ]
            lineage.append(TableLineage(
                table=fqtn, context_markdown=format_context_markdown(fqtn, context),
                upstream=upstream_fqtns, downstream=downstream_fqtns,
            ))

            table_node_id = f"table:{fqtn}"
            asset_guid = (context.get("asset") or {}).get("guid") if context.get("found") else None
            add_nodes.append(
                GraphNode(
                    id=table_node_id, type="table", label=fqtn, space_ids=space_ids,
                    catalog=catalog, table_schema=schema, table_name=table, atlan_guid=asset_guid,
                    first_seen_turn=turn_number, last_seen_turn=turn_number, source="atlan",
                ).model_dump()
            )
            for upstream in context.get("upstream") or []:
                source_fqtn = f"{upstream['catalog']}.{upstream['schema']}.{upstream['table']}"
                source_node_id = f"table:{source_fqtn}"
                add_nodes.append(
                    GraphNode(
                        id=source_node_id, type="table", label=source_fqtn,
                        catalog=upstream["catalog"], table_schema=upstream["schema"], table_name=upstream["table"],
                        first_seen_turn=turn_number, last_seen_turn=turn_number, source="atlan",
                    ).model_dump()
                )
                add_edges.append(
                    GraphEdge(
                        from_=source_node_id, to=table_node_id, relation="feeds",
                        first_seen_turn=turn_number, source="atlan",
                    ).model_dump(by_alias=True)
                )
            for downstream in context.get("downstream") or []:
                target_fqtn = f"{downstream['catalog']}.{downstream['schema']}.{downstream['table']}"
                target_node_id = f"table:{target_fqtn}"
                add_nodes.append(
                    GraphNode(
                        id=target_node_id, type="table", label=target_fqtn,
                        catalog=downstream["catalog"], table_schema=downstream["schema"], table_name=downstream["table"],
                        first_seen_turn=turn_number, last_seen_turn=turn_number, source="atlan",
                    ).model_dump()
                )
                add_edges.append(
                    GraphEdge(
                        from_=table_node_id, to=target_node_id, relation="feeds",
                        first_seen_turn=turn_number, source="atlan",
                    ).model_dump(by_alias=True)
                )
        return lineage, {"add_nodes": add_nodes, "add_edges": add_edges}

    _GRAPH_RELEVANCE_STOPWORDS = {
        "a", "an", "the", "of", "for", "me", "you", "can", "could", "please", "provide", "give",
        "show", "get", "tell", "about", "data", "and", "or", "to", "in", "on", "with", "this",
        "that", "it", "its", "what", "which", "who", "is", "are", "from", "database",
    }

    @classmethod
    def _relevant_node_ids(cls, nodes: dict[str, dict], edges: list[dict], question: str) -> set[str]:
        """Only entities/domains whose label shares a real word with the current question (plus
        anything one hop away in the graph) count as relevant -- the point is to stop unrelated
        domains from earlier, unrelated questions leaking into this turn's context.
        """
        question_words = {
            word for word in re.findall(r"[A-Za-z0-9_]+", question.lower())
            if len(word) >= 3 and word not in cls._GRAPH_RELEVANCE_STOPWORDS
        }
        if not question_words:
            return set()
        matched: set[str] = set()
        for node_id, node in nodes.items():
            label_words = set(re.findall(r"[A-Za-z0-9_]+", str(node.get("label", "")).lower()))
            if label_words & question_words:
                matched.add(node_id)
        if not matched:
            return set()
        expanded = set(matched)
        for edge in edges:
            if edge.get("from") in matched:
                expanded.add(edge.get("to"))
            if edge.get("to") in matched:
                expanded.add(edge.get("from"))
        return expanded

    @classmethod
    def _graph_context_text(cls, intent_graph: dict[str, Any], question: str, limit: int = 15) -> str:
        """Render a short summary of the session's ontology graph so the intent-understanding
        prompt can reuse entities/domains/lineage discovered in earlier turns -- scoped to only
        what's actually relevant to the current question, so an unrelated earlier topic (e.g. a
        customs/US11 question) never bleeds into a completely different one (e.g. PS4 data).
        """
        nodes = {node.get("id"): node for node in intent_graph.get("nodes", []) if isinstance(node, dict)}
        edges = [edge for edge in intent_graph.get("edges", []) if isinstance(edge, dict)]
        if not nodes:
            return ""
        relevant = cls._relevant_node_ids(nodes, edges, question)
        if not relevant:
            return ""
        lines = []
        entities = [node for node in nodes.values() if node.get("type") in ("entity", "domain") and node.get("id") in relevant]
        if entities:
            entity_bits = [f"{node['id']} ({node.get('label')})" for node in entities[-limit:]]
            lines.append("Known entities/domains relevant to this question (reuse these ids if the same concept recurs): " + ", ".join(entity_bits))
        domain_edges = [edge for edge in edges if edge.get("relation") == "belongs_to_domain" and (edge.get("from") in relevant or edge.get("to") in relevant)]
        for edge in domain_edges[-limit:]:
            source = nodes.get(edge.get("from"), {})
            target = nodes.get(edge.get("to"), {})
            lines.append(f"- {source.get('label', edge.get('from'))} belongs_to_domain {target.get('label', edge.get('to'))}")
        feeds_edges = [edge for edge in edges if edge.get("relation") == "feeds" and (edge.get("from") in relevant or edge.get("to") in relevant)]
        for edge in feeds_edges[-limit:]:
            source = nodes.get(edge.get("from"), {})
            target = nodes.get(edge.get("to"), {})
            space_ids = target.get("space_ids") or []
            space_note = f" (surfaced by: {', '.join(space_ids)})" if space_ids else ""
            lines.append(f"- {source.get('label', edge.get('from'))} feeds {target.get('label', edge.get('to'))}{space_note}")
        return "\n".join(lines)

    def _build_graph(
        self, spaces: list[Space], context: SessionContext, workspace: WorkspaceClient, llm: Any | None,
        user_email: str = "unknown", pre_matched_skill: Any | None = None,
    ):
        by_id = {space.space_id: space for space in spaces}
        node_for_id = {space.space_id: f"space_{index}" for index, space in enumerate(spaces)}
        builder = StateGraph(GraphState)

        def understand_intent(state: GraphState) -> dict[str, Any]:
            graph_context = self._graph_context_text(context.intent_graph, state["question"])
            # A saved skill already maps this standalone question to Genie space IDs. Avoid both
            # intent and catalog-routing LLM calls; explicit FAQ/Atlan questions are excluded before
            # the graph is built, so a skill can never override those governed direct routes.
            intent = (
                IntentResult(intent_summary=state["question"])
                if pre_matched_skill
                else self._understand_intent(state["question"], state["history"], graph_context, llm)
            )

            # Turn identity, `mentions` (turn -> entity/domain), and `refines` (turn -> prior turn)
            # edges are system facts, not left to the LLM, so every turn always links into the graph.
            existing_turns = [node for node in context.intent_graph.get("nodes", []) if isinstance(node, dict) and node.get("type") == "turn"]
            turn_number = len(existing_turns) + 1
            turn_id = f"turn:{turn_number}"
            add_nodes: list[dict[str, Any]] = [
                GraphNode(
                    id=turn_id, type="turn", label=state["question"][:120], question_text=state["question"],
                    first_seen_turn=turn_number, last_seen_turn=turn_number, source="llm",
                ).model_dump()
            ]
            add_edges: list[dict[str, Any]] = []
            if turn_number > 1:
                add_edges.append(
                    GraphEdge(
                        from_=turn_id, to=f"turn:{turn_number - 1}", relation="refines",
                        first_seen_turn=turn_number, source="llm",
                    ).model_dump(by_alias=True)
                )
            for node in intent.add_nodes:
                add_nodes.append(node.model_dump())
                if node.type in ("entity", "domain"):
                    add_edges.append(
                        GraphEdge(
                            from_=turn_id, to=node.id, relation="mentions",
                            first_seen_turn=turn_number, source="llm",
                        ).model_dump(by_alias=True)
                    )
            add_edges.extend(edge.model_dump(by_alias=True) for edge in intent.add_edges)

            merged_graph = SessionContextStore.apply_patch(
                context.intent_graph, {"add_nodes": add_nodes, "add_edges": add_edges}
            )
            reasoning = [f"Understood intent: {intent.intent_summary or state['question']}"]
            if intent.context_recap:
                reasoning.append(f"Carried forward context: {intent.context_recap}")
            return {
                "intent_summary": intent.intent_summary or state["question"],
                "context_recap": intent.context_recap,
                "intent_graph": merged_graph,
                "turn_id": turn_id,
                "reasoning": reasoning,
            }

        builder.add_node("understand_intent", understand_intent)
        builder.add_edge(START, "understand_intent")

        def route(state: GraphState) -> dict[str, Any]:
            active_still_enabled = [sid for sid in context.active_space_ids if sid in by_id]
            is_ai_application_metadata = (
                self._is_ai_application_metadata_question(state["intent_summary"])
                or self._is_ai_application_metadata_question(state["question"])
                or (bool(active_still_enabled) and self._is_metadata_field_follow_up(state["question"]))
            )
            if is_ai_application_metadata:
                return {
                    "selected_space_ids": ["__atlan_ai_application__"],
                    "reasoning": ["Detected an AI Application metadata question -- resolving the active Genie Agent in Atlan."],
                }
            if self._is_genie_catalog_question(state["intent_summary"]) or self._is_genie_catalog_question(state["question"]):
                return {
                    "selected_space_ids": ["__genie_catalog__"],
                    "reasoning": ["Detected a Genie Agent catalog question -- using Atlan AI Application descriptions without querying Genie."],
                }
            if self._is_lineage_question(state["intent_summary"]) or self._is_lineage_question(state["question"]):
                refs = extract_table_refs(state["intent_summary"]) or extract_table_refs(state["question"])
                if refs:
                    tables = ", ".join(f"{c}.{s}.{t}" for c, s, t in refs)
                    note = f"Detected a table-lineage question referencing {tables}"
                else:
                    note = "Detected a table-lineage question"
                return {
                    "selected_space_ids": ["__atlan_lineage__"],
                    "reasoning": [f"{note} -- routing directly to Atlan instead of any Genie space."],
                }
            if self.knowledge and self.knowledge.enabled and self._is_knowledge_question(state["intent_summary"]):
                return {
                    "selected_space_ids": ["__knowledge__"],
                    "reasoning": ["Detected an onboarding, FAQ, or product-guidance question -- routing to the Knowledge assistant."],
                }
            if self._is_glossary_question(state["intent_summary"]) or self._is_glossary_question(state["question"]):
                return {
                    "selected_space_ids": ["__atlan_glossary__"],
                    "reasoning": ["Detected a glossary/business-term question -- routing directly to Atlan instead of any Genie space."],
                }
            if self._is_data_product_question(state["intent_summary"]) or self._is_data_product_question(state["question"]):
                return {
                    "selected_space_ids": ["__atlan_data_product__"],
                    "reasoning": ["Detected a data product question -- routing directly to Atlan instead of any Genie space."],
                }
            # A single active space plus a non-empty context_recap means intent-understanding already
            # recognized this as a same-conversation follow-up -- skip the full-catalog routing call
            # (every space's description/instructions) and just keep talking to that space.
            if len(active_still_enabled) == 1 and state["context_recap"]:
                selected = active_still_enabled
                chosen = f"{by_id[selected[0]].display_name} ({selected[0]})"
                return {
                    "selected_space_ids": selected,
                    "reasoning": [f"Continuing follow-up with the active Genie space: {chosen}."],
                }
            # A saved skill (learned mapping from a prior turn) matches by keyword overlap alone --
            # no LLM call -- so it's tried next, before paying for the full routing-catalog prompt.
            if self.skills:
                matched_skill = pre_matched_skill or self.skills.match(
                    state["intent_summary"] or state["question"], user_email,
                )
                if matched_skill:
                    valid_ids = [sid for sid in matched_skill.space_ids if sid in by_id]
                    if valid_ids:
                        chosen = ", ".join(f"{by_id[sid].display_name} ({sid})" for sid in valid_ids)
                        return {
                            "selected_space_ids": valid_ids,
                            "reasoning": [f"Matched saved skill '{matched_skill.name}' -- routing directly to {chosen} (no routing lookup needed)."],
                        }
            selected, why = self._route(state["intent_summary"], state["history"], spaces, context.active_space_ids, llm)
            if selected:
                chosen = ", ".join(f"{by_id[sid].display_name} ({sid})" for sid in selected if sid in by_id)
                line = f"Routing: {why} Hitting Genie space(s): {chosen}." if why else f"Hitting Genie space(s): {chosen}."
            else:
                line = f"Routing: {why}" if why else "No enabled Genie space matched this question's intent."
            # Offer to save this as a skill only when it went through the full routing decision above
            # (not a skill match or follow-up) and it actually resolved to space(s).
            suggestion = (
                {
                    "question": state["intent_summary"] or state["question"],
                    "space_ids": selected,
                    "display_names": [by_id[sid].display_name for sid in selected if sid in by_id],
                }
                if selected and state["turn_id"] == "turn:1" else None
            )
            return {"selected_space_ids": selected, "reasoning": [line], "skill_suggestion": suggestion}

        builder.add_node("route", route)
        builder.add_edge("understand_intent", "route")

        def no_match(state: GraphState) -> dict[str, Any]:
            return {"final_answer": "No matching Genie space found for this question."}

        builder.add_node("no_match", no_match)
        builder.add_edge("no_match", END)

        def atlan_lineage(state: GraphState) -> dict[str, Any]:
            if not self.atlan or not self.atlan.enabled:
                return {
                    "final_answer": "Atlan isn't configured, so I can't look up table lineage.",
                    "reasoning": ["Atlan client is not configured/enabled; couldn't look up lineage."],
                }
            refs = extract_table_refs(state["intent_summary"]) or extract_table_refs(state["question"])
            trace: list[str] = []
            if not refs:
                # No fully-qualified catalog.schema.table -- try resolving a bare table name
                # (e.g. "lineage of ekpo") against Atlan by name instead of giving up.
                candidates = extract_bare_table_candidates(state["intent_summary"]) or extract_bare_table_candidates(state["question"])
                for candidate in candidates:
                    try:
                        matches = self.atlan.find_table_by_name(candidate)
                    except Exception as exc:
                        logger.warning("Atlan bare-name lookup failed for %s", candidate, exc_info=exc)
                        continue
                    if matches:
                        trace.append(
                            f"No fully-qualified table given; searched Atlan for '{candidate}' by name and "
                            f"found {len(matches)} match(es)."
                        )
                        refs = [(match["catalog"], match["schema"], match["name"]) for match in matches]
                        break
                if not refs:
                    return {
                        "final_answer": "I detected a table lineage question but couldn't find a matching "
                        "table in Atlan -- please include the full catalog.schema.table name.",
                        "reasoning": trace + ["No matching table found in Atlan by name; asked the user to include the full table name."],
                    }
            blocks = []
            direct_lineage: list[TableLineage] = []
            for catalog, schema, table in refs:
                fqtn = f"{catalog}.{schema}.{table}"
                try:
                    context = self.atlan.get_table_context(catalog, schema, table)
                except Exception as exc:
                    logger.warning("Atlan lineage lookup failed for %s", fqtn, exc_info=exc)
                    blocks.append(f"**{fqtn}** — Atlan lookup failed: {exc}")
                    trace.append(f"Atlan lookup for {fqtn} failed: {exc}")
                    continue
                blocks.append(format_context_markdown(fqtn, context))
                upstream_fqtns = [
                    f"{u['catalog']}.{u['schema']}.{u['table']}"
                    for u in context.get("upstream") or [] if u.get("catalog") and u.get("schema") and u.get("table")
                ]
                downstream_fqtns = [
                    f"{d['catalog']}.{d['schema']}.{d['table']}"
                    for d in context.get("downstream") or [] if d.get("catalog") and d.get("schema") and d.get("table")
                ]
                direct_lineage.append(TableLineage(
                    table=fqtn, context_markdown=format_context_markdown(fqtn, context),
                    upstream=upstream_fqtns, downstream=downstream_fqtns,
                ))
                trace.append(
                    f"Looked up {fqtn} in Atlan: {'found' if context.get('found') else 'not found'} in catalog"
                    + (f", {len(context.get('upstream') or [])} upstream, {len(context.get('downstream') or [])} downstream." if context.get("found") else ".")
                )
            return {"final_answer": "\n\n".join(blocks), "reasoning": trace, "direct_lineage": direct_lineage}

        builder.add_node("atlan_lineage", atlan_lineage)
        builder.add_edge("atlan_lineage", END)

        def atlan_glossary(state: GraphState) -> dict[str, Any]:
            if not self.atlan or not self.atlan.enabled:
                return {
                    "final_answer": "Atlan isn't configured, so I can't look up glossary terms.",
                    "reasoning": ["Atlan client is not configured/enabled; couldn't look up the glossary term."],
                }
            candidates = extract_glossary_term_candidates(state["intent_summary"]) or extract_glossary_term_candidates(state["question"])
            if not candidates:
                return {
                    "final_answer": "I detected a glossary question but couldn't tell which term you meant -- "
                    "please name the specific business term.",
                    "reasoning": ["No glossary term candidate could be extracted from the question."],
                }
            trace: list[str] = []
            blocks = []
            for candidate in candidates:
                try:
                    matches = self.atlan.find_glossary_terms(candidate)
                except Exception as exc:
                    logger.warning("Atlan glossary lookup failed for %s", candidate, exc_info=exc)
                    blocks.append(f"**{candidate}** — Atlan lookup failed: {exc}")
                    trace.append(f"Atlan glossary lookup for {candidate} failed: {exc}")
                    continue
                blocks.append(format_glossary_markdown(candidate, matches))
                trace.append(f"Looked up '{candidate}' in Atlan's glossary: {len(matches)} match(es).")
            return {"final_answer": "\n\n".join(blocks), "reasoning": trace}

        builder.add_node("atlan_glossary", atlan_glossary)
        builder.add_edge("atlan_glossary", END)

        def atlan_data_product(state: GraphState) -> dict[str, Any]:
            if not self.atlan or not self.atlan.enabled:
                return {
                    "final_answer": "Atlan isn't configured, so I can't look up data products.",
                    "reasoning": ["Atlan client is not configured/enabled; couldn't look up the data product."],
                }
            candidates = extract_data_product_candidates(state["intent_summary"]) or extract_data_product_candidates(state["question"])
            trace: list[str] = []
            blocks = []
            if not candidates:
                # No specific product name given -- list what's available, like the Genie catalog route.
                try:
                    products = self.atlan.discover_data_products()
                except Exception as exc:
                    logger.warning("Atlan data product discovery failed", exc_info=exc)
                    return {
                        "final_answer": f"Atlan data product lookup failed: {exc}",
                        "reasoning": [f"Atlan data product discovery failed: {exc}"],
                    }
                return {
                    "final_answer": format_data_product_markdown("Data products", products),
                    "reasoning": [f"Listed {len(products)} data product(s) from Atlan."],
                }
            for candidate in candidates:
                try:
                    matches = self.atlan.find_data_products(candidate)
                except Exception as exc:
                    logger.warning("Atlan data product lookup failed for %s", candidate, exc_info=exc)
                    blocks.append(f"**{candidate}** — Atlan lookup failed: {exc}")
                    trace.append(f"Atlan data product lookup for {candidate} failed: {exc}")
                    continue
                blocks.append(format_data_product_markdown(candidate, matches))
                trace.append(f"Looked up '{candidate}' in Atlan's data products: {len(matches)} match(es).")
            return {"final_answer": "\n\n".join(blocks), "reasoning": trace}

        builder.add_node("atlan_data_product", atlan_data_product)
        builder.add_edge("atlan_data_product", END)

        def atlan_ai_application(state: GraphState) -> dict[str, Any]:
            active_ids = [space_id for space_id in context.active_space_ids if space_id in by_id]
            applications: list[dict[str, Any]] = []
            source = "the synchronized Atlan registry"
            if self.atlan and self.atlan.enabled:
                try:
                    applications = self.atlan.discover_ai_applications()
                    source = "live Atlan AI Application metadata"
                except Exception as exc:
                    logger.warning("Live Atlan AI Application lookup failed; using synced metadata", exc_info=exc)

            if not active_ids:
                lowered = state["intent_summary"].lower()
                active_ids = [
                    app.get("space_id") for app in applications
                    if app.get("space_id") in by_id and app.get("name", "").lower() in lowered
                ]
            if not active_ids:
                return {
                    "final_answer": "I couldn't determine which AI Application you mean. Ask a Genie question first or name the Genie Agent.",
                    "reasoning": ["No active or explicitly named Genie Agent was available for the Atlan metadata lookup."],
                }

            live_by_id = {app.get("space_id"): app for app in applications if app.get("space_id")}
            records = [live_by_id.get(space_id) or self._space_metadata(by_id[space_id]) for space_id in active_ids]
            return {
                "final_answer": "\n\n".join(self._format_ai_application_metadata(record) for record in records),
                "reasoning": [f"Loaded Atlan metadata for {len(records)} active AI Application(s) using {source}."],
            }

        builder.add_node("atlan_ai_application", atlan_ai_application)
        builder.add_edge("atlan_ai_application", END)

        def knowledge_answer(state: GraphState) -> dict[str, Any]:
            if not self.knowledge:
                return {"final_answer": "Knowledge endpoint is not configured."}
            question = state["intent_summary"]
            if state["context_recap"]:
                question = f"Context so far: {state['context_recap']}\n\nQuestion: {question}"
            response = self.knowledge.ask(question, workspace)
            if response.error:
                return {
                    "final_answer": "The Knowledge assistant could not answer this question.",
                    "responses": [response],
                    "reasoning": [f"Knowledge assistant request failed: {response.error}"],
                }
            return {
                "final_answer": response.answer,
                "responses": [response],
                "reasoning": ["Knowledge assistant answered from the configured documentation endpoint."],
            }

        builder.add_node("knowledge_answer", knowledge_answer)
        builder.add_edge("knowledge_answer", END)

        def genie_catalog(_: GraphState) -> dict[str, Any]:
            descriptions = {space.space_id: space.description for space in spaces}
            description_source = "the synced Atlan registry"
            if self.atlan and self.atlan.enabled:
                try:
                    applications = self.atlan.discover_ai_applications()
                    descriptions.update({
                        app["space_id"]: app.get("description") or descriptions.get(app["space_id"], "")
                        for app in applications if app.get("space_id") in by_id
                    })
                    description_source = "live Atlan AI Application metadata"
                except Exception as exc:
                    logger.warning("Live Atlan AI Application discovery failed; using synced descriptions", exc_info=exc)
            entries = [
                f"- **{space.display_name}**: {self._crisp_description(descriptions.get(space.space_id, ''))}"
                for space in spaces
            ]
            return {
                "final_answer": "## Available Genie Agents\n\n" + "\n".join(entries),
                "reasoning": [f"Listed {len(entries)} enabled Genie Agent(s) using {description_source}."],
            }

        builder.add_node("genie_catalog", genie_catalog)
        builder.add_edge("genie_catalog", END)

        for space in spaces:
            def query_space(state: GraphState, current: Space = space) -> dict[str, Any]:
                existing_conversation_id = context.genie_conversations.get(current.space_id)
                # Genie already retains context for an open conversation; only new spaces need a recap.
                if existing_conversation_id:
                    genie_question = state["intent_summary"]
                elif state["context_recap"]:
                    genie_question = f"Context so far: {state['context_recap']}\n\nQuestion: {state['intent_summary']}"
                else:
                    genie_question = state["intent_summary"]
                response = self.genie.ask(current, genie_question, existing_conversation_id, workspace)
                outcome = f"failed ({response.error})" if response.error else "answered successfully"
                trace = f"Called Genie space '{current.display_name}' ({current.space_id}) — {outcome}."
                return {"responses": [response], "reasoning": [trace]}

            node_name = node_for_id[space.space_id]
            builder.add_node(node_name, query_space)
            builder.add_edge(node_name, "merge")

        def paths(state: GraphState) -> list[str]:
            if state["selected_space_ids"] == ["__genie_catalog__"]:
                return ["genie_catalog"]
            if state["selected_space_ids"] == ["__atlan_ai_application__"]:
                return ["atlan_ai_application"]
            if state["selected_space_ids"] == ["__atlan_lineage__"]:
                return ["atlan_lineage"]
            if state["selected_space_ids"] == ["__atlan_glossary__"]:
                return ["atlan_glossary"]
            if state["selected_space_ids"] == ["__atlan_data_product__"]:
                return ["atlan_data_product"]
            if state["selected_space_ids"] == ["__knowledge__"]:
                return ["knowledge_answer"]
            selected = [node_for_id[item] for item in state["selected_space_ids"] if item in by_id]
            return selected if selected else ["no_match"]

        builder.add_conditional_edges(
            "route", paths, list(node_for_id.values()) + ["no_match", "genie_catalog", "atlan_ai_application", "atlan_lineage", "atlan_glossary", "atlan_data_product", "knowledge_answer"]
        )

        def merge(state: GraphState) -> dict[str, Any]:
            answer = self._merge(state["intent_summary"], state["responses"], llm)
            successful = [item for item in state["responses"] if not item.error]
            note = f"Synthesized final answer from {len(successful)} source(s)." if len(successful) > 1 else ""
            return {"final_answer": answer, "reasoning": [note] if note else []}

        builder.add_node("merge", merge)
        builder.add_edge("merge", END)
        return builder.compile()

    def _understand_intent(self, question: str, history: str, graph_context: str, llm: Any | None) -> IntentResult:
        if not llm:
            return IntentResult(intent_summary=question)
        prompt = (
            "Analyze the latest user message in the context of the conversation so far. Produce: "
            "1) a self-contained restatement of the user's intent that resolves pronouns and follow-up "
            "references (e.g. 'what about last week' becomes the full standalone question), "
            "2) a short recap of prior context relevant to a system seeing this conversation for the "
            "first time (empty string if this is the first turn), and "
            "3) any new business entities/domains worth remembering (add_nodes: type must be exactly "
            "'entity' or 'domain', never 'turn'/'space'/'table' -- those are managed by the system), plus "
            "any relations between them (add_edges: relation must be exactly 'belongs_to_domain' or "
            "'resolves_to' -- never 'mentions'/'answered_by'/'refines', which are also system-managed). "
            "If the same entity/domain from the known list below recurs, reuse its exact id instead of "
            "minting a new one, so repeated mentions merge onto the same node across turns.\n\n"
            + (f"Known entities/domains/lineage from earlier turns:\n{graph_context}\n\n" if graph_context else "")
            + f"Recent conversation:\n{history}\n\nLatest message:\n{question}"
        )
        try:
            result = llm.with_structured_output(IntentResult).invoke(prompt)
            return result if isinstance(result, IntentResult) else IntentResult(**result)
        except Exception as exc:
            logger.warning("Intent understanding failed; using raw question", exc_info=exc)
            return IntentResult(intent_summary=question)

    @staticmethod
    def _is_lineage_question(text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in _LINEAGE_KEYWORDS)

    @staticmethod
    def _is_glossary_question(text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in _GLOSSARY_KEYWORDS)

    @staticmethod
    def _is_data_product_question(text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in _DATA_PRODUCT_KEYWORDS)

    @staticmethod
    def _is_ai_application_metadata_question(text: str) -> bool:
        lowered = text.lower()
        mentions_application = any(term in lowered for term in ("ai application", "genie agent", "genie space", "atlan"))
        return mentions_application and any(keyword in lowered for keyword in _AI_APPLICATION_METADATA_KEYWORDS)

    @staticmethod
    def _is_metadata_field_follow_up(text: str) -> bool:
        lowered = text.lower()
        matched_fields = {
            keyword for keyword in _AI_APPLICATION_METADATA_KEYWORDS
            if re.search(rf"\b{re.escape(keyword)}\b", lowered)
        }
        return len(matched_fields) >= 2

    @staticmethod
    def _is_knowledge_question(text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in _KNOWLEDGE_KEYWORDS)

    @staticmethod
    def _is_genie_catalog_question(text: str) -> bool:
        lowered = text.lower()
        mentions_agents = "genie agent" in lowered or "genie space" in lowered
        return mentions_agents and any(keyword in lowered for keyword in _GENIE_CATALOG_KEYWORDS)

    @staticmethod
    def _crisp_description(description: str, limit: int = 180) -> str:
        text = " ".join(description.split()).strip(" -")
        if not text:
            return "No Atlan AI Application description is available."
        sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
        if len(sentence) > limit:
            sentence = sentence[:limit].rsplit(" ", 1)[0].rstrip(".,;:") + "."
        elif sentence[-1] not in ".!?":
            sentence += "."
        return sentence

    @staticmethod
    def _space_metadata(space: Space) -> dict[str, Any]:
        return {
            "name": space.display_name,
            "space_id": space.space_id,
            "qualified_name": space.atlan_qualified_name,
            "description": space.description,
            "certificate": space.certificate_status,
            "connector": space.connector,
            "deployment_environment": space.deployment_environment,
            "workspace_name": space.workspace_name,
            "agent_owner": space.agent_owner,
            "domains": space.domains,
            "tags": space.tags,
        }

    @staticmethod
    def _format_ai_application_metadata(application: dict[str, Any]) -> str:
        value = lambda key, default="Not specified": application.get(key) or default
        tags = ", ".join(application.get("tags") or []) or "None"
        domains = ", ".join(application.get("domains") or []) or "Not specified"
        return "\n".join([
            f"## {value('name', 'Atlan AI Application')}",
            "",
            str(value("description", "No description is available.")),
            "",
            f"- **Genie space ID:** `{value('space_id')}`",
            f"- **Owner:** {value('agent_owner')}",
            f"- **Certification:** {value('certificate')}",
            f"- **Domain:** {domains}",
            f"- **Deployment environment:** {value('deployment_environment')}",
            f"- **Workspace:** {value('workspace_name')}",
            f"- **Connector:** {value('connector')}",
            f"- **Tags:** {tags}",
            f"- **Atlan qualified name:** `{value('qualified_name')}`",
        ])

    def _route(self, intent_summary: str, history: str, spaces: list[Space], active: list[str], llm: Any | None) -> tuple[list[str], str]:
        if llm:
            catalog = "\n".join(
                f"- {space.space_id}: {space.display_name} — {space.description}"
                + (f" (routing hint: {space.instructions})" if space.instructions else "")
                for space in spaces
            )
            prompt = (
                "Route the latest question to the smallest sufficient set of Genie spaces, based on "
                "whether each space's description genuinely covers the question's intent. "
                f"Choose at most {self.settings.max_routed_spaces} exact IDs. Routing hints describe when "
                "to pick a space; they are not part of the question itself. If none of the spaces' "
                "descriptions plausibly answer this question, return an empty space_ids list -- do not "
                "guess or default to any space just to return something. Also give a short 'reasoning' "
                "explaining, in terms a user can follow, why these space(s) were picked or why none were.\n\n"
                f"Spaces:\n{catalog}\n\nRecent conversation:\n{history}\n\nQuestion:\n{intent_summary}"
            )
            try:
                decision = llm.with_structured_output(RouteDecision).invoke(prompt)
                allowed = {space.space_id for space in spaces}
                if isinstance(decision, RouteDecision):
                    routed_ids, why = decision.space_ids, decision.reasoning
                else:
                    routed_ids, why = decision.get("space_ids", []), decision.get("reasoning", "")
                # A clean LLM response is authoritative even when empty -- that's a genuine "no match",
                # not a failure, so it must not fall through to the heuristic fallback below.
                chosen = [item for item in routed_ids if item in allowed][: self.settings.max_routed_spaces]
                return chosen, why or ("No space's description matched this question." if not chosen else "")
            except Exception as exc:
                logger.warning("LLM routing failed; using deterministic fallback", exc_info=exc)
                return self._heuristic_route(intent_summary, spaces, active, why=f"LLM routing call failed ({exc}); ")
        return self._heuristic_route(intent_summary, spaces, active, why="No LLM endpoint configured; ")

    def _heuristic_route(
        self, question: str, spaces: list[Space], active: list[str], why: str = ""
    ) -> tuple[list[str], str]:
        words = set(question.lower().replace("_", " ").split())
        scored = []
        matched_words: dict[str, list[str]] = {}
        for space in spaces:
            haystack = f"{space.display_name} {space.description}".lower().replace("_", " ")
            hits = [word for word in words if len(word) > 2 and word in haystack]
            scored.append((len(hits), space.space_id))
            if hits:
                matched_words[space.space_id] = hits
        matches = [space_id for score, space_id in sorted(scored, reverse=True) if score > 0]
        if matches:
            chosen = matches[: self.settings.max_routed_spaces]
            by_id = {space.space_id: space.display_name for space in spaces}
            detail = "; ".join(
                f"{by_id[sid]} matched keywords: {', '.join(matched_words[sid][:5])}" for sid in chosen
            )
            return chosen, f"{why}used keyword-overlap fallback (imprecise -- fix the LLM endpoint for accurate routing). {detail}."
        # No keyword overlap with any space's description: stay on the active conversation's
        # space(s) if there is one (a legitimate follow-up), otherwise honestly report no match
        # instead of defaulting to an arbitrary space.
        still_enabled = [item for item in active if any(space.space_id == item for space in spaces)]
        if still_enabled:
            return still_enabled[: self.settings.max_routed_spaces], (
                "No keyword overlap with any space's description; continuing the active conversation's space."
            )
        return [], "No space's description matched this question (keyword fallback), and no active conversation to continue."

    def _merge(self, question: str, responses: list[SourceAnswer], llm: Any | None) -> str:
        successful = [item for item in responses if not item.error]
        if not successful:
            details = "; ".join(f"{item.display_name}: {item.error}" for item in responses)
            return f"I couldn't get an answer from the selected Genie spaces. {details}"
        if len(successful) == 1:
            return successful[0].answer
        source_text = "\n\n".join(f"[{item.display_name}]\n{item.answer}" for item in successful)
        if llm:
            try:
                result = llm.invoke(
                    "Synthesize a concise, coherent answer using only the supplied Genie results. "
                    "Preserve material disagreements and attribute them to the named source.\n\n"
                    f"Question: {question}\n\nResults:\n{source_text}"
                )
                return str(result.content)
            except Exception as exc:
                logger.warning("LLM merge failed; returning attributed source answers", exc_info=exc)
        return source_text

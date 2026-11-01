from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Fixed ontology for the session intent graph — kept small and closed so intent summaries
# across different questions in the same session merge into one connected graph instead of
# accumulating disconnected, free-text relations.
GraphNodeType = Literal["turn", "entity", "domain", "table", "space"]
GraphRelation = Literal["mentions", "answered_by", "resolves_to", "belongs_to_domain", "feeds", "refines"]
GraphSource = Literal["llm", "genie", "atlan"]

SPACE_DISPLAY_NAME_MAX_LENGTH = 120
SPACE_DESCRIPTION_MAX_LENGTH = 2000
SPACE_QUALIFIED_NAME_MAX_LENGTH = 1000


class GraphNode(BaseModel):
    id: str
    type: GraphNodeType
    label: str
    first_seen_turn: int | None = None
    last_seen_turn: int | None = None
    source: GraphSource = "llm"
    space_ids: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    confidence: float | None = None
    question_text: str | None = None
    catalog: str | None = None
    table_schema: str | None = None
    table_name: str | None = None
    atlan_guid: str | None = None


class GraphEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    from_: str = Field(alias="from")
    to: str
    relation: GraphRelation
    first_seen_turn: int | None = None
    source: GraphSource = "llm"
    confidence: float | None = None


class SpaceBase(BaseModel):
    display_name: str = Field(min_length=1, max_length=SPACE_DISPLAY_NAME_MAX_LENGTH)
    description: str = Field(default="", max_length=SPACE_DESCRIPTION_MAX_LENGTH)
    instructions: str = Field(default="", max_length=8000)
    connection_params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    # "atlan" rows are populated/refreshed by the AI Application sync; "manual" rows are admin-created.
    source: Literal["manual", "atlan"] = "manual"
    atlan_qualified_name: str | None = Field(default=None, max_length=SPACE_QUALIFIED_NAME_MAX_LENGTH)
    # Read from the "AI Apps Metadata" custom metadata set on the Atlan AIApplication asset.
    deployment_environment: str | None = None
    workspace_name: str | None = None
    agent_owner: str | None = None
    certificate_status: str | None = None
    domains: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    connector: str | None = None
    readme: str | None = None

    @field_validator("display_name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("display_name cannot be blank")
        return cleaned


class SpaceUpdate(BaseModel):
    """Admin-tunable fields only -- display_name/description/etc. are Atlan-owned and
    refreshed by the sync; only routing behavior not modeled in Atlan can be overridden here.
    """
    instructions: str | None = Field(default=None, max_length=8000)
    connection_params: dict[str, Any] | None = None
    enabled: bool | None = None


class Space(SpaceBase):
    model_config = ConfigDict(from_attributes=True)
    space_id: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    last_synced_at: datetime | None = None


class AtlanSyncResult(BaseModel):
    added: int
    updated: int
    skipped_no_space_id: int
    total_seen: int


class GenieTableCandidate(BaseModel):
    full_name: str
    table_type: str = ""
    databricks_comment: str = ""
    atlan_description: str = ""
    atlan_owners: list[str] = Field(default_factory=list)
    certificate_status: str = ""
    atlan_found: bool = False
    match_score: int = Field(default=0, ge=0, le=100)
    match_terms: list[str] = Field(default_factory=list)
    recommended: bool = False


class GenieTableDiscovery(BaseModel):
    tables: list[GenieTableCandidate]
    accessible_count: int
    offset: int = 0
    next_offset: int | None = None
    warehouse_id: str


class GenieSpaceDraft(BaseModel):
    title: str = Field(min_length=5, max_length=120)
    description: str = Field(min_length=20, max_length=2000)
    table_names: list[str] = Field(min_length=1, max_length=50)

    @field_validator("title", "description")
    @classmethod
    def clean_builder_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("table_names")
    @classmethod
    def clean_table_names(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            name = item.strip().lower()
            parts = name.split(".")
            if len(parts) != 3 or not all(part and part.replace("_", "").isalnum() for part in parts):
                raise ValueError(f"invalid table name '{item}'; expected catalog.schema.table")
            if name not in cleaned:
                cleaned.append(name)
        return cleaned


class GenieSpacePreview(BaseModel):
    title: str
    description: str
    warehouse_id: str
    parent_path: str | None = None
    tables: list[GenieTableCandidate]
    confirmation_token: str


class GenieSpaceCreate(GenieSpaceDraft):
    confirmation_token: str = Field(min_length=64, max_length=64)
    confirmed: Literal[True]


class GenieSpaceCreated(BaseModel):
    space: Space
    genie_url: str
    tables: list[str]


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=16000)


class QueryResultColumn(BaseModel):
    name: str
    type_name: str | None = None


class QueryResult(BaseModel):
    title: str | None = None
    columns: list[QueryResultColumn]
    rows: list[list[str | None]]
    truncated: bool = False


class SourceAnswer(BaseModel):
    space_id: str
    display_name: str
    answer: str
    conversation_id: str | None = None
    error: str | None = None
    # catalog.schema.table names referenced by Genie's generated SQL, used to look up
    # business context/lineage in Atlan.
    referenced_tables: list[str] = Field(default_factory=list)
    query_results: list[QueryResult] = Field(default_factory=list)


class TableLineage(BaseModel):
    table: str
    context_markdown: str
    upstream: list[str] = Field(default_factory=list)
    downstream: list[str] = Field(default_factory=list)


class UsageEntryOut(BaseModel):
    question: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None = None
    created_at: datetime


class UsageSummaryOut(BaseModel):
    question_count: int
    total_tokens: int
    total_cost_usd: float | None = None


class SkillBase(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    trigger_phrases: list[str] = Field(
        default_factory=list, min_length=1,
        description="Example questions/keywords that should match this skill (e.g. 'campaign spend', "
        "'marketing roi'). Matched by keyword overlap against future questions -- no LLM call needed.",
    )
    space_ids: list[str] = Field(min_length=1, description="Genie space(s) to route to when this skill matches.")
    visibility: Literal["private", "shared"] = "private"

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("name cannot be blank")
        return cleaned

    @field_validator("trigger_phrases")
    @classmethod
    def clean_phrases(cls, value: list[str]) -> list[str]:
        cleaned = [phrase.strip() for phrase in value if phrase and phrase.strip()]
        if not cleaned:
            raise ValueError("at least one trigger phrase is required")
        return cleaned


class SkillCreate(SkillBase):
    pass


class SkillUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    trigger_phrases: list[str] | None = None
    space_ids: list[str] | None = None
    visibility: Literal["private", "shared"] | None = None


class Skill(SkillBase):
    skill_id: str
    owner_email: str
    created_at: datetime
    updated_at: datetime


class SkillSuggestion(BaseModel):
    """Offered back to the user after a normal (non-skill, non-follow-up) routing decision, so
    they can save it as a reusable skill instead of paying the full routing-catalog cost again."""
    question: str
    space_ids: list[str]
    display_names: list[str]


class ChatResponse(BaseModel):
    answer: str
    answered_by: list[str]
    sources: list[SourceAnswer]
    lineage: list[TableLineage] = Field(default_factory=list)
    reasoning: list[str] = Field(
        default_factory=list,
        description="Step-by-step trace of how intent was understood, which space(s) were "
        "considered/selected and why, and what was actually called -- shown to the user for transparency.",
    )
    skill_suggestion: SkillSuggestion | None = Field(
        default=None,
        description="Present only when this turn was routed via full LLM/heuristic routing (not an "
        "existing skill, follow-up, or direct Atlan lookup) -- clients may offer to save it as a skill.",
    )
    token_usage: dict[str, int] | None = Field(
        default=None,
        description="Prompt/completion/total token counts for this turn's LLM calls, read from the "
        "MLflow trace autologged for the request. None if no LLM was called or usage wasn't captured.",
    )
    estimated_cost_usd: float | None = Field(
        default=None,
        description="Approximate cost for this turn's LLM token usage, computed from the configured "
        "per-million-token price -- not real billing data. None if pricing isn't configured or usage "
        "wasn't available.",
    )

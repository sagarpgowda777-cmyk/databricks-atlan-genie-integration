from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    registry_table: str
    session_context_table: str
    chat_sessions_table: str
    chat_messages_table: str
    max_sessions_per_user: int
    skills_table: str
    max_skills_per_user: int
    usage_history_table: str
    pg_host: str
    pg_port: str
    pg_database: str
    pg_user: str
    pg_sslmode: str
    pg_endpoint: str
    llm_endpoint: str | None
    knowledge_endpoint: str | None
    genie_builder_warehouse_id: str = ""
    genie_builder_parent_path: str = ""
    genie_builder_signing_secret: str = ""
    genie_builder_max_tables: int = 500
    genie_timeout_seconds: int = 300
    knowledge_timeout_seconds: int = 120
    max_routed_spaces: int = 3
    atlan_base_url: str = ""
    atlan_client_id: str = ""
    atlan_client_secret: str = ""
    mlflow_tracking_uri: str = "databricks"
    mlflow_experiment_id: str = ""
    # $ per 1M input/output tokens for the configured llm_endpoint model -- used only to turn the
    # token counts MLflow already captures per trace into an estimated cost; 0 disables the estimate
    # since it's not real billing data, just a configurable approximation.
    llm_input_price_per_million: float = 0.0
    llm_output_price_per_million: float = 0.0

    @classmethod
    def from_env(cls) -> "Settings":
        pg_host = os.getenv("PGHOST", "")
        pg_database = os.getenv("PGDATABASE", "")
        pg_user = os.getenv("PGUSER", "")
        pg_endpoint = os.getenv("DATABASE_ENDPOINT", "")
        if not (pg_host and pg_database and pg_user):
            raise RuntimeError("PGHOST, PGDATABASE, and PGUSER are required (attach a database resource)")
        if not pg_endpoint:
            raise RuntimeError("DATABASE_ENDPOINT is required (add a valueFrom entry for the database resource key)")
        return cls(
            registry_table=os.getenv("GENIE_REGISTRY_TABLE", "public.registry_table"),
            session_context_table=os.getenv("GENIE_SESSION_CONTEXT_TABLE", "public.session_context"),
            chat_sessions_table=os.getenv("GENIE_CHAT_SESSIONS_TABLE", "public.chat_sessions"),
            chat_messages_table=os.getenv("GENIE_CHAT_MESSAGES_TABLE", "public.chat_messages"),
            max_sessions_per_user=int(os.getenv("MAX_SESSIONS_PER_USER", "5")),
            skills_table=os.getenv("GENIE_SKILLS_TABLE", "public.keystone_skills"),
            max_skills_per_user=int(os.getenv("MAX_SKILLS_PER_USER", "10")),
            usage_history_table=os.getenv("GENIE_USAGE_HISTORY_TABLE", "public.keystone_usage_history"),
            pg_host=pg_host,
            pg_port=os.getenv("PGPORT", "5432"),
            pg_database=pg_database,
            pg_user=pg_user,
            pg_sslmode=os.getenv("PGSSLMODE", "require"),
            pg_endpoint=pg_endpoint,
            llm_endpoint=os.getenv("DATABRICKS_LLM_ENDPOINT") or None,
            knowledge_endpoint=os.getenv("DATABRICKS_KNOWLEDGE_ENDPOINT") or None,
            genie_builder_warehouse_id=os.getenv("GENIE_BUILDER_WAREHOUSE_ID", ""),
            genie_builder_parent_path=os.getenv("GENIE_BUILDER_PARENT_PATH", ""),
            genie_builder_signing_secret=os.getenv("GENIE_BUILDER_SIGNING_SECRET", ""),
            genie_builder_max_tables=int(os.getenv("GENIE_BUILDER_MAX_TABLES", "500")),
            genie_timeout_seconds=int(os.getenv("GENIE_TIMEOUT_SECONDS", "300")),
            knowledge_timeout_seconds=int(os.getenv("KNOWLEDGE_TIMEOUT_SECONDS", "120")),
            max_routed_spaces=int(os.getenv("MAX_ROUTED_SPACES", "3")),
            atlan_base_url=os.getenv("ATLAN_BASE_URL", ""),
            atlan_client_id=os.getenv("ATLAN_CLIENT_ID", ""),
            atlan_client_secret=os.getenv("ATLAN_CLIENT_SECRET", ""),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", ""),
            mlflow_experiment_id=os.getenv("MLFLOW_EXPERIMENT_ID", ""),
            llm_input_price_per_million=float(os.getenv("LLM_INPUT_PRICE_PER_MILLION", "0")),
            llm_output_price_per_million=float(os.getenv("LLM_OUTPUT_PRICE_PER_MILLION", "0")),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()




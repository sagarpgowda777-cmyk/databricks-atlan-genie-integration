# Databricks–Atlan Genie Integration

This repository contains the Python backend for **Keystone Genie**, a FastAPI service that routes questions to Databricks Genie spaces and can enrich answers with Atlan metadata and a Databricks knowledge endpoint.

## What it does

- Routes a user's question to one or more enabled Genie spaces and combines the answers.
- Streams progress and the final response over Server-Sent Events (SSE).
- Uses Atlan for AI application discovery, table context, data products, glossary terms, and lineage when configured.
- Supports creating Genie spaces from tables visible to the signed-in user.
- Persists the space registry, chat history and session context, skills, and usage history in Lakebase PostgreSQL.
- Exposes APIs for managing spaces, skills, sessions, and usage.

## Project layout

```text
backend_integration/
  api.py               FastAPI application factory and HTTP routes
  config.py            Environment-based settings
  orchestrator.py      Question routing and answer orchestration
  genie_client.py      Databricks Genie MCP client
  genie_builder.py     Genie space table discovery, preview, and creation
  atlan_client.py      Atlan metadata and lineage client
  knowledge_client.py  Databricks knowledge endpoint client
  registry.py          Lakebase-backed Genie space registry
  session_store.py     Persisted chat and routing context
  skills_store.py      User skill storage
  usage_store.py       Per-user usage history
  memory.py            In-process conversation memory
  models.py            API request and response models
```

## Requirements

- Python 3.10 or later.
- A Databricks workspace and an authenticated Databricks SDK environment.
- A Lakebase PostgreSQL database and a database endpoint configured for credential generation.
- At least one Genie space registered and enabled to answer Genie questions.
- The Python packages imported by the service: FastAPI, Uvicorn, Pydantic, Psycopg 3, Databricks SDK, Databricks MCP, LangChain Core, LangGraph, Requests, and MLflow. The optional Databricks LLM integration also uses `databricks-langchain`.

This repository does not currently include a dependency lockfile or deployment manifest. Install dependencies through your organization's approved environment or deployment process, and pin versions there.

## Configuration

The application reads configuration from environment variables. The four required variables are `PGHOST`, `PGDATABASE`, `PGUSER`, and `DATABASE_ENDPOINT`.

| Variable | Required | Default | Description |
|---|---:|---|---|
| `PGHOST` | Yes | — | Lakebase PostgreSQL host. |
| `PGDATABASE` | Yes | — | PostgreSQL database name. |
| `PGUSER` | Yes | — | PostgreSQL database user. |
| `DATABASE_ENDPOINT` | Yes | — | Lakebase endpoint identifier used by the Databricks SDK to generate short-lived database credentials. |
| `PGPORT` | No | `5432` | PostgreSQL port. |
| `PGSSLMODE` | No | `require` | PostgreSQL TLS mode. |
| `GENIE_REGISTRY_TABLE` | No | `public.registry_table` | Genie space registry table. |
| `GENIE_SESSION_CONTEXT_TABLE` | No | `public.session_context` | Persisted routing and conversation context table. |
| `GENIE_CHAT_SESSIONS_TABLE` | No | `public.chat_sessions` | Chat session table. |
| `GENIE_CHAT_MESSAGES_TABLE` | No | `public.chat_messages` | Chat message table. |
| `GENIE_SKILLS_TABLE` | No | `public.keystone_skills` | User skills table. |
| `GENIE_USAGE_HISTORY_TABLE` | No | `public.keystone_usage_history` | Usage history table. |
| `MAX_SESSIONS_PER_USER` | No | `5` | Maximum retained chat sessions per user. |
| `MAX_SKILLS_PER_USER` | No | `10` | Maximum skills per user. |
| `DATABRICKS_LLM_ENDPOINT` | No | Unset | Databricks model-serving endpoint used for LLM routing and answer orchestration. Without it, LLM-assisted routing is unavailable. |
| `DATABRICKS_KNOWLEDGE_ENDPOINT` | No | Unset | Optional Databricks Responses API endpoint for knowledge questions. |
| `GENIE_BUILDER_WAREHOUSE_ID` | No | Empty | SQL warehouse used by Genie space table discovery and creation. |
| `GENIE_BUILDER_PARENT_PATH` | No | Empty | Optional workspace parent path for created Genie spaces. |
| `GENIE_BUILDER_SIGNING_SECRET` | No | Empty | Secret used to sign Genie space preview confirmations. The builder is enabled only when this and the warehouse ID are set. |
| `GENIE_BUILDER_MAX_TABLES` | No | `500` | Maximum table results requested by the builder. |
| `GENIE_TIMEOUT_SECONDS` | No | `300` | Maximum time allowed for Genie requests. |
| `KNOWLEDGE_TIMEOUT_SECONDS` | No | `120` | Timeout for the knowledge endpoint request. |
| `MAX_ROUTED_SPACES` | No | `3` | Maximum number of Genie spaces selected for a question. |
| `ATLAN_BASE_URL` | No | Empty | Atlan tenant base URL. |
| `ATLAN_CLIENT_ID` | No | Empty | Atlan API client ID. |
| `ATLAN_CLIENT_SECRET` | No | Empty | Atlan API client secret. |
| `MLFLOW_TRACKING_URI` | No | Empty | MLflow tracking URI, for example `databricks`. |
| `MLFLOW_EXPERIMENT_ID` | No | Empty | When set, enables MLflow experiment selection and LangChain autologging. |
| `LLM_INPUT_PRICE_PER_MILLION` | No | `0` | Optional estimated input-token price in USD per million tokens. Zero disables cost estimates. |
| `LLM_OUTPUT_PRICE_PER_MILLION` | No | `0` | Optional estimated output-token price in USD per million tokens. Zero disables cost estimates. |

Store credentials in your deployment platform's secret manager. Do not commit credentials or local environment files.

### Lakebase tables

Before starting the service, provision the configured registry, session context, chat sessions, chat messages, skills, and usage history tables in the database. The store classes contain the table definitions in their `ensure_table()` methods, but the application does not run those methods automatically. Provision tables through an authorized migration or administrator; the application role may not have permission to create tables in the `public` schema. The startup sequence reads the registry table, so the registry must exist and be accessible.

The application uses Databricks-generated, short-lived PostgreSQL credentials for database connections. The Databricks identity running the service needs permission to generate credentials for the configured database endpoint and the database grants required to read/write the tables.

## Run locally

After installing the required dependencies and configuring Databricks authentication and the required environment variables, start the FastAPI app factory with Uvicorn:

```powershell
uvicorn backend_integration.api:create_app --factory --host 127.0.0.1 --port 8000
```

The service listens at `http://127.0.0.1:8000`. FastAPI's interactive API documentation is available at `/docs`, and its OpenAPI schema at `/openapi.json`.

For Databricks Apps, enable **User authorization**. The chat and Genie builder endpoints require the `X-Forwarded-Access-Token` header provided by the app platform so Databricks calls run on behalf of the signed-in user. User identity headers such as `X-Forwarded-Email` are used to scope user-specific sessions, skills, and usage.

## API overview

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Basic service health and LLM configuration status. |
| `GET` | `/api/me` | Returns the forwarded user identity, if present. |
| `GET` | `/api/spaces` | List registered Genie spaces. |
| `PATCH` | `/api/spaces/{space_id}` | Update a space. |
| `DELETE` | `/api/spaces/{space_id}` | Delete a space from the registry. |
| `POST` | `/api/spaces/sync-atlan` | Discover and sync Atlan AI applications. Requires Atlan configuration. |
| `GET` | `/api/spaces/builder/tables` | Discover tables visible to the signed-in user. |
| `POST` | `/api/spaces/builder/preview` | Preview a proposed Genie space and receive a confirmation token. |
| `POST` | `/api/spaces/builder/create` | Create the confirmed Genie space and register it. |
| `POST` | `/api/chat` | Ask a question and receive a complete response. |
| `POST` | `/api/chat/stream` | Ask a question and receive progress/final events as SSE. |
| `GET` | `/api/sessions` | List the caller's chat sessions. |
| `GET` | `/api/sessions/{session_id}/messages` | Get messages for a chat session. |
| `DELETE` | `/api/sessions/{session_id}` | Clear in-process memory for a session. |
| `GET` | `/api/usage` | Get the caller's usage summary and recent entries. |
| `GET` | `/api/usage/export` | Get the caller's full retained usage history. |
| `GET` | `/api/skills` | List the caller's skills. |
| `POST` | `/api/skills` | Create a skill. |
| `PATCH` | `/api/skills/{skill_id}` | Update a skill. |
| `DELETE` | `/api/skills/{skill_id}` | Delete a skill. |

The chat request body uses `session_id` and `message`. See the generated `/docs` page for the complete request and response schemas.

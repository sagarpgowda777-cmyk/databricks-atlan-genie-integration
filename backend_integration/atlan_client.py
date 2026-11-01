"""Backend Atlan client — enriches a Genie answer with catalog context and lineage.

Adapted from atlan_connection.py's standalone CLI script into a reusable,
importable client for the FastAPI app: same Keycloak client_credentials auth
and lineage-traversal logic, but callable per-request from the orchestrator
instead of run interactively.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_DATABRICKS_TYPE_NAMES = {"Table", "View", "MaterialisedView", "DatabricksTable"}
# catalog.schema.table (optionally backtick-quoted) referenced in a Genie SQL query.
_TABLE_REF_RE = re.compile(
    r"`?([A-Za-z_][A-Za-z0-9_]*)`?\.`?([A-Za-z_][A-Za-z0-9_]*)`?\.`?([A-Za-z_][A-Za-z0-9_]*)`?"
)
_MAX_TABLES_PER_ANSWER = 3
# Common words that show up in a lineage question but are never a table name themselves
# (e.g. "Can you provide me lineage of ekpo" -> only "ekpo" is a real candidate).
_BARE_NAME_STOPWORDS = {
    "a", "an", "the", "of", "for", "me", "you", "can", "could", "please", "provide", "give",
    "show", "get", "tell", "about", "table", "tables", "lineage", "upstream", "downstream",
    "data", "flow", "source", "sources", "come", "comes", "from", "does", "is", "are", "and",
    "or", "to", "in", "on", "with", "this", "that", "it", "its", "what", "which", "who",
}
_BARE_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Common words that show up in a glossary question but are never the term itself
# (e.g. "Can you provide me glossary details for matdoc" -> only "matdoc" is a candidate).
_GLOSSARY_STOPWORDS = {
    "a", "an", "the", "of", "for", "me", "you", "can", "could", "please", "provide", "give",
    "show", "get", "tell", "about", "glossary", "term", "terms", "definition", "meaning",
    "details", "detail", "data", "and", "or", "to", "in", "on", "with", "this", "that", "it",
    "its", "what", "which", "who", "is", "are", "from", "does", "mean", "means", "explain",
}

# Custom metadata set + field where a Genie space ID (32-char hex) is stored on an AIApplication asset.
# "AI Apps Metadata" is the CM set shown in Atlan's UI for AI Application assets; "Agent ID"
# holds the Genie space ID directly. Verified live (2026-08-17): Atlan's search/indexsearch
# endpoint never returns businessAttributes/relationshipAttributes regardless of what's
# requested — GET /api/meta/entity/guid/{guid} must be used instead. businessAttributes there
# are keyed by internal CM/field UUIDs (e.g. "NaQzmvQ3A4TMLn2Jeaeo5s"), not display names, so
# the UUID -> display name mapping must be resolved via GET /api/meta/types/typedefs.
_AI_APP_CM_SET = "AI Apps Metadata"
_AI_APP_CM_FIELD_AGENT_ID = "Agent ID"
_AI_APP_CM_FIELD_DEPLOYMENT_ENV = "Deployment Environment"
_AI_APP_CM_FIELD_WORKSPACE_NAME = "Workspace Name"
_AI_APP_CM_FIELD_AGENT_OWNER = "Agent Owner"
_HEX32_RE = re.compile(r"[0-9a-f]{32}")


def _parse_qualified_name(qualified_name: str) -> tuple[str, str, str]:
    """Fallback: derive (catalog, schema, table) from an Atlan qualifiedName
    when databaseName/schemaName/name attributes aren't present (common on
    the lineage endpoint's slimmer attribute set).
    """
    if not qualified_name:
        return ("", "", "")
    parts = [p for p in qualified_name.split("/") if p]
    if len(parts) < 3:
        return ("", "", "")
    table, schema, catalog = parts[-1], parts[-2], parts[-3]
    return (catalog, schema, table)


def extract_table_refs(sql_text: str) -> list[tuple[str, str, str]]:
    """Pull unique (catalog, schema, table) tuples out of a SQL string, in
    order of first appearance, capped at _MAX_TABLES_PER_ANSWER.
    """
    if not sql_text:
        return []
    seen: dict[tuple[str, str, str], None] = {}
    for match in _TABLE_REF_RE.finditer(sql_text):
        ref = (match.group(1), match.group(2), match.group(3))
        seen.setdefault(ref, None)
        if len(seen) >= _MAX_TABLES_PER_ANSWER:
            break
    return list(seen)


def extract_bare_table_candidates(text: str, limit: int = 3) -> list[str]:
    """Pull plausible bare table-name tokens (e.g. "ekpo" in "lineage of ekpo") out of a
    question that doesn't contain a fully-qualified catalog.schema.table reference.
    """
    if not text:
        return []
    seen: dict[str, None] = {}
    for match in _BARE_WORD_RE.finditer(text):
        word = match.group(0)
        if word.lower() in _BARE_NAME_STOPWORDS or len(word) < 3:
            continue
        seen.setdefault(word, None)
        if len(seen) >= limit:
            break
    return list(seen)


def extract_glossary_term_candidates(text: str, limit: int = 3) -> list[str]:
    """Pull plausible glossary-term tokens (e.g. "matdoc" in "glossary details for matdoc")
    out of a glossary/business-term question.
    """
    if not text:
        return []
    seen: dict[str, None] = {}
    for match in _BARE_WORD_RE.finditer(text):
        word = match.group(0)
        if word.lower() in _GLOSSARY_STOPWORDS or len(word) < 3:
            continue
        seen.setdefault(word, None)
        if len(seen) >= limit:
            break
    return list(seen)


# Common words that show up in a data-product question but are never the product name itself
# (e.g. "Can you provide me the customer 360 data product" -> only "customer"/"360" are candidates).
_DATA_PRODUCT_STOPWORDS = {
    "a", "an", "the", "of", "for", "me", "you", "can", "could", "please", "provide", "give",
    "show", "get", "tell", "about", "data", "product", "products", "details", "detail",
    "and", "or", "to", "in", "on", "with", "this", "that", "it", "its", "what", "which",
    "who", "is", "are", "from", "does", "available", "list", "all",
}


def extract_data_product_candidates(text: str, limit: int = 3) -> list[str]:
    """Pull plausible data-product-name tokens out of a data-product question."""
    if not text:
        return []
    seen: dict[str, None] = {}
    for match in _BARE_WORD_RE.finditer(text):
        word = match.group(0)
        if word.lower() in _DATA_PRODUCT_STOPWORDS or len(word) < 3:
            continue
        seen.setdefault(word, None)
        if len(seen) >= limit:
            break
    return list(seen)


def _extract_space_id_from_biz(cm_fields: dict, fallback_text: str = "") -> str | None:
    """Extract a Genie space_id (32-char hex) from the resolved "AI Apps Metadata" fields
    (already keyed by display name — see AtlanClient._resolve_ai_app_cm), falling back to
    scanning `fallback_text` (e.g. qualifiedName/description/readme).
    """
    val = (cm_fields.get(_AI_APP_CM_FIELD_AGENT_ID) or "").strip()
    if val:
        match = _HEX32_RE.search(val)
        if match:
            return match.group(0)
    if fallback_text:
        match = _HEX32_RE.search(fallback_text)
        if match:
            return match.group(0)
    return None


def _extract_app_metadata_cm(cm_fields: dict) -> dict[str, str | None]:
    """Pull deployment_environment/workspace_name/agent_owner out of the resolved
    "AI Apps Metadata" fields dict. Missing fields are None.
    """
    return {
        "deployment_environment": (cm_fields.get(_AI_APP_CM_FIELD_DEPLOYMENT_ENV) or "").strip() or None,
        "workspace_name": (cm_fields.get(_AI_APP_CM_FIELD_WORKSPACE_NAME) or "").strip() or None,
        "agent_owner": (cm_fields.get(_AI_APP_CM_FIELD_AGENT_OWNER) or "").strip() or None,
    }


def _extract_domain_names(entity: dict) -> list[str]:
    names: dict[str, None] = {}

    def collect(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            attrs = value.get("attributes") or {}
            name = value.get("displayText") or value.get("name") or attrs.get("name") or attrs.get("displayName")
            if isinstance(name, str) and name.strip():
                names.setdefault(name.strip(), None)

    attrs = entity.get("attributes") or {}
    relationships = entity.get("relationshipAttributes") or {}
    for key, value in relationships.items():
        if "domain" in key.lower():
            collect(value)
    for key in ("domains", "domain", "domainNames"):
        collect(attrs.get(key))
    return list(names)


def _extract_readme_text(entity: dict) -> str:
    """Pull the linked Readme entity's markdown content off an Atlan entity's
    relationshipAttributes.readme, as a dict or a one-item list.
    """
    readme_rel = (entity.get("relationshipAttributes") or {}).get("readme") or {}
    if isinstance(readme_rel, list):
        readme_rel = readme_rel[0] if readme_rel else {}
    readme_attrs = readme_rel.get("attributes") or {}
    return (readme_attrs.get("description") or readme_attrs.get("userDescription") or "").strip()


class AtlanClient:
    """Looks up a Databricks table in Atlan and summarizes its business
    context and upstream lineage (how it was built / what feeds it).
    """

    def __init__(self, base_url: str, client_id: str, client_secret: str):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self._token_url = f"{self.base_url}/auth/realms/default/protocol/openid-connect/token"
        self._token_cache: dict[str, Any] = {}
        self._cm_field_map_cache: dict[str, str] | None = None  # field internal id -> display name
        self._cm_set_name_cache: str | None = None  # "AI Apps Metadata" -> internal set id
        self._tag_name_cache: dict[str, str] | None = None  # classification internal id -> display name
        self._domain_name_cache: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.client_id and self.client_secret)

    @staticmethod
    def _decode_jwt_exp(token: str) -> float:
        try:
            seg = token.split(".")[1]
            seg += "=" * (4 - len(seg) % 4)
            return float(json.loads(base64.b64decode(seg)).get("exp", 0))
        except Exception:
            return 0.0

    def _get_token(self) -> str:
        cached = self._token_cache.get("access_token", "")
        exp = self._token_cache.get("exp", 0)
        if cached and exp > time.time() + 60:
            return cached
        resp = requests.post(
            self._token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token", "")
        if not token:
            raise RuntimeError("Atlan auth response had no access_token")
        self._token_cache["access_token"] = token
        self._token_cache["exp"] = self._decode_jwt_exp(token) or (time.time() + 3600)
        return token

    def find_table_asset(self, catalog: str, schema: str, table: str) -> dict | None:
        """Look up a Databricks table/view asset in Atlan by catalog.schema.table."""
        token = self._get_token()
        url = f"{self.base_url}/api/meta/search/indexsearch"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "dsl": {
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"__typeName.keyword": list(_DATABRICKS_TYPE_NAMES)}},
                            {"term": {"name.keyword": table}},
                            {"term": {"schemaName.keyword": schema}},
                            {"term": {"databaseName.keyword": catalog}},
                        ]
                    }
                },
                "size": 5,
                "_source": True,
            },
            "attributes": [
                "name", "qualifiedName", "typeName", "ownerUsers", "databaseName", "schemaName",
                "certificateStatus", "description", "userDescription", "connectorName",
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        entities = data.get("entities") or data.get("hits", {}).get("hits") or []
        if not entities:
            return None

        ent = entities[0]
        attrs = ent.get("attributes") or ent.get("_source") or {}
        qualified_name = attrs.get("qualifiedName", "")
        catalog_attr = attrs.get("databaseName") or ""
        schema_attr = attrs.get("schemaName") or ""
        name_attr = attrs.get("name") or ""
        if not (catalog_attr and schema_attr and name_attr):
            parsed_catalog, parsed_schema, parsed_table = _parse_qualified_name(qualified_name)
            catalog_attr = catalog_attr or parsed_catalog
            schema_attr = schema_attr or parsed_schema
            name_attr = name_attr or parsed_table
        return {
            "guid": ent.get("guid") or ent.get("_id"),
            "name": name_attr,
            "type_name": ent.get("typeName") or attrs.get("typeName"),
            "qualified_name": qualified_name,
            "catalog": catalog_attr,
            "schema": schema_attr,
            "description": attrs.get("userDescription") or attrs.get("description") or "",
            "owners": attrs.get("ownerUsers") or [],
            "certificate_status": attrs.get("certificateStatus") or "",
        }

    def find_table_by_name(self, name: str, max_results: int = 5) -> list[dict]:
        """Look up Databricks table/view assets in Atlan by bare name only (no catalog/schema),
        for lineage questions that don't specify the full path (e.g. "lineage of ekpo").
        Tries the name as-given plus common casing variants since name.keyword is exact-match.
        """
        token = self._get_token()
        url = f"{self.base_url}/api/meta/search/indexsearch"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        variants = list({name, name.upper(), name.lower(), name.capitalize()})
        payload = {
            "dsl": {
                "query": {
                    "bool": {
                        "filter": [{"terms": {"__typeName.keyword": list(_DATABRICKS_TYPE_NAMES)}}],
                        "should": [{"term": {"name.keyword": variant}} for variant in variants],
                        "minimum_should_match": 1,
                    }
                },
                "size": max_results,
                "_source": True,
            },
            "attributes": [
                "name", "qualifiedName", "typeName", "ownerUsers", "databaseName", "schemaName",
                "certificateStatus", "description", "userDescription", "connectorName",
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        entities = data.get("entities") or data.get("hits", {}).get("hits") or []

        results = []
        for ent in entities:
            attrs = ent.get("attributes") or ent.get("_source") or {}
            qualified_name = attrs.get("qualifiedName", "")
            catalog_attr = attrs.get("databaseName") or ""
            schema_attr = attrs.get("schemaName") or ""
            name_attr = attrs.get("name") or ""
            if not (catalog_attr and schema_attr and name_attr):
                parsed_catalog, parsed_schema, parsed_table = _parse_qualified_name(qualified_name)
                catalog_attr = catalog_attr or parsed_catalog
                schema_attr = schema_attr or parsed_schema
                name_attr = name_attr or parsed_table
            if not (catalog_attr and schema_attr and name_attr):
                continue
            results.append({
                "guid": ent.get("guid") or ent.get("_id"),
                "name": name_attr,
                "catalog": catalog_attr,
                "schema": schema_attr,
                "qualified_name": qualified_name,
            })
        return results

    def find_upstream_sources(self, guid: str) -> list[dict]:
        """Walk Atlan's lineage graph upstream (direction=INPUT) from `guid`,
        returning every Databricks Table/View source asset found (deduplicated).
        """
        return self._walk_lineage(guid, direction="INPUT")

    def find_downstream_targets(self, guid: str) -> list[dict]:
        """Walk Atlan's lineage graph downstream (direction=OUTPUT) from `guid`,
        returning every Databricks Table/View asset that consumes it (deduplicated).
        """
        return self._walk_lineage(guid, direction="OUTPUT")

    def _walk_lineage(self, guid: str, direction: str) -> list[dict]:
        if not guid:
            return []
        token = self._get_token()
        url = f"{self.base_url}/api/meta/lineage/{guid}"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        params = {"direction": direction, "hideProcess": "true"}
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        guid_entity_map = data.get("guidEntityMap") or {}
        found = []
        for entity_guid, ent in guid_entity_map.items():
            if entity_guid == guid:
                continue
            type_name = ent.get("typeName", "")
            if type_name not in _DATABRICKS_TYPE_NAMES:
                continue
            attrs = ent.get("attributes") or {}
            # The lineage endpoint returns entities with a sparse/empty `attributes` dict
            # regardless of any `attributes` query params -- the reliable way to get
            # qualifiedName/name/databaseName/schemaName is the same per-entity GET used
            # for AI Application discovery.
            full_attrs = (self._fetch_entity_by_guid(entity_guid, token).get("attributes") or {})
            attrs = {**attrs, **{k: v for k, v in full_attrs.items() if v}}
            connector_name = (attrs.get("connectorName") or "").strip().lower()
            qualified_name = attrs.get("qualifiedName", "")
            if connector_name:
                if connector_name != "databricks":
                    continue
            elif qualified_name and not qualified_name.lower().startswith("default/databricks/"):
                continue
            catalog_attr = attrs.get("databaseName") or ""
            schema_attr = attrs.get("schemaName") or ""
            name_attr = attrs.get("name") or ""
            if not (catalog_attr and schema_attr and name_attr):
                parsed_catalog, parsed_schema, parsed_table = _parse_qualified_name(qualified_name)
                catalog_attr = catalog_attr or parsed_catalog
                schema_attr = schema_attr or parsed_schema
                name_attr = name_attr or parsed_table
            found.append({
                "type_name": type_name,
                "catalog": catalog_attr,
                "schema": schema_attr,
                "table": name_attr,
            })
        return found

    def _get_ai_app_cm_maps(self, token: str) -> tuple[str | None, dict[str, str]]:
        """Resolve the "AI Apps Metadata" custom-metadata set: returns
        (internal_set_name, {field_internal_id: field_display_name}). Cached for the
        client's lifetime — businessAttributes on entities are keyed by these internal
        UUIDs, not display names, so this mapping is required to read them.
        """
        if self._cm_set_name_cache is not None:
            return self._cm_set_name_cache, self._cm_field_map_cache or {}
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        resp = requests.get(
            f"{self.base_url}/api/meta/types/typedefs",
            headers=headers,
            params={"type": "business_metadata"},
            timeout=15,
        )
        resp.raise_for_status()
        for cm_def in resp.json().get("businessMetadataDefs", []):
            if cm_def.get("displayName") == _AI_APP_CM_SET:
                self._cm_set_name_cache = cm_def.get("name")
                self._cm_field_map_cache = {
                    a.get("name"): a.get("displayName") for a in cm_def.get("attributeDefs", [])
                }
                return self._cm_set_name_cache, self._cm_field_map_cache
        self._cm_set_name_cache = ""  # mark resolved-but-not-found so we don't refetch every call
        self._cm_field_map_cache = {}
        return None, {}

    def _get_tag_name_map(self, token: str) -> dict[str, str]:
        """Resolve classification (tag) internal ids to display names. Cached for the
        client's lifetime — entities' classificationNames are internal UUIDs, not the
        human-readable tag names shown in Atlan's UI.
        """
        if self._tag_name_cache is not None:
            return self._tag_name_cache
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        resp = requests.get(
            f"{self.base_url}/api/meta/types/typedefs",
            headers=headers,
            params={"type": "classification"},
            timeout=15,
        )
        resp.raise_for_status()
        self._tag_name_cache = {
            c.get("name"): c.get("displayName") for c in resp.json().get("classificationDefs", [])
        }
        return self._tag_name_cache

    def _fetch_entity_by_guid(self, guid: str, token: str) -> dict:
        """GET the full entity (businessAttributes + relationshipAttributes included) —
        Atlan's search/indexsearch endpoint never returns these regardless of what's
        requested in the payload, so a follow-up per-entity fetch is required.
        """
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        resp = requests.get(f"{self.base_url}/api/meta/entity/guid/{guid}", headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json().get("entity", {})

    def _resolve_domain_names(self, entity: dict, token: str) -> list[str]:
        names = _extract_domain_names(entity)
        for guid in (entity.get("attributes") or {}).get("domainGUIDs") or []:
            if not isinstance(guid, str) or not guid:
                continue
            if guid not in self._domain_name_cache:
                try:
                    domain = self._fetch_entity_by_guid(guid, token)
                    attrs = domain.get("attributes") or {}
                    name = attrs.get("name") or domain.get("displayText") or ""
                    self._domain_name_cache[guid] = name.strip() if isinstance(name, str) else ""
                except requests.RequestException as exc:
                    logger.warning("Could not resolve Atlan domain %s: %s", guid, exc)
                    self._domain_name_cache[guid] = ""
            name = self._domain_name_cache[guid]
            if name and name not in names:
                names.append(name)
        return names

    def discover_ai_applications(self, page_size: int = 100, max_pages: int = 10) -> list[dict]:
        """Query Atlan for all active AIApplication assets (the registry source of truth),
        paginating until exhausted or `max_pages` is reached. Each entry:
        {"name", "qualified_name", "description", "certificate", "connector", "space_id",
        "deployment_environment", "workspace_name", "agent_owner", "tags", "readme"}.
        `space_id` is read from the "AI Apps Metadata" custom metadata set's "Agent ID"
        field first, falling back to a 32-char hex scan of the qualifiedName/description/readme.
        Entries with no resolvable space_id are still returned (space_id will be None) so
        callers can report how many were skipped.

        Note: businessAttributes/relationshipAttributes are NOT returned by indexsearch on
        this tenant (verified live) — each matching asset requires a follow-up GET
        /api/meta/entity/guid/{guid} call to fetch them, resolved via typedef lookups since
        Atlan keys them by internal UUID rather than display name.
        """
        token = self._get_token()
        cm_set_name, cm_field_map = self._get_ai_app_cm_maps(token)
        tag_name_map = self._get_tag_name_map(token)
        url = f"{self.base_url}/api/meta/search/indexsearch"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        apps: list[dict] = []
        for page in range(max_pages):
            payload = {
                "dsl": {
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"__typeName.keyword": "AIApplication"}},
                                {"term": {"__state": "ACTIVE"}},
                            ]
                        }
                    },
                    "from": page * page_size,
                    "size": page_size,
                },
                "attributes": [
                    "name", "qualifiedName", "description", "userDescription",
                    "certificateStatus", "connectorName",
                ],
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            entities = resp.json().get("entities", [])
            if not entities:
                break
            for ent in entities:
                attrs = ent.get("attributes") or {}
                name = (attrs.get("name") or "").strip()
                guid = ent.get("guid")
                if not name or not guid:
                    continue
                qualified_name = attrs.get("qualifiedName") or ""
                description = (attrs.get("userDescription") or attrs.get("description") or "").strip()

                full_entity = self._fetch_entity_by_guid(guid, token)
                biz = full_entity.get("businessAttributes") or {}
                cm_raw = biz.get(cm_set_name) if cm_set_name else None
                cm_fields = {
                    cm_field_map.get(field_id, field_id): value
                    for field_id, value in (cm_raw or {}).items()
                }
                readme_text = _extract_readme_text(full_entity)

                space_id = (
                    _extract_space_id_from_biz(cm_fields, qualified_name)
                    or _extract_space_id_from_biz({}, description)
                    or _extract_space_id_from_biz({}, readme_text)
                )
                tag_ids = full_entity.get("classificationNames") or ent.get("classificationNames") or []
                tags = [tag_name_map.get(tag_id, tag_id) for tag_id in tag_ids]

                apps.append({
                    "name": name,
                    "qualified_name": qualified_name,
                    "description": description,
                    "certificate": attrs.get("certificateStatus") or "",
                    "connector": attrs.get("connectorName") or "",
                    "space_id": space_id,
                    "domains": self._resolve_domain_names(full_entity, token),
                    "tags": tags,
                    "readme": readme_text,
                    **_extract_app_metadata_cm(cm_fields),
                })
            if len(entities) < page_size:
                break
        return apps

    def get_table_context(self, catalog: str, schema: str, table: str) -> dict:
        """Return a structured summary of a table's Atlan context + upstream/downstream
        lineage: {"found": bool, "asset": dict|None, "upstream": list[dict],
        "downstream": list[dict], "error": str|None}.
        """
        try:
            asset = self.find_table_asset(catalog, schema, table)
            if not asset:
                return {"found": False, "asset": None, "upstream": [], "downstream": [], "error": None}
            upstream = self.find_upstream_sources(asset["guid"])
            downstream = self.find_downstream_targets(asset["guid"])
            return {"found": True, "asset": asset, "upstream": upstream, "downstream": downstream, "error": None}
        except Exception as exc:
            logger.warning("Atlan lookup failed for %s.%s.%s", catalog, schema, table, exc_info=exc)
            return {"found": False, "asset": None, "upstream": [], "downstream": [], "error": str(exc)}

    def find_glossary_terms(self, name: str, max_results: int = 5) -> list[dict]:
        """Look up AtlasGlossaryTerm assets by (partial, case-insensitive) name match."""
        token = self._get_token()
        url = f"{self.base_url}/api/meta/search/indexsearch"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "dsl": {
                "query": {
                    "bool": {
                        "filter": [{"term": {"__typeName.keyword": "AtlasGlossaryTerm"}}],
                        "must": [{"match": {"name": name}}],
                    }
                },
                "size": max_results,
                "_source": True,
            },
            "attributes": [
                "name", "qualifiedName", "shortDescription", "longDescription", "userDescription",
                "certificateStatus", "ownerUsers", "anchor",
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        entities = data.get("entities") or data.get("hits", {}).get("hits") or []
        results = []
        for ent in entities:
            attrs = ent.get("attributes") or ent.get("_source") or {}
            results.append({
                "guid": ent.get("guid") or ent.get("_id"),
                "name": attrs.get("name") or "",
                "qualified_name": attrs.get("qualifiedName") or "",
                "description": attrs.get("userDescription") or attrs.get("longDescription") or attrs.get("shortDescription") or "",
                "certificate_status": attrs.get("certificateStatus") or "",
                "owners": attrs.get("ownerUsers") or [],
            })
        return results

    def find_data_products(self, name: str, max_results: int = 5) -> list[dict]:
        """Look up DataProduct assets in Atlan by (partial, case-insensitive) name match."""
        token = self._get_token()
        url = f"{self.base_url}/api/meta/search/indexsearch"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "dsl": {
                "query": {
                    "bool": {
                        "filter": [{"term": {"__typeName.keyword": "DataProduct"}}],
                        "must": [{"match": {"name": name}}],
                    }
                },
                "size": max_results,
                "_source": True,
            },
            "attributes": [
                "name", "qualifiedName", "description", "userDescription",
                "certificateStatus", "ownerUsers", "daProductStatus",
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        entities = data.get("entities") or data.get("hits", {}).get("hits") or []
        results = []
        for ent in entities:
            attrs = ent.get("attributes") or ent.get("_source") or {}
            results.append({
                "guid": ent.get("guid") or ent.get("_id"),
                "name": attrs.get("name") or "",
                "qualified_name": attrs.get("qualifiedName") or "",
                "description": attrs.get("userDescription") or attrs.get("description") or "",
                "certificate_status": attrs.get("certificateStatus") or "",
                "owners": attrs.get("ownerUsers") or [],
                "status": attrs.get("daProductStatus") or "",
            })
        return results

    def discover_data_products(self, page_size: int = 100, max_pages: int = 10) -> list[dict]:
        """Query Atlan for all active DataProduct assets, for "what data products are available"
        catalog-style questions. Same shape as find_data_products() entries."""
        token = self._get_token()
        url = f"{self.base_url}/api/meta/search/indexsearch"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        products: list[dict] = []
        for page in range(max_pages):
            payload = {
                "dsl": {
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"__typeName.keyword": "DataProduct"}},
                                {"term": {"__state": "ACTIVE"}},
                            ]
                        }
                    },
                    "from": page * page_size,
                    "size": page_size,
                },
                "attributes": [
                    "name", "qualifiedName", "description", "userDescription",
                    "certificateStatus", "ownerUsers", "daProductStatus",
                ],
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            entities = resp.json().get("entities", [])
            if not entities:
                break
            for ent in entities:
                attrs = ent.get("attributes") or {}
                name = (attrs.get("name") or "").strip()
                if not name:
                    continue
                products.append({
                    "guid": ent.get("guid"),
                    "name": name,
                    "qualified_name": attrs.get("qualifiedName") or "",
                    "description": (attrs.get("userDescription") or attrs.get("description") or "").strip(),
                    "certificate_status": attrs.get("certificateStatus") or "",
                    "owners": attrs.get("ownerUsers") or [],
                    "status": attrs.get("daProductStatus") or "",
                })
            if len(entities) < page_size:
                break
        return products


def format_context_markdown(fqtn: str, context: dict) -> str:
    """Render a get_table_context() result as a short markdown block."""
    if context.get("error"):
        return f"**{fqtn}** — Atlan lookup failed: {context['error']}"
    if not context.get("found"):
        return f"**{fqtn}** — not found in Atlan's catalog."

    asset = context["asset"]
    lines = [f"**{fqtn}**"]
    if asset.get("description"):
        lines.append(f"- Description: {asset['description']}")
    if asset.get("owners"):
        lines.append(f"- Owners: {', '.join(asset['owners'])}")
    if asset.get("certificate_status"):
        lines.append(f"- Certification: {asset['certificate_status']}")

    upstream = context.get("upstream") or []
    named_upstream = [u for u in upstream if u.get("catalog") and u.get("schema") and u.get("table")]
    if named_upstream:
        sources = ", ".join(f"{u['catalog']}.{u['schema']}.{u['table']}" for u in named_upstream)
        lines.append(f"- Built from (upstream sources): {sources}")
    elif upstream:
        lines.append(f"- Built from {len(upstream)} upstream source(s), but Atlan didn't return their names.")
    else:
        lines.append("- No upstream Databricks sources found in lineage (likely a root/source table).")

    downstream = context.get("downstream") or []
    named_downstream = [d for d in downstream if d.get("catalog") and d.get("schema") and d.get("table")]
    if named_downstream:
        targets = ", ".join(f"{d['catalog']}.{d['schema']}.{d['table']}" for d in named_downstream)
        lines.append(f"- Feeds into (downstream consumers): {targets}")
    elif downstream:
        lines.append(f"- Feeds {len(downstream)} downstream consumer(s), but Atlan didn't return their names.")
    else:
        lines.append("- No downstream Databricks consumers found in lineage (likely a terminal/reporting table).")
    return "\n".join(lines)


def format_glossary_markdown(term: str, matches: list[dict]) -> str:
    """Render find_glossary_terms() results as a short markdown block."""
    if not matches:
        return f"**{term}** — no matching glossary term found in Atlan."
    blocks = []
    for match in matches:
        lines = [f"**{match['name'] or term}**"]
        if match.get("description"):
            lines.append(f"- Definition: {match['description']}")
        if match.get("owners"):
            lines.append(f"- Owners: {', '.join(match['owners'])}")
        if match.get("certificate_status"):
            lines.append(f"- Certification: {match['certificate_status']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_data_product_markdown(name: str, matches: list[dict]) -> str:
    """Render find_data_products()/discover_data_products() results as a short markdown block."""
    if not matches:
        return f"**{name}** — no matching data product found in Atlan."
    blocks = []
    for match in matches:
        lines = [f"**{match['name'] or name}**"]
        if match.get("description"):
            lines.append(f"- Description: {match['description']}")
        if match.get("owners"):
            lines.append(f"- Owners: {', '.join(match['owners'])}")
        if match.get("status"):
            lines.append(f"- Status: {match['status']}")
        if match.get("certificate_status"):
            lines.append(f"- Certification: {match['certificate_status']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

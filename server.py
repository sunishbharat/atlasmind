"""
server.py — FastAPI server for AtlasMind.

Endpoints:
    GET  /query?q=<natural language>&profile=<name>&limit=<n>
    POST /query  { "query": "...", "profile": "...", "limit": N }

The SentenceTransformer model and pgvector annotations are loaded once at
startup and reused across all requests.

Run:
    uv run uvicorn server:app --reload
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

sys.path.insert(0, "src")

from atlasmind import atlasmind, atlasmind_multi_project, normalize_issue
from config import get_profile
from config_fields import FIELD_META
from main import (
    _detect_fields,
    _detect_post_filters,
    _fields_from_order_by,
    _parse_limit,
    _parse_per_project_limits,
    _sanitize_jql,
    generate_jql,
    load_annotations_into_db,
)
from models import ChartSpec, QueryRequest, QueryResponse
from settings import DEFAULT_ANNOTATION_FILE, MAX_RESULTS

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# -- Startup state (shared across requests) ---------------------------
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading JQL annotations and embedding model...")
    _state["model"] = load_annotations_into_db(DEFAULT_ANNOTATION_FILE)
    logger.info("Model ready.")
    yield
    _state.clear()


app = FastAPI(title="AtlasMind", lifespan=lifespan)


# -- Helpers ----------------------------------------------------------

def _extract_filters(issues: list[dict]) -> dict[str, list[str]]:
    """Build filter facets from normalised issues for frontend filter dropdowns.

    Groups unique non-null values for each filterable field across all issues.

    Args:
        issues: List of normalised issue dicts (output of normalize_issue()).

    Returns:
        Dict mapping field name to sorted list of unique values.
    """
    facet_fields = ["status", "issuetype", "priority", "assignee", "sprint", "labels"]
    facets: dict[str, set] = {f: set() for f in facet_fields}
    for issue in issues:
        for field in facet_fields:
            value = issue.get(field)
            if value is None:
                continue
            if isinstance(value, list):
                facets[field].update(v for v in value if v)
            else:
                facets[field].add(value)
    return {k: sorted(v) for k, v in facets.items() if v}


def _build_response(
    profile,
    response_type: str,
    llm_result: dict,
    jira_result: dict | None = None,
) -> QueryResponse:
    """Build a uniform QueryResponse for both general and JQL query results.

    Args:
        profile:       Active Profile object.
        response_type: "general" or "jql".
        llm_result:    Dict returned by generate_jql() with keys jql, chart_spec, answer.
        jira_result:   Dict returned by atlasmind(); required when response_type is "jql".

    Returns:
        QueryResponse with a consistent shape regardless of response_type.
    """
    chart_spec = None
    if llm_result.get("chart_spec"):
        try:
            chart_spec = ChartSpec(**llm_result["chart_spec"])
        except Exception:
            pass

    if response_type == "general":
        return QueryResponse(
            type=response_type,
            profile=profile.name,
            jira_base_url=profile.jira_base_url,
            answer=llm_result.get("answer"),
            chart_spec=chart_spec,
        )

    normalised = [normalize_issue(r) for r in jira_result.get("raw_issues", [])]
    return QueryResponse(
        type=response_type,
        profile=profile.name,
        jira_base_url=profile.jira_base_url,
        answer=llm_result.get("answer"),
        jql=jira_result.get("jql"),
        total=jira_result.get("total", 0),
        shown=jira_result.get("shown", 0),
        examined=jira_result.get("examined", 0),
        post_filters=jira_result.get("post_filters", []),
        display_fields=jira_result.get("display_fields", []),
        issues=normalised,
        chart_spec=chart_spec,
        filters=_extract_filters(normalised),
    )


# -- Shared query execution -------------------------------------------

async def _execute_query(q: str, profile_name: str | None, limit: int | None, model) -> QueryResponse:
    """Core pipeline: NL query -> LLM -> Jira -> QueryResponse."""
    try:
        active_profile = get_profile(profile_name)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    llm_result = await generate_jql(q, model)

    if not llm_result["jql"]:
        return _build_response(active_profile, "general", llm_result=llm_result)

    jql, jql_limit = _sanitize_jql(llm_result["jql"])
    logger.info("Generated JQL : %s", jql)

    max_results  = limit or jql_limit or _parse_limit(q) or MAX_RESULTS
    fields       = _detect_fields(q)
    post_filters = _detect_post_filters(q)

    for order_field in _fields_from_order_by(jql):
        if order_field not in fields and order_field in FIELD_META:
            fields.append(order_field)

    for pf in post_filters:
        if pf.field not in fields and pf.field in FIELD_META:
            fields.insert(0, pf.field)
    fields = fields[:9]

    per_project = _parse_per_project_limits(q)
    if per_project:
        result = await atlasmind_multi_project(
            active_profile,
            jql_query=jql,
            per_project_limits=per_project,
            fields=fields,
            post_filters=post_filters,
        )
    else:
        result = await atlasmind(
            active_profile,
            jql_query=jql,
            max_results=max_results,
            fields=fields,
            post_filters=post_filters,
        )

    if result is None:
        raise HTTPException(status_code=400, detail=f"JQL query failed: {jql}")

    return _build_response(active_profile, "jql", llm_result=llm_result, jira_result=result)


# -- Query endpoints --------------------------------------------------

@app.get("/query", response_model=QueryResponse)
async def query_get(
    q:       str       = Query(..., description="Natural language Jira query"),
    profile: str | None = Query(None, description="Profile name from profiles.json"),
    limit:   int | None = Query(None, description="Max results (overrides query hint)"),
):
    """Convert a natural language query to JQL and return matching Jira issues as JSON."""
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialised.")
    return await _execute_query(q, profile, limit, model)


@app.post("/query", response_model=QueryResponse)
async def query_post(request: QueryRequest):
    """Convert a natural language query to JQL and return matching Jira issues as JSON."""
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialised.")
    return await _execute_query(request.query, request.profile, request.limit, model)

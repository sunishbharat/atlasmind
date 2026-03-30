"""
server.py — FastAPI server for AtlasMind.

Exposes a single endpoint:
    GET /query?q=<natural language>&profile=<name>&limit=<n>

The SentenceTransformer model and pgvector annotations are loaded once at
startup and reused across all requests.  atlasmind() returns a structured
dict which FastAPI serialises directly to JSON.

Run:
    uv run uvicorn server:app --reload
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

sys.path.insert(0, "src")

from atlasmind import atlasmind, atlasmind_multi_project
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
from settings import DEFAULT_ANNOTATION_FILE, MAX_RESULTS

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# ── Startup state (shared across requests) ───────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading JQL annotations and embedding model...")
    _state["model"] = load_annotations_into_db(DEFAULT_ANNOTATION_FILE)
    logger.info("Model ready.")
    yield
    _state.clear()


app = FastAPI(title="AtlasMind", lifespan=lifespan)


# ── Query endpoint ───────────────────────────────────────────────────

@app.get("/query")
async def query(
    q:       str       = Query(..., description="Natural language Jira query"),
    profile: str | None = Query(None, description="Profile name from profiles.json"),
    limit:   int | None = Query(None, description="Max results (overrides query hint)"),
):
    """Convert a natural language query to JQL and return matching Jira issues as JSON."""
    try:
        active_profile = get_profile(profile)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialised.")

    jql = await generate_jql(q, model)
    jql, jql_limit = _sanitize_jql(jql)
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

    return result

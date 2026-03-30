import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

import httpx
from sentence_transformers import SentenceTransformer

from atlasmind import atlasmind, MAX_RESULTS
from config import get_profile, print_profiles
from config_fields import (
    DEFAULT_DISPLAY_FIELDS, FIELD_META, QUERY_FIELD_MAP,
    POST_FILTER_PATTERNS, POST_FILTER_FETCH_MULTIPLIER,
)
from settings import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE, DEFAULT_ANNOTATION_FILE

sys.path.insert(0, str(Path(__file__).parent / "src"))
from document_processor_test import update_pgvector_from_annotations, test_embeddings_jql
from jql_annotation_parser import parse_jql_annotations

logger = logging.getLogger(__name__)


class PostFilter(NamedTuple):
    field:     str
    operator:  str
    threshold: int


# ── DB bootstrap ─────────────────────────────────────────────────────

def load_annotations_into_db(annotation_file: str) -> SentenceTransformer:
    """Parse annotation file and load (comment, jql, embedding) rows into pgvector."""
    path = Path(annotation_file)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    pairs = parse_jql_annotations(str(path))
    logger.info("Loaded %d annotation pairs from %s", len(pairs), path.name)
    model = update_pgvector_from_annotations(pairs)
    return model


# ── JQL generation ───────────────────────────────────────────────────

def _build_prompt(query: str, examples: list[tuple]) -> str:
    example_block = "\n\n".join(
        f"-- {annotation}\n{jql}"
        for _, annotation, jql, _ in examples
    )
    return (
        f"You are a Jira JQL expert. Use the examples below as references. "
        f"Each example is in the format '-- <annotation>\\n<jql>'. "
        f"Generate a single valid JQL statement for the user's request using these rules:\n"
        f"- Do not use placeholder values like 'ProjectName' or 'USERNAME'.\n"
        f"- If no specific project is mentioned, omit the project filter.\n"
        f"- Do NOT use date arithmetic between two fields. JQL does not support this.\n"
        f"  INVALID: resolutiondate >= created + 20d\n"
        f"  INVALID: resolutiondate - created > 20d\n"
        f"  CORRECT: project = ZOOKEEPER AND resolution IS NOT EMPTY ORDER BY resolutiondate DESC\n"
        f"  (duration filtering like 'took more than N days' is handled separately — omit it from JQL)\n"
        f"- Do NOT append LIMIT — result count is controlled externally.\n"
        f"- Output only the JQL — no explanation, no markdown, no comments.\n\n"
        f"Examples:\n{example_block}\n\n"
        f"User request: {query}\n"
        f"JQL:"
    )


async def generate_jql(query: str, model: SentenceTransformer) -> str:
    """Retrieve top-5 similar examples from pgvector then ask Ollama to generate JQL."""
    examples, _ = test_embeddings_jql(query, model)
    if not examples:
        raise RuntimeError("No examples found in pgvector — was the DB loaded?")

    prompt = _build_prompt(query, examples)
    logger.debug("Ollama prompt:\n%s", prompt)

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": OLLAMA_TEMPERATURE}},
        )
        response.raise_for_status()
        jql = response.json()["response"].strip()

    # Strip any accidental markdown fences
    jql = re.sub(r"^```[a-z]*\n?", "", jql, flags=re.IGNORECASE).strip("`").strip()
    return jql


def _remove_field_arithmetic(jql: str) -> str:
    """Remove unsupported field-to-field date arithmetic from JQL.

    JQL does not support arithmetic between two fields. The LLM produces two forms:
      - resolutiondate >= created + 20d
      - resolutiondate - created > 20d
    Both are stripped so the query remains valid. Duration filtering is handled
    by the days_to_fix post-filter in Python.
    """
    # Form A: field >= otherfield + Nd  /  field <= otherfield - Nd
    jql = re.sub(
        r"(?:AND\s+|OR\s+)?\w+\s*[><=!]+\s*\w+\s*[+\-]\s*\d+[smhdwMy]\b",
        "", jql, flags=re.IGNORECASE,
    )
    # Form B: field - otherfield > Nd  /  field + otherfield < Nd
    jql = re.sub(
        r"(?:AND\s+|OR\s+)?\w+\s*[+\-]\s*\w+\s*[><=!]+\s*\d+[smhdwMy]\b",
        "", jql, flags=re.IGNORECASE,
    )
    # Clean up empty parentheses left after clause removal e.g. "AND ()"
    jql = re.sub(r"(?:AND\s+|OR\s+)?\(\s*\)", "", jql, flags=re.IGNORECASE)
    # Clean up orphaned AND / OR at start or end
    jql = re.sub(r"^\s*(?:AND|OR)\s+", "", jql, flags=re.IGNORECASE)
    jql = re.sub(r"\s+(?:AND|OR)\s*$", "", jql, flags=re.IGNORECASE)
    return jql.strip()


def _sanitize_jql(jql: str) -> tuple[str, int | None]:
    """Clean LLM-generated JQL and extract maxResults limit.

    Applies _remove_field_arithmetic then strips any SQL-style LIMIT clause,
    returning it as the maxResults value for the API call.
    """
    jql = _remove_field_arithmetic(jql)
    m = re.search(r"\s+LIMIT\s+(\d+)\s*$", jql, flags=re.IGNORECASE)
    if m:
        return jql[:m.start()].strip(), int(m.group(1))
    return jql, None


# ── Query helpers ────────────────────────────────────────────────────

def _detect_fields(query: str) -> list[str]:
    seen: list[str] = []
    for pattern, field in QUERY_FIELD_MAP:
        if re.search(pattern, query, re.IGNORECASE) and field not in seen:
            if field in FIELD_META:
                seen.append(field)
    if not seen:
        return list(DEFAULT_DISPLAY_FIELDS)
    for default in DEFAULT_DISPLAY_FIELDS:
        if default not in seen:
            seen.append(default)
    return seen[:9]


def _parse_limit(query: str) -> int:
    # Explicit record-count patterns only — avoid grabbing numbers that belong
    # to filter conditions like "more than 20 days" or "last 7 days".
    m = re.search(r'\b(\d+)\s+issue', query, re.IGNORECASE)
    if not m:
        m = re.search(r'\b(?:top|first|last|show|list|get|fetch)\s+(\d+)\b', query, re.IGNORECASE)
    return int(m.group(1)) if m else MAX_RESULTS


def _detect_post_filters(query: str) -> list[PostFilter]:
    filters: list[PostFilter] = []
    seen: set[PostFilter] = set()
    for pattern, field_name, operator, unit_map in POST_FILTER_PATTERNS:
        m = pattern.search(query)
        if m:
            quantity  = int(m.group(1))
            unit_stem = m.group(2).lower().rstrip("s")
            threshold = quantity * unit_map.get(unit_stem, 1)
            pf = PostFilter(field=field_name, operator=operator, threshold=threshold)
            if pf not in seen:
                filters.append(pf)
                seen.add(pf)
    return filters


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog="atlasmind",
        description="AtlasMind — Query Jira using natural language via local Ollama LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", default="list all open issues", metavar="TEXT",
        help="Natural language question to convert to JQL and run against Jira.",
    )
    parser.add_argument(
        "--profile", default=None, metavar="NAME",
        help="Atlassian profile to use (from profiles.json). Run --list-profiles to see options.",
    )
    parser.add_argument(
        "--list-profiles", action="store_true",
        help="List all configured profiles and exit.",
    )
    parser.add_argument(
        "--annotation-file", default=DEFAULT_ANNOTATION_FILE, metavar="PATH",
        help=(
            "Path to JQL annotation file used to seed the pgvector DB. "
            f"(default: {DEFAULT_ANNOTATION_FILE})"
        ),
    )
    parser.add_argument(
        "--reload-db", action="store_true",
        help="Force reload of the pgvector DB from the annotation file.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8",
    )

    args = parse_args()

    if args.list_profiles:
        print_profiles()
        return

    profile = get_profile(args.profile)
    logger.info("Profile : %s - %s", profile.name, profile.jira_base_url)

    # ── 1. Load annotation embeddings into pgvector ──────────────────
    logger.info("Loading JQL annotations into pgvector from: %s", args.annotation_file)
    model = load_annotations_into_db(args.annotation_file)

    # ── 2. Generate JQL via similarity search + Ollama ───────────────
    logger.info("Query   : %s", args.query)
    jql = asyncio.run(generate_jql(args.query, model))
    jql, jql_limit = _sanitize_jql(jql)
    logger.info("Generated JQL : %s", jql)

    # ── 3. Run JQL against Jira ──────────────────────────────────────
    # jql_limit (from LLM LIMIT clause) takes precedence over NL query limit
    limit        = jql_limit or _parse_limit(args.query)
    fields       = _detect_fields(args.query)
    post_filters = _detect_post_filters(args.query)

    for pf in post_filters:
        if pf.field not in fields and pf.field in FIELD_META:
            fields.insert(0, pf.field)
    fields = fields[:9]

    asyncio.run(atlasmind(
        profile,
        jql_query=jql,
        max_results=limit,
        fields=fields,
        post_filters=post_filters,
    ))


if __name__ == "__main__":
    main()

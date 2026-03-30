"""
main.py — CLI entry point for AtlasMind.

Pipeline:
  1. Load JQL annotation file → generate embeddings → store in pgvector DB.
  2. Accept a natural language query from the user.
  3. Similarity search pgvector for the top-5 most relevant (annotation, JQL) examples.
  4. Build a few-shot prompt and call the local Ollama LLM to generate JQL.
  5. Sanitize the generated JQL (strip invalid arithmetic, LIMIT clauses).
  6. Execute the JQL against Jira via atlasmind() and print a formatted results table.

Configuration is read from settings.py; field/filter metadata from config_fields.py.
Run with --help for full CLI usage.
"""

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
    """A Python-side filter applied after the Jira API fetch.

    Used for constraints that JQL cannot express (e.g. computed fields like
    days_to_fix). Evaluated by _apply_post_filters() in atlasmind.py.

    Attributes:
        field: FIELD_META key whose extractor produces a numeric string.
        operator: Comparison operator — one of ">", ">=", "<", "<=".
        threshold: Numeric threshold value in canonical units (days).
    """
    field:     str
    operator:  str
    threshold: int


# ── DB bootstrap ─────────────────────────────────────────────────────

def load_annotations_into_db(annotation_file: str) -> SentenceTransformer:
    """Parse a JQL annotation file and load embeddings into the pgvector database.

    Args:
        annotation_file: Path to the annotation file (.md format with comment/JQL pairs).

    Returns:
        SentenceTransformer: The embedding model used, for reuse in similarity search.

    Raises:
        FileNotFoundError: If the annotation file does not exist at the given path.
    """
    path = Path(annotation_file)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    pairs = parse_jql_annotations(str(path))
    logger.info("Loaded %d annotation pairs from %s", len(pairs), path.name)
    model = update_pgvector_from_annotations(pairs)
    return model


# ── JQL generation ───────────────────────────────────────────────────

def _build_prompt(query: str, examples: list[tuple]) -> str:
    """Build a few-shot prompt for the Ollama LLM from pgvector similarity results.

    Formats the top-N retrieved (annotation, jql) examples as reference context
    and appends the user's natural language query. Includes explicit rules to
    prevent common LLM mistakes such as field arithmetic and SQL LIMIT clauses.

    Args:
        query: The user's natural language query string.
        examples: List of rows from test_embeddings_jql() — each is
                  (id, annotation, jql, distance).

    Returns:
        str: The fully formatted prompt string ready to send to Ollama.
    """
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
    """Generate a JQL string from a natural language query using RAG + Ollama.

    Retrieves the top-5 most semantically similar (annotation, JQL) pairs from
    pgvector, builds a few-shot prompt, and sends it to the local Ollama LLM.
    Strips markdown fences from the response before returning.

    Args:
        query: The user's natural language query string.
        model: SentenceTransformer model used to encode the query for similarity search.

    Returns:
        str: Raw JQL string as generated by the LLM (before sanitization).

    Raises:
        RuntimeError: If pgvector returns no examples (DB not loaded).
        httpx.HTTPStatusError: If the Ollama API request fails.
    """
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
    """Infer Jira display fields from keywords in the natural language query.

    Scans the query against QUERY_FIELD_MAP regex patterns and returns matching
    FIELD_META keys. Falls back to DEFAULT_DISPLAY_FIELDS when no keywords match.
    Always appends any missing DEFAULT_DISPLAY_FIELDS at the end, capped at 9.

    Args:
        query: The user's natural language query string.

    Returns:
        list[str]: Ordered list of FIELD_META keys to display as table columns.
    """
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
    """Extract an explicit record count from the natural language query.

    Matches patterns like "show 20 issues" or "top 5 bugs". Deliberately avoids
    bare number matching to prevent filter values (e.g. "20 days", "last 7 days")
    from being misinterpreted as record limits.

    Args:
        query: The user's natural language query string.

    Returns:
        int: The extracted record limit, or MAX_RESULTS if no count is found.
    """
    # Explicit record-count patterns only — avoid grabbing numbers that belong
    # to filter conditions like "more than 20 days" or "last 7 days".
    m = re.search(r'\b(\d+)\s+issue', query, re.IGNORECASE)
    if not m:
        m = re.search(r'\b(?:top|first|last|show|list|get|fetch)\s+(\d+)\b', query, re.IGNORECASE)
    return int(m.group(1)) if m else MAX_RESULTS


def _detect_post_filters(query: str) -> list[PostFilter]:
    """Detect time-based constraints in the query that JQL cannot express.

    Scans the query against POST_FILTER_PATTERNS and converts matches into
    PostFilter objects with a canonical threshold in days. Deduplicates
    identical filters that might match multiple patterns.

    Args:
        query: The user's natural language query string.

    Returns:
        list[PostFilter]: Filters to apply in Python after the Jira API fetch.
    """
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
    """Define and parse CLI arguments for the AtlasMind entry point.

    Returns:
        argparse.Namespace: Parsed arguments including query, profile,
        annotation_file, reload_db, and list_profiles flags.
    """
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
    """Orchestrate the full AtlasMind pipeline from CLI args to Jira table output.

    1. Parse CLI arguments and resolve the active Jira profile.
    2. Load JQL annotation pairs into pgvector (generates embeddings via SentenceTransformer).
    3. Convert the natural language query to JQL via RAG + Ollama.
    4. Sanitize the JQL (strip field arithmetic and LIMIT clauses).
    5. Detect display fields and Python-side post-filters from the NL query.
    6. Execute the JQL against Jira and print a formatted results table.
    """
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

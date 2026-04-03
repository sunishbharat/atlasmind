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
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

import psycopg2

import httpx
from sentence_transformers import SentenceTransformer

from atlasmind import atlasmind, atlasmind_multi_project, MAX_RESULTS, check_rovo_mcp
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


# -- DB bootstrap -----------------------------------------------------

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
    try:
        model = update_pgvector_from_annotations(pairs)
    except psycopg2.OperationalError as exc:
        logger.error("Failed to connect to pgvector database: %s", exc)
        logger.error(
            "Database may not exist. Run the following commands to create it:(update password if different)\n\n"
            "  set PGPASSWORD=postgres\n"
            '  psql -h localhost -p 5432 -U postgres -c "CREATE DATABASE jql_vectordb;"\n'
            "  set PGPASSWORD=\n"
        )
        raise
    return model


from pathlib import Path

PROMPTS_DIR = Path(__file__).parent 

def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


# -- JQL generation ---------------------------------------------------

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
    system_prompt = load_prompt("./src/system_prompt.md")
    return (
        str(system_prompt) +
        f"\n\nUse the JQL examples below as references. "
        f"Each example is in the format '-- <annotation>\\n<jql>'.\n"
        f"JQL rules:\n"
        f"- Do not use placeholder values like 'ProjectName' or 'USERNAME'.\n"
        f"- If no specific project is mentioned, omit the project filter.\n"
        f"- Do NOT use date arithmetic between two fields. JQL does not support this.\n"
        f"  INVALID: resolutiondate >= created + 20d\n"
        f"  INVALID: resolutiondate - created > 20d\n"
        f"  CORRECT: project = ZOOKEEPER AND resolution IS NOT EMPTY ORDER BY resolutiondate DESC\n"
        f"  (duration filtering like 'took more than N days' is handled separately — omit it from JQL)\n"
        f"- Do NOT append LIMIT — result count is controlled externally.\n\n"
        f"Examples:\n{example_block}\n\n"
        f"User request: {query}\n"
        f"JSON:"
    )


async def generate_jql(query: str, model: SentenceTransformer) -> dict:
    """Generate a structured LLM response from a natural language query using RAG + Ollama.

    Retrieves the top-5 most semantically similar (annotation, JQL) pairs from
    pgvector, builds a few-shot prompt, and sends it to the local Ollama LLM.

    The LLM returns JSON with keys: ``jql``, ``chart_spec``, ``answer``.
    For general (non-Jira) queries ``jql`` and ``chart_spec`` are null.

    Args:
        query: The user's natural language query string.
        model: SentenceTransformer model used to encode the query for similarity search.

    Returns:
        dict with keys:
            - ``jql``        (str | None)  — JQL string, or None for general queries.
            - ``chart_spec`` (dict | None) — chart visualisation spec, or None.
            - ``answer``     (str)         — human-readable description or general answer.

    Raises:
        RuntimeError: If pgvector returns no examples (DB not loaded) or Ollama unreachable.
    """
    examples, _ = test_embeddings_jql(query, model)
    if not examples:
        raise RuntimeError("No examples found in pgvector — was the DB loaded?")

    prompt = _build_prompt(query, examples)
    logger.debug("Ollama prompt:\n%s", prompt)

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": OLLAMA_TEMPERATURE}},
            )
            response.raise_for_status()
            text = response.json()["response"].strip()
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_URL}. "
            "Ensure Ollama is running: ollama serve"
        )

    # Strip any accidental markdown fences
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE).strip("`").strip()

    try:
        result = json.loads(text)
        if "jql" not in result:
            raise ValueError("missing 'jql' key")
    except (json.JSONDecodeError, ValueError):
        logger.warning("LLM response was not valid JSON — treating as general answer.")
        return {"jql": None, "chart_spec": None, "answer": text}

    if not result.get("jql"):
        logger.info("Non-JQL query detected — returning general answer without Jira REST API call.")
    return {
        "jql":        result.get("jql") or None,
        "chart_spec": result.get("chart_spec"),
        "answer":     result.get("answer", ""),
    }


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


# -- Query helpers ----------------------------------------------------

def _fields_from_order_by(jql: str) -> list[str]:
    """Return FIELD_META keys found in the ORDER BY clause of the given JQL."""
    m = re.search(r"\bORDER\s+BY\b(.+?)(?=\bLIMIT\b|$)", jql, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    result = []
    for token in re.split(r"[,\s]+", m.group(1).strip()):
        token = token.strip()
        if token.upper() in ("ASC", "DESC", ""):
            continue
        for key in FIELD_META:
            if key.lower() == token.lower():
                result.append(key)
                break
    return result


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

    When multiple per-project counts are stated (e.g. "4 issues from HIVE and
    5 from KAFKA") their values are summed so all requested issues are fetched.
    Falls back to MAX_RESULTS only when the query contains no explicit count.

    Args:
        query: The user's natural language query string.

    Returns:
        int: The sum of all explicit counts found, or MAX_RESULTS if none.
    """
    # Per-project counts: "4 issues from HIVE and 5 issues from KAFKA"
    per_project = re.findall(r'\b(\d+)\s+issues?\s+(?:from\s+)?(?:project\s+)?[A-Z][A-Z0-9_]*', query, re.IGNORECASE)
    if len(per_project) > 1:
        return sum(int(n) for n in per_project)

    # Single explicit count: "list 5 issues" / "top 10 bugs"
    m = re.search(r'\b(\d+)\s+issue', query, re.IGNORECASE)
    if not m:
        m = re.search(r'\b(?:top|first|last|show|list|get|fetch)\s+(\d+)\b', query, re.IGNORECASE)
    return int(m.group(1)) if m else MAX_RESULTS


def _parse_per_project_limits(query: str) -> dict[str, int]:
    """Detect per-project counts like '2 issues from HIVE and 4 from HADOOP'.

    Returns {PROJECT: limit} when different counts per project are found,
    empty dict otherwise (single limit applies to all projects).
    """
    matches = re.findall(
        r'\b(\d+)\s+issues?\s+(?:from\s+)?(?:project\s+)?([A-Z][A-Z0-9_]*)',
        query,
        re.IGNORECASE,
    )
    if len(matches) < 2:
        return {}
    return {proj.upper(): int(n) for n, proj in matches}


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


# -- Output -----------------------------------------------------------

def _print_table(result: dict) -> None:
    """Render the structured result dict returned by atlasmind() as a table."""
    display_fields = result["display_fields"]
    issues         = result["issues"]

    # Column specs: (header_label, fixed_width)
    cols: list[tuple[str, int]] = [("Key", 12)]
    for field_name in display_fields:
        label, width, _ = FIELD_META[field_name]
        cols.append((label, width))
    cols.append(("Summary", 0))

    header      = "".join(label.ljust(width) for label, width in cols[:-1]) + cols[-1][0]
    total_width = max(80, len(header) + 10)

    print(f"\nProfile : {result['profile']}  ({result['jira_base_url']})")
    print(f"JQL     : {result['jql']}")
    if result["post_filters"]:
        print(f"Filter  : {' AND '.join(result['post_filters'])}  (applied in Python after fetch)")
    print(f"Found {result['total']} issue(s) in Jira, showing {result['shown']} (examined {result['examined']}):\n")
    print(header)
    print("-" * total_width)

    for issue in issues:
        row = ""
        for field_name, (_, width) in zip(
            ["key"] + display_fields,
            cols[:-1],
        ):
            row += str(issue.get(field_name, ""))[:width].ljust(width)
        row += str(issue.get("summary", ""))
        print(row)

    print("-" * total_width)
    if result["post_filters"]:
        print(
            f"Retrieved {result['shown']} matching issue(s) "
            f"(examined {result['examined']} of {result['total']} total; post-filter applied)."
        )
    else:
        print(f"Retrieved {result['shown']} of {result['total']} total issue(s).")


# -- CLI --------------------------------------------------------------

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
        epilog="""
modes:
  CLI (default)   Run a single query and print results as a table to stdout.
  --server        Start a FastAPI HTTP server; results are returned as JSON.
                  Endpoint: GET /query?q=<natural language>&profile=<name>&limit=<n>

examples:
  # Single query (CLI)
  uv run python main.py --query "list 5 open bugs in KAFKA"
  uv run python main.py --query "show issues resolved in ZOOKEEPER last 30 days" --profile work

  # Server mode
  uv run python main.py --server
  uv run python main.py --server --host 127.0.0.1 --port 9000

  # Then query the server
  curl "http://localhost:8000/query?q=list+5+bugs+in+KAFKA"
  curl "http://localhost:8000/query?q=open+issues+in+HADOOP&limit=20&profile=work"
""",
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
    parser.add_argument(
        "--server", action="store_true",
        help="Start AtlasMind as a FastAPI HTTP server instead of running a single query.",
    )
    parser.add_argument(
        "--check-mcp", action="store_true",
        help="Check connectivity to the Atlassian Rovo MCP server and list available tools.",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", metavar="HOST",
        help="Host to bind the server to (default: 0.0.0.0). Only used with --server.",
    )
    parser.add_argument(
        "--port", type=int, default=8000, metavar="PORT",
        help="Port to bind the server to (default: 8000). Only used with --server.",
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

    if args.check_mcp:
        profile = get_profile(args.profile)
        logger.info("Profile : %s", profile.name)
        logger.info("Jira URL: %s", profile.jira_base_url)
        asyncio.run(check_rovo_mcp(profile))
        return

    if args.server:
        import uvicorn
        from server import app
        logger.info("Starting AtlasMind server on %s:%d", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    profile = get_profile(args.profile)
    logger.info("Profile : %s - %s", profile.name, profile.jira_base_url)

    # -- 1. Load annotation embeddings into pgvector ------------------
    logger.info("Loading JQL annotations into pgvector from: %s", args.annotation_file)
    model = load_annotations_into_db(args.annotation_file)

    # -- 2. Generate JQL via similarity search + Ollama ---------------
    logger.info("Query   : %s", args.query)
    llm_result = asyncio.run(generate_jql(args.query, model))
    if not llm_result["jql"]:
        print(llm_result["answer"])
        return
    jql, jql_limit = _sanitize_jql(llm_result["jql"])
    logger.info("Generated JQL : %s", jql)

    # -- 3. Run JQL against Jira --------------------------------------
    # jql_limit (from LLM LIMIT clause) takes precedence over NL query limit
    limit        = jql_limit or _parse_limit(args.query)
    fields       = _detect_fields(args.query)
    post_filters = _detect_post_filters(args.query)

    # Surface any fields the LLM put in ORDER BY that aren't already shown
    for order_field in _fields_from_order_by(jql):
        if order_field not in fields and order_field in FIELD_META:
            fields.append(order_field)

    for pf in post_filters:
        if pf.field not in fields and pf.field in FIELD_META:
            fields.insert(0, pf.field)
    fields = fields[:9]

    per_project = _parse_per_project_limits(args.query)
    if per_project:
        result = asyncio.run(atlasmind_multi_project(
            profile,
            jql_query=jql,
            per_project_limits=per_project,
            fields=fields,
            post_filters=post_filters,
        ))
    else:
        result = asyncio.run(atlasmind(
            profile,
            jql_query=jql,
            max_results=limit,
            fields=fields,
            post_filters=post_filters,
        ))
    if result:
        _print_table(result)


if __name__ == "__main__":
    main()

import argparse
import asyncio
import re
import sys
from typing import NamedTuple

from atlasmind import atlasmind, MAX_RESULTS
from config import get_profile, print_profiles
from config_fields import (
    DEFAULT_DISPLAY_FIELDS, FIELD_META, QUERY_FIELD_MAP,
    POST_FILTER_PATTERNS, POST_FILTER_FETCH_MULTIPLIER,
)
from jql_generator import get_generator


class PostFilter(NamedTuple):
    field:     str   # e.g. "days_to_fix"
    operator:  str   # ">", ">=", "<", "<="
    threshold: int   # canonical value (days)


def _detect_fields(query: str) -> list[str]:
    """Return ordered list of Jira field names referenced in the query (max 9, summary added by atlasmind)."""
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
    """Extract a numeric limit from the natural language query, or return MAX_RESULTS."""
    m = re.search(r'\b(\d+)\s+issue', query, re.IGNORECASE)
    if not m:
        m = re.search(r'\b(?:top|first|last|show|list|get|fetch)\s+(\d+)\b', query, re.IGNORECASE)
    if not m:
        m = re.search(r'\b(\d+)\b', query)
    return int(m.group(1)) if m else MAX_RESULTS


def _detect_post_filters(query: str) -> list[PostFilter]:
    """Detect time-based constraints that JQL cannot express and return them as PostFilters."""
    filters: list[PostFilter] = []
    seen: set[PostFilter] = set()
    for pattern, field_name, operator, unit_map in POST_FILTER_PATTERNS:
        m = pattern.search(query)
        if m:
            quantity  = int(m.group(1))
            unit_stem = m.group(2).lower().rstrip("s")   # "days"→"day", "weeks"→"week"
            threshold = quantity * unit_map.get(unit_stem, 1)
            pf = PostFilter(field=field_name, operator=operator, threshold=threshold)
            if pf not in seen:
                filters.append(pf)
                seen.add(pf)
    return filters


_DESCRIPTION = """\
AtlasMind — Query Jira using natural language.

Converts a plain-English question into JQL, runs it against a configured
Jira instance, and displays the results.

Examples:
  uv run python main.py --query "list open bugs in KAFKA"
  uv run python main.py --query "show 20 critical issues updated this week"
  uv run python main.py --profile personal --query "my unresolved tasks"
  uv run python main.py --backend rovo --query "all blockers assigned to me"
  uv run python main.py --examples-file data/my_queries.json --query "open issues"
  uv run python main.py --list-profiles
"""

_EPILOG = """\
Environment variables (all overridden by the corresponding CLI flag):
  ATLASMIND_PROFILE     Default profile name (same as --profile)
  JQL_BACKEND           Default backend: 'local' or 'rovo' (same as --backend)
  JQL_LOCAL_MODEL       Ollama model name  (default: qwen2.5-coder:7b-instruct)
  JQL_OLLAMA_URL        Ollama base URL    (default: http://localhost:11434)
  JQL_EXAMPLES_FOLDER   Path to folder of JQL examples files (default: data/)
  JQL_EXAMPLES_FILE     Path to a single JQL examples file (same as --examples-file)
  ATLASSIAN_OAUTH_TOKEN OAuth 2.1 bearer token required for the 'rovo' backend
  ATLASSIAN_TOKEN       Fallback Jira API token (used when not set per-profile)

Profile configuration:
  Profiles are stored in profiles.json (gitignored).
  Copy profiles.json.example → profiles.json and fill in your credentials.
  Each profile supports: jira_url, email, token, client_id, client_secret,
  jira_type ('cloud' | 'server').

JQL examples file formats (--examples-file):
  .md   Markdown with ```jql blocks; entries prefixed "-- N. Description"
  .json [{"description": "...", "jql": "..."}]  or  [["desc", "jql"], ...]
  .csv  Two columns: description, jql  (header row optional)
"""


def parse_args():
    parser = argparse.ArgumentParser(
        prog="atlasmind",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    query_group = parser.add_argument_group("query options")
    query_group.add_argument(
        "--query", default="list all open issues", metavar="TEXT",
        help=(
            "Natural language question to convert to JQL and run against Jira. "
            "The result limit is inferred from the query (e.g. 'list 20 issues' → 20); "
            f"defaults to {MAX_RESULTS} when no number is mentioned. "
            "(default: 'list all open issues')"
        ),
    )

    profile_group = parser.add_argument_group("profile & connection options")
    profile_group.add_argument(
        "--profile", default=None, metavar="NAME",
        help=(
            "Atlassian profile to use, as defined in profiles.json. "
            "Overrides ATLASMIND_PROFILE env var. "
            "Run --list-profiles to see available profiles."
        ),
    )
    profile_group.add_argument(
        "--list-profiles", action="store_true",
        help="List all configured profiles and exit.",
    )

    backend_group = parser.add_argument_group("JQL generation backend options")
    backend_group.add_argument(
        "--backend", choices=["local", "rovo"], default=None, metavar="{local,rovo}",
        help=(
            "JQL generator backend. "
            "'local' uses a local Ollama model (requires Ollama running). "
            "'rovo' uses the Atlassian Rovo MCP server (requires ATLASSIAN_OAUTH_TOKEN). "
            "Overrides JQL_BACKEND env var. (default: local)"
        ),
    )
    backend_group.add_argument(
        "--examples-file", default=None, metavar="PATH",
        help=(
            "Path to a single JQL examples file used as few-shot context for the local LLM. "
            "Supported formats: .md (markdown with ```jql blocks), "
            ".json ([{description, jql}]), .csv (description, jql columns). "
            "Overrides JQL_EXAMPLES_FILE env var. "
            "(default: data/apache_jira_500_jql_queries.md)"
        ),
    )
    backend_group.add_argument(
        "--examples-folder", default=None, metavar="PATH",
        help=(
            "Path to a folder containing JQL examples files (.md, .json, .csv). "
            "All supported files in the folder are loaded and merged. "
            "Takes precedence over --examples-file when both are given. "
            "Overrides JQL_EXAMPLES_FOLDER env var. "
            "(default: data/)"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_profiles:
        print_profiles()
        return

    # Load profile and backend
    profile   = get_profile(args.profile)
    generator = get_generator(
        args.backend,
        examples_file   = args.examples_file,
        examples_folder = args.examples_folder,
    )

    print(f"Profile : {profile.name} — {profile.jira_base_url} ({profile.email})")
    print(f"Backend : {type(generator).__name__}\n")

    # Health check for the selected backend only
    ok = asyncio.run(generator.health_check())
    print()
    if not ok:
        print("Backend is not available. Exiting.")
        sys.exit(1)

    # Generate JQL from natural language
    print(f"Query         : {args.query}")
    jql = asyncio.run(generator.generate(args.query, profile=profile))
    print(f"Generated JQL : {jql}\n")

    # Run JQL against Jira
    limit        = _parse_limit(args.query)
    fields       = _detect_fields(args.query)
    post_filters = _detect_post_filters(args.query)

    # Ensure post-filter fields appear in the display
    for pf in post_filters:
        if pf.field not in fields and pf.field in FIELD_META:
            fields.insert(0, pf.field)
    fields = fields[:9]

    asyncio.run(atlasmind(profile, jql_query=jql, max_results=limit,
                          fields=fields, post_filters=post_filters))


if __name__ == "__main__":
    main()

"""
JQL validation and auto-correction pipeline.
Adapted from Jira-Whisperer for async httpx + AtlasMind Profile.

Fetches Jira metadata (priorities, statuses, project keys) once per profile
and caches it in-process. Uses the cache to:

  1. Fix DURING / BETWEEN syntax  → >= / <= equivalents
  2. Fix wrong priority names     → closest valid name via alias table + fuzzy match
  3. Quote unquoted multi-word values inside IN(…) clauses

Also provides relax_and_retry() for zero-results recovery.
"""

import re
import difflib
import logging
from dataclasses import dataclass, field

from config_fields import FIELD_ALIASES

import httpx

logger = logging.getLogger(__name__)


# ── Data models ─────────────────────────────────────────────────────

@dataclass
class JiraMetadata:
    priorities  : list[str] = field(default_factory=list)
    project_keys: list[str] = field(default_factory=list)
    project_names: dict[str, str] = field(default_factory=dict)   # key → name
    statuses    : list[str] = field(default_factory=list)
    field_names : list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    original_jql: str
    fixed_jql   : str
    changes     : list[str]


# ── Metadata cache (keyed by profile name) ──────────────────────────

_cache: dict[str, JiraMetadata] = {}


async def fetch_metadata(profile, force_refresh: bool = False) -> JiraMetadata:
    """Fetch and cache Jira metadata for the given profile."""
    if profile.name in _cache and not force_refresh:
        return _cache[profile.name]

    auth    = (profile.email, profile.token) if profile.email and profile.token else None
    headers = {"Accept": "application/json"}
    base    = profile.jira_base_url

    timeout = httpx.Timeout(connect=10, read=30, write=10, pool=5)

    async with httpx.AsyncClient(auth=auth, headers=headers, timeout=timeout) as client:
        async def _get(path: str) -> list:
            r = await client.get(f"{base}/rest/api/2/{path}")
            r.raise_for_status()
            return r.json()

        priorities    = [p["name"] for p in await _get("priority")]
        projects_raw  = await _get("project")
        project_names = {p["key"]: p["name"] for p in projects_raw}
        project_keys  = list(project_names.keys())
        statuses      = [s["name"] for s in await _get("status")]
        field_names   = [f["name"] for f in await _get("field")]

        meta = JiraMetadata(
            priorities    = priorities,
            project_keys  = project_keys,
            project_names = project_names,
            statuses      = statuses,
            field_names   = field_names,
        )

    _cache[profile.name] = meta
    logger.debug(
        "Metadata cached for '%s': %d priorities, %d projects, %d statuses",
        profile.name, len(priorities), len(project_keys), len(statuses),
    )
    return meta


def clear_cache(profile_name: str | None = None) -> None:
    if profile_name:
        _cache.pop(profile_name, None)
    else:
        _cache.clear()


# ── Fixer 1 — DURING syntax ─────────────────────────────────────────
# created DURING ("2023-01-01","2023-12-31")
# → created >= "2023-01-01" AND created <= "2023-12-31"

_DURING_RE = re.compile(
    r'(\w+)\s+DURING\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)',
    re.IGNORECASE,
)


def _fix_during_syntax(jql: str) -> tuple[str, list[str]]:
    changes: list[str] = []

    def _repl(m: re.Match) -> str:
        field, d1, d2 = m.group(1), m.group(2), m.group(3)
        changes.append(f'DURING syntax: replaced `{m.group(0)}` with >= / <= form')
        return f'{field} >= "{d1}" AND {field} <= "{d2}"'

    return _DURING_RE.sub(_repl, jql), changes


# ── Fixer 2 — BETWEEN syntax ────────────────────────────────────────
# created BETWEEN "2023-01-01" AND "2023-12-31"
# → created >= "2023-01-01" AND created <= "2023-12-31"

_BETWEEN_RE = re.compile(
    r'(\w+)\s+BETWEEN\s+"([^"]+)"\s+AND\s+"([^"]+)"',
    re.IGNORECASE,
)


def _fix_between_syntax(jql: str) -> tuple[str, list[str]]:
    changes: list[str] = []

    def _repl(m: re.Match) -> str:
        field, d1, d2 = m.group(1), m.group(2), m.group(3)
        changes.append(f'BETWEEN syntax: replaced `{m.group(0)}` with >= / <= form')
        return f'{field} >= "{d1}" AND {field} <= "{d2}"'

    return _BETWEEN_RE.sub(_repl, jql), changes


# ── Fixer 3 — wrong priority names ──────────────────────────────────

_PRIORITY_ALIASES: dict[str, list[str]] = {
    "critical": ["blocker", "critical"],
    "high"    : ["major", "high"],
    "medium"  : ["minor", "medium", "normal"],
    "low"     : ["minor", "low", "trivial"],
    "urgent"  : ["blocker", "critical"],
    "normal"  : ["minor", "medium", "normal"],
}

_IN_PRIORITY_RE = re.compile(r'(priority\s+in\s*\()([^)]+)(\))', re.IGNORECASE)
_EQ_PRIORITY_RE = re.compile(r'(priority\s*=\s*)("[^"]*"|\S+)',   re.IGNORECASE)


def _fix_priority_values(jql: str, valid_priorities: list[str]) -> tuple[str, list[str]]:
    changes: list[str] = []
    valid_lower = {p.lower(): p for p in valid_priorities}

    def _replace_token(token: str) -> str:
        raw = token.strip().strip('"').strip("'")
        if raw.lower() in valid_lower:
            return token
        for alias, candidates in _PRIORITY_ALIASES.items():
            if raw.lower() == alias:
                for c in candidates:
                    if c in valid_lower:
                        replacement = valid_lower[c]
                        changes.append(f'Priority alias: "{raw}" → "{replacement}"')
                        return f'"{replacement}"' if " " in replacement else replacement
        matches = difflib.get_close_matches(raw.lower(), valid_lower.keys(), n=1, cutoff=0.6)
        if matches:
            replacement = valid_lower[matches[0]]
            changes.append(f'Priority fuzzy match: "{raw}" → "{replacement}"')
            return f'"{replacement}"' if " " in replacement else replacement
        return token

    def _fix_in(m: re.Match) -> str:
        tokens = [t.strip() for t in m.group(2).split(",")]
        return f"{m.group(1)}{', '.join(_replace_token(t) for t in tokens)}{m.group(3)}"

    def _fix_eq(m: re.Match) -> str:
        return f"{m.group(1)}{_replace_token(m.group(2))}"

    fixed = _IN_PRIORITY_RE.sub(_fix_in, jql)
    fixed = _EQ_PRIORITY_RE.sub(_fix_eq, fixed)
    return fixed, changes


# ── Fixer 4 — unquoted multi-word values in IN(…) ───────────────────
# status in (Open, In Progress) → status in (Open, "In Progress")

_IN_VALUES_RE = re.compile(r'(\bin\s*\()([^)]+)(\))', re.IGNORECASE)


def _fix_unquoted_values(jql: str) -> tuple[str, list[str]]:
    changes: list[str] = []

    def _fix_list(m: re.Match) -> str:
        def _quote(token: str) -> str:
            t = token.strip()
            if t.startswith('"') or t.startswith("'"):
                return t
            if " " in t:
                changes.append(f'Quoted multi-word value: {t}')
                return f'"{t}"'
            return t
        tokens = [t.strip() for t in m.group(2).split(",")]
        return f"{m.group(1)}{', '.join(_quote(t) for t in tokens)}{m.group(3)}"

    return _IN_VALUES_RE.sub(_fix_list, jql), changes


# ── Fixer 5 — strip LIMIT clause ────────────────────────────────────
# LIMIT does not exist in JQL; models often add it anyway.

_LIMIT_RE = re.compile(r'\s+LIMIT\s+\d+', re.IGNORECASE)


def _fix_limit_clause(jql: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    fixed = _LIMIT_RE.sub("", jql).strip()
    if fixed != jql:
        changes.append("Removed invalid LIMIT clause (use maxResults API param instead)")
    return fixed, changes


# ── Fixer 6 — strip issueFunction clauses ───────────────────────────
# issueFunction requires the ScriptRunner plugin; it is not standard JQL.
# Pattern: issueFunction in someFn(...) possibly preceded/followed by AND/OR

_ISSUE_FUNCTION_RE = re.compile(
    r'issueFunction\s+in\s+\w+\s*\([^)]*\)\s*(?:AND\s+|OR\s+)?',
    re.IGNORECASE,
)
# Also catch trailing AND/OR if issueFunction appears at the start
_LEADING_AND_RE = re.compile(r'^\s*AND\s+', re.IGNORECASE)


def _fix_issue_function(jql: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    fixed = _ISSUE_FUNCTION_RE.sub("", jql)
    fixed = _LEADING_AND_RE.sub("", fixed).strip()
    if fixed != jql:
        changes.append("Removed unsupported issueFunction clause (ScriptRunner plugin not available in standard JQL)")
    return fixed, changes


# ── Fixer 7 — fix invalid field name aliases ────────────────────────
# Loaded from config_fields.py — edit that file to add project-specific aliases.

_ALIAS_LOWER = {re.compile(k, re.IGNORECASE): v for k, v in FIELD_ALIASES.items()}


def _fix_field_aliases(jql: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    fixed = jql
    for pattern, replacement in _ALIAS_LOWER.items():
        new = pattern.sub(replacement, fixed)
        if new != fixed:
            alias = pattern.pattern.replace(r'\b', '').strip()
            changes.append(f"Field alias: '{alias}' → '{replacement}'")
            fixed = new
    return fixed, changes


# ── Fixer 8 — strip date arithmetic between two fields ──────────────
# JQL does not support field-to-field arithmetic like:
#   resolved <= created + 10d   resolved >= updated - 5d
# These use the reserved '+' / '-' characters and will cause a 400 error.

_DATE_ARITH_RE = re.compile(
    r'(?:AND\s+)?\w+\s*(?:<=|>=|<|>|=)\s*\(?\w+\s*[+\-]\s*\d+[dwm]\)?',
    re.IGNORECASE,
)


def _fix_date_arithmetic(jql: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    fixed = _DATE_ARITH_RE.sub("", jql)
    fixed = _LEADING_AND_RE.sub("", fixed).strip()
    # Clean up any trailing AND that was left behind
    fixed = re.sub(r'\s+AND\s*$', '', fixed, flags=re.IGNORECASE).strip()
    if fixed != jql:
        changes.append(
            "Removed unsupported date arithmetic (JQL does not support field ± Nd expressions; "
            "use absolute date offsets on a single field, e.g. created >= '-10d')"
        )
    return fixed, changes


# ── Main validation pipeline ─────────────────────────────────────────

def validate_and_fix(jql: str, metadata: JiraMetadata) -> ValidationResult:
    """Run all fixers in order and return a ValidationResult with the change log."""
    all_changes: list[str] = []
    current = jql

    current, c = _fix_issue_function(current);                           all_changes.extend(c)
    current, c = _fix_field_aliases(current);                            all_changes.extend(c)
    current, c = _fix_date_arithmetic(current);                          all_changes.extend(c)
    current, c = _fix_limit_clause(current);                             all_changes.extend(c)
    current, c = _fix_during_syntax(current);                            all_changes.extend(c)
    current, c = _fix_between_syntax(current);                           all_changes.extend(c)
    current, c = _fix_priority_values(current, metadata.priorities);     all_changes.extend(c)
    current, c = _fix_unquoted_values(current);                          all_changes.extend(c)

    if all_changes:
        logger.info("JQL auto-corrected (%d change(s)): %s → %s", len(all_changes), jql, current)

    return ValidationResult(original_jql=jql, fixed_jql=current, changes=all_changes)


# ── Zero-results retry ───────────────────────────────────────────────

_RELAX_PATTERNS: list[tuple[str, str]] = [
    (r'\s+AND\s+\w+\s*>=\s*"[^"]*"',                                   "removed date >= constraint"),
    (r'\s+AND\s+\w+\s*<=\s*"[^"]*"',                                   "removed date <= constraint"),
    (r'\s+AND\s+\w+\s*>\s*"[^"]*"',                                    "removed date > constraint"),
    (r'\s+AND\s+\w+\s*<\s*"[^"]*"',                                    "removed date < constraint"),
    (r'\s+AND\s+priority\s+in\s*\([^)]*\)',                             "removed priority IN filter"),
    (r'\s+AND\s+priority\s*=\s*(?:"[^"]*"|\S+)',                        "removed priority = filter"),
    (r'\s+AND\s+assignee\s*(?:in\s*\([^)]*\)|=\s*(?:"[^"]*"|\S+))',    "removed assignee filter"),
]


def relax_and_retry(jql: str) -> tuple[str, list[str]]:
    """Strip non-essential constraints to broaden a zero-results query."""
    relaxed = jql
    removed: list[str] = []

    for pattern, description in _RELAX_PATTERNS:
        new_jql = re.sub(pattern, "", relaxed, flags=re.IGNORECASE).strip()
        if new_jql != relaxed:
            removed.append(description)
            relaxed = new_jql

    relaxed = re.sub(r'^\s*AND\s+', '', relaxed, flags=re.IGNORECASE).strip()

    if removed:
        logger.info("JQL relaxed: %s → %s  (removed: %s)", jql, relaxed, removed)

    return relaxed, removed

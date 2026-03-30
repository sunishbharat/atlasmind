"""
Field configuration for AtlasMind.

Edit this file to customise field behaviour for your Jira project / dataset.
No other code changes are needed.

Three sections to configure:
  1. FIELD_META         — display label, column width, and value extractor per field
  2. DEFAULT_DISPLAY_FIELDS — fields shown when the query doesn't mention any specific ones
  3. QUERY_FIELD_MAP    — maps natural language keywords → Jira API field names
  4. FIELD_ALIASES      — invalid field names the LLM tends to generate → correct names

Adding a custom field example:
    In FIELD_META add:
        "customfield_10020": ("Sprint", 20, lambda f: (f.get("customfield_10020") or {}).get("name", "")),
    In QUERY_FIELD_MAP add:
        (r'\\bsprint\\b', "customfield_10020"),
"""

import re as _re
from datetime import datetime

# ── Date helpers (used by computed fields below) ─────────────────────

def _parse_jira_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _days_to_fix(f: dict) -> str:
    """Computed field: days between created and resolutiondate."""
    resolved = _parse_jira_dt(f.get("resolutiondate"))
    created  = _parse_jira_dt(f.get("created"))
    if resolved and created:
        return str((resolved - created).days)
    return ""


# ── 1. Field metadata ─────────────────────────────────────────────────
#
# Format:  "jira_api_field_name": ("Display Label", column_width, extractor_fn)
#
# extractor_fn receives the issue's `fields` dict and returns a display string.
# column_width is the fixed character width for the column (0 = fill remaining).
# Summary is always appended last automatically — do not add it here.
#
# Computed fields (days_to_fix) are virtual — they are NOT sent to the API
# but calculated from other fields fetched behind the scenes.

FIELD_META: dict[str, tuple[str, int, callable]] = {
    "status"        : ("Status",       15, lambda f: (f.get("status") or {}).get("name", "")),
    "assignee"      : ("Assignee",     20, lambda f: (f.get("assignee") or {}).get("displayName", "Unassigned")),
    "reporter"      : ("Reporter",     20, lambda f: (f.get("reporter") or {}).get("displayName", "")),
    "priority"      : ("Priority",     10, lambda f: (f.get("priority") or {}).get("name", "")),
    "issuetype"     : ("Type",         14, lambda f: (f.get("issuetype") or {}).get("name", "")),
    "created"       : ("Created",      12, lambda f: (f.get("created") or "")[:10]),
    "updated"       : ("Updated",      12, lambda f: (f.get("updated") or "")[:10]),
    "resolutiondate": ("Resolved",     12, lambda f: (f.get("resolutiondate") or "")[:10]),
    "duedate"       : ("Due Date",     12, lambda f: f.get("duedate") or ""),
    "resolution"    : ("Resolution",   14, lambda f: (f.get("resolution") or {}).get("name", "Unresolved")),
    "components"    : ("Components",   20, lambda f: ", ".join(c["name"] for c in f.get("components", []))),
    "fixVersions"   : ("Fix Version",  14, lambda f: ", ".join(v["name"] for v in f.get("fixVersions", []))),
    "labels"        : ("Labels",       20, lambda f: ", ".join(f.get("labels", []))),
    "description"   : ("Description",  40, lambda f: ((f.get("description") or "")[:38] + "..") if len(f.get("description") or "") > 40 else (f.get("description") or "")),
    # ── Computed fields ───────────────────────────────────────────────
    # These are calculated locally; their dependencies are auto-fetched.
    "days_to_fix"   : ("Days To Fix",  12, _days_to_fix),
    # ── Custom fields ─────────────────────────────────────────────────
    # Uncomment and adjust for your Jira instance, e.g.:
    # "customfield_10020": ("Sprint",  20, lambda f: (f.get("customfield_10020") or [{}])[-1].get("name", "")),
    # "customfield_10016": ("Story Pts", 8, lambda f: str(f.get("customfield_10016") or "")),
}

# Fields that require fetching additional API fields to compute their value.
# Format: computed_field_name → [required_api_fields]
COMPUTED_FIELD_DEPS: dict[str, list[str]] = {
    "days_to_fix": ["created", "resolutiondate"],
}


# ── 2. Default display fields ─────────────────────────────────────────
#
# Shown when the user's query doesn't mention any specific fields.
# Must be keys from FIELD_META above.

DEFAULT_DISPLAY_FIELDS: list[str] = ["status", "assignee", "created"]


# ── 3. Query field map ────────────────────────────────────────────────
#
# Maps regex patterns (matched against the user's natural language query)
# to Jira API field names.  Checked in order — first match wins per field.
# Add entries here to recognise custom terminology or project-specific fields.

QUERY_FIELD_MAP: list[tuple[str, str]] = [
    (r'\bassignee\b',                                                                         "assignee"),
    (r'\breporter\b',                                                                         "reporter"),
    (r'\bpriority\b',                                                                         "priority"),
    (r'\bstatus\b',                                                                           "status"),
    (r'\btype\b|\bissue\s*type\b',                                                            "issuetype"),
    (r'\bcreated\b',                                                                          "created"),
    (r'\bupdated\b|\bmodified\b',                                                             "updated"),
    (r'\bresolved\b|\bclosed\b|\bclose\b',                                                    "resolutiondate"),
    (r'\bdays?\s+to\s+(?:fix|close|resolve|complete)\b'
     r'|\btime\s+to\s+(?:fix|close|resolve)\b'
     r'|\bhow\s+(?:long|many\s+days)\b'
     r'|\bnumber\s+of\s+days\b',                                                              "days_to_fix"),
    (r'\bdue\b|\bdue\s*date\b',                                                               "duedate"),
    (r'\bresolution\b',                                                                       "resolution"),
    (r'\bcomponent\b',                                                                        "components"),
    (r'\bfix\s*version\b|\bfixversion\b',                                                     "fixVersions"),
    (r'\blabel\b',                                                                            "labels"),
    (r'\bdescription\b',                                                                      "description"),
    # ── Add custom field patterns below ───────────────────────────────
    # (r'\bsprint\b',   "customfield_10020"),
    # (r'\bstory\s*points?\b|\bsp\b',  "customfield_10016"),
]


# ── 4. Field aliases (JQL validator auto-correction) ─────────────────
#
# Maps regex patterns of invalid field names the LLM tends to generate
# → the correct Jira JQL field name.
# Applied by jql_validator before sending the query to Jira.

FIELD_ALIASES: dict[str, str] = {
    r'\bclosed\b'       : 'resolved',
    r'\bclose_date\b'   : 'resolved',
    r'\bcloseDate\b'    : 'resolved',
    r'\bcompletion\b'   : 'resolved',
    r'\bdue\b'          : 'duedate',
    r'\bdue_date\b'     : 'duedate',
    r'\bdueDate\b'      : 'duedate',
    r'\bmodified\b'     : 'updated',
    r'\blast_updated\b' : 'updated',
    r'\btype\b(?=\s*(?:=|!=|in\b|not\b))': 'issuetype',
    # ── Add project-specific aliases below ────────────────────────────
    # r'\bresolutionDate\b': 'resolutiondate',
}


# ── 5. Post-processing filters ────────────────────────────────────────
#
# Constraints that JQL cannot express (e.g. date arithmetic between two fields)
# are detected from the NL query and applied in Python after fetching from Jira.
#
# POST_FILTER_PATTERNS: list of (regex, field_name, operator, unit_map)
#   regex      — must capture group 1 = quantity (int), group 2 = unit word
#   field_name — key in FIELD_META whose extractor returns a numeric string
#   operator   — one of ">", ">=", "<", "<="
#   unit_map   — maps unit stem → days  e.g. {"day": 1, "week": 7, "month": 30}
#
# POST_FILTER_FETCH_MULTIPLIER: factor applied to max_results when fetching from
# Jira to compensate for records filtered out in Python.  Raise if filters are
# highly selective (e.g. "took more than 90 days").

POST_FILTER_FETCH_MULTIPLIER: int = 20

POST_FILTER_PATTERNS: list[tuple] = [
    # "took more than N days/weeks/months to fix/close/resolve"
    (_re.compile(r'\btook?\s+more\s+than\s+(\d+)\s*(day|week|month)s?\b', _re.IGNORECASE),
     "days_to_fix", ">",  {"day": 1, "week": 7, "month": 30}),
    # "more than N days to fix/resolve/close"  (alternative word order)
    (_re.compile(r'\bmore\s+than\s+(\d+)\s*(day|week|month)s?\s+to\s+(?:fix|resolve|close)\b', _re.IGNORECASE),
     "days_to_fix", ">",  {"day": 1, "week": 7, "month": 30}),
    # "took at least N days/weeks/months"
    (_re.compile(r'\btook?\s+at\s+least\s+(\d+)\s*(day|week|month)s?\b', _re.IGNORECASE),
     "days_to_fix", ">=", {"day": 1, "week": 7, "month": 30}),
    # "fixed/resolved/closed in less than N days/weeks/months"
    (_re.compile(r'\b(?:fix(?:ed)?|resolved?|closed?)\s+in\s+less\s+than\s+(\d+)\s*(day|week|month)s?\b', _re.IGNORECASE),
     "days_to_fix", "<",  {"day": 1, "week": 7, "month": 30}),
    # "less than N days to fix/resolve/close"
    (_re.compile(r'\bless\s+than\s+(\d+)\s*(day|week|month)s?\s+to\s+(?:fix|resolve|close)\b', _re.IGNORECASE),
     "days_to_fix", "<",  {"day": 1, "week": 7, "month": 30}),
    # "resolved/fixed/closed within N days/weeks/months"
    (_re.compile(r'\b(?:resolved?|fixed?|closed?)\s+within\s+(\d+)\s*(day|week|month)s?\b', _re.IGNORECASE),
     "days_to_fix", "<=", {"day": 1, "week": 7, "month": 30}),
    # "took at most N days/weeks/months"
    (_re.compile(r'\btook?\s+at\s+most\s+(\d+)\s*(day|week|month)s?\b', _re.IGNORECASE),
     "days_to_fix", "<=", {"day": 1, "week": 7, "month": 30}),
    # ── Add custom post-filter patterns below ─────────────────────────
]

import asyncio
import base64
import logging
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
from dotenv import load_dotenv, set_key
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from requests_oauthlib import OAuth2Session

from config import Profile, get_profile
from config_fields import FIELD_META, DEFAULT_DISPLAY_FIELDS, COMPUTED_FIELD_DEPS, POST_FILTER_FETCH_MULTIPLIER
from settings import (
    ROVO_MCP_URL,
    OAUTH_REDIRECT_URI as REDIRECT_URI,
    OAUTH_SCOPES       as SCOPES,
    OAUTH_AUTH_URL     as AUTH_URL,
    OAUTH_TOKEN_URL    as TOKEN_URL,
    OAUTH_ENV_FILE     as ENV_FILE,
    DEFAULT_JQL,
    MAX_RESULTS,
)

load_dotenv()

logger = logging.getLogger(__name__)


# ── OAuth 2.1 ───────────────────────────────────────────────────────

def _wait_for_callback() -> str:
    callback_url = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            callback_url["url"] = f"http://localhost:3334{self.path}"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Authorization successful! You can close this tab.")

        def log_message(self, *args):
            pass

    HTTPServer(("localhost", 3334), Handler).handle_request()
    return callback_url["url"]


def get_oauth_token(profile: Profile) -> str:
    cached = os.getenv("ATLASSIAN_OAUTH_TOKEN")
    if cached:
        return cached

    if not profile.client_id or not profile.client_secret:
        raise RuntimeError(
            "client_id and client_secret must be set in profiles.json (or env) for OAuth 2.1."
        )

    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    session = OAuth2Session(profile.client_id, scope=SCOPES, redirect_uri=REDIRECT_URI)
    auth_url, _ = session.authorization_url(AUTH_URL, audience="api.atlassian.com")

    logger.info("Opening browser for Atlassian authorization...")
    logger.info("If the browser does not open, visit:\n  %s", auth_url)
    webbrowser.open(auth_url)

    token = session.fetch_token(
        TOKEN_URL,
        authorization_response=_wait_for_callback(),
        client_secret=profile.client_secret,
    )
    access_token = token["access_token"]
    set_key(ENV_FILE, "ATLASSIAN_OAUTH_TOKEN", access_token)
    logger.info("OAuth token obtained and cached in .env")
    return access_token


# ── MCP transports ──────────────────────────────────────────────────

def _transport_bearer(token: str) -> StreamableHttpTransport:
    return StreamableHttpTransport(
        url=ROVO_MCP_URL,
        headers={"Authorization": f"Bearer {token}"},
    )


def _transport_basic(profile: Profile) -> StreamableHttpTransport:
    credentials = base64.b64encode(
        f"{profile.email}:{profile.token}".encode()
    ).decode()
    return StreamableHttpTransport(
        url=ROVO_MCP_URL,
        headers={"Authorization": f"Basic {credentials}"},
    )


# ── MCP health check ────────────────────────────────────────────────

async def check_rovo_mcp(profile: Profile, bearer_token: str | None = None):
    transport   = _transport_bearer(bearer_token) if bearer_token else _transport_basic(profile)
    auth_method = "OAuth 2.1" if bearer_token else "Basic Auth"

    logger.info("Checking Rovo MCP server (%s) for profile '%s'...", auth_method, profile.name)
    try:
        async with Client(transport) as client:
            tools = await client.list_tools()
            logger.info("Server is UP - %d tool(s) available:", len(tools))
            for t in tools:
                logger.info("  - %s", t.name)
    except Exception as e:
        logger.warning("Server is UNREACHABLE: %s", e)


# ── Jira REST API ───────────────────────────────────────────────────

async def get_cloud_id(profile: Profile) -> str:
    async with httpx.AsyncClient(auth=(profile.email, profile.token)) as client:
        r = await client.get(f"{profile.jira_base_url}/_edge/tenant_info")
        r.raise_for_status()
        return r.json()["cloudId"]


# Field config loaded from config_fields.py
_FIELD_META     = FIELD_META
_DEFAULT_FIELDS = DEFAULT_DISPLAY_FIELDS


def _apply_post_filters(issues: list, filters: list, limit: int) -> tuple[list, int]:
    """Apply post-filters to issues, return (passing_issues, examined_count)."""
    _OPS = {
        ">"  : lambda a, b: a >  b,
        ">=" : lambda a, b: a >= b,
        "<"  : lambda a, b: a <  b,
        "<=" : lambda a, b: a <= b,
    }
    passing = []
    for issue in issues:
        ok = True
        for pf in filters:
            if pf.field not in _FIELD_META:
                continue
            _, _, extractor = _FIELD_META[pf.field]
            try:
                value = int(extractor(issue["fields"]))
            except (ValueError, TypeError):
                ok = False
                break
            if not _OPS[pf.operator](value, pf.threshold):
                ok = False
                break
        if ok:
            passing.append(issue)
        if len(passing) >= limit:
            break
    return passing[:limit], len(issues)


async def validate_jql(client: httpx.AsyncClient, profile: Profile, jql: str) -> str | None:
    """Validate JQL against Jira's parse endpoint.

    Returns the first error message if invalid, None if the JQL is valid.
    Validation is skipped (returns None) if the parse endpoint itself fails.
    """
    response = await client.post(
        f"{profile.jira_base_url}/rest/api/2/jql/parse",
        json={"queries": [jql]},
    )
    if not response.is_success:
        logger.warning("JQL validation endpoint unavailable (HTTP %d) — skipping", response.status_code)
        return None
    errors = response.json().get("queries", [{}])[0].get("errors", [])
    return errors[0] if errors else None


async def atlasmind(
    profile:      Profile,
    jql_query:    str        = DEFAULT_JQL,
    max_results:  int        = MAX_RESULTS,
    fields:       list[str] | None = None,
    post_filters: list       = None,
):
    # Build the ordered field list: dynamic fields (excluding summary) + summary last
    display_fields = fields if fields is not None else _DEFAULT_FIELDS
    # Ensure only known fields, cap at 9 extra (+ summary = 10 total)
    display_fields = [f for f in display_fields if f in _FIELD_META][:9]
    # Computed fields are virtual — exclude from API request, add their dependencies instead
    extra_api = []
    for field in display_fields:
        for dep in COMPUTED_FIELD_DEPS.get(field, []):
            if dep not in display_fields and dep not in extra_api:
                extra_api.append(dep)
    computed  = set(COMPUTED_FIELD_DEPS.keys())
    api_fields = [f for f in display_fields if f not in computed] + extra_api + ["summary"]

    post_filters = post_filters or []
    fetch_limit  = max_results * POST_FILTER_FETCH_MULTIPLIER if post_filters else max_results

    auth    = (profile.email, profile.token) if profile.email and profile.token else None
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    async with httpx.AsyncClient(auth=auth, headers=headers) as client:
        error = await validate_jql(client, profile, jql_query)
        if error:
            logger.error("Invalid JQL: %s", error)
            logger.error("JQL was: %s", jql_query)
            return

        if profile.is_cloud:
            response = await client.post(
                f"{profile.jira_base_url}/rest/api/3/search/jql",
                json={"jql": jql_query, "maxResults": fetch_limit, "fields": api_fields},
            )
        else:
            response = await client.get(
                f"{profile.jira_base_url}/rest/api/2/search",
                params={"jql": jql_query, "maxResults": fetch_limit, "fields": ",".join(api_fields)},
            )

        if not response.is_success:
            logger.error("HTTP %d: %s", response.status_code, response.text[:200])
        response.raise_for_status()
        data = response.json()

    issues = data.get("issues", [])
    total  = data.get("total", 0)

    examined = len(issues)
    if post_filters:
        issues, examined = _apply_post_filters(issues, post_filters, max_results)

    # ── Build column specs ──────────────────────────────────────────
    cols = [("Key", 12, lambda issue: issue["key"])]
    for field_name in display_fields:
        label, width, extractor = _FIELD_META[field_name]
        cols.append((label, width, lambda issue, ex=extractor: ex(issue["fields"])))
    # Summary always last, fills remaining width
    cols.append(("Summary", 0, lambda issue: issue["fields"].get("summary", "")))

    # ── Header ──────────────────────────────────────────────────────
    header = "".join(label.ljust(width) for label, width, _ in cols[:-1]) + cols[-1][0]
    total_width = max(80, len(header) + 10)

    print(f"\nProfile : {profile.name}  ({profile.jira_base_url})")
    print(f"JQL     : {jql_query}")
    if post_filters:
        descs = [f"{pf.field} {pf.operator} {pf.threshold}" for pf in post_filters]
        print(f"Filter  : {' AND '.join(descs)}  (applied in Python after fetch)")
    print(f"Found {total} issue(s) in Jira, showing {len(issues)} (examined {examined}):\n")
    print(header)
    print("-" * total_width)

    for issue in issues:
        row = "".join(str(fn(issue))[:width].ljust(width) for _, width, fn in cols[:-1])
        row += str(cols[-1][2](issue))
        print(row)

    print("-" * total_width)
    if post_filters:
        print(f"Retrieved {len(issues)} matching issue(s) (examined {examined} of {total} total; post-filter applied).")
    else:
        print(f"Retrieved {len(issues)} of {total} total issue(s).")


if __name__ == "__main__":
    logging.basicConfig(
        stream=__import__("sys").stdout,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8",
    )
    profile = get_profile()
    asyncio.run(check_rovo_mcp(profile))
    asyncio.run(atlasmind(profile))

"""
JQL Generator — converts natural language to JQL.

Backends:
  local  — Ollama local model (default: qwen2.5-coder:7b-instruct)
  rovo   — Atlassian Rovo MCP server (requires OAuth 2.1 Bearer token)

Select via JQL_BACKEND env var or the --backend CLI flag.

Prompt engineering follows the Jira-Whisperer pattern:
  - Today's date injected for relative queries
  - Live Jira metadata (projects, statuses, priorities) fetched and injected
  - Full JQL reference documentation included in prompt
  - 4-stage JQL validation pipeline runs after generation
  - Zero-results retry with progressive constraint relaxation
"""

import asyncio
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime

import httpx
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from jql_reference import JQL_REFERENCE
from jql_examples import ExampleStore, find_examples, example_count

# Condensed reference for smaller/local models — covers the most common mistakes
_JQL_QUICK_REFERENCE = """
JQL quick rules:
- Values with spaces MUST be quoted: status in (Open, "In Progress", "In Review")
- LIMIT / GROUP BY / HAVING / COUNT() do NOT exist in JQL
- maxResults is an API param, NOT JQL — never put it in the query string
- Use statusCategory != Done for "open/unresolved" issues
- Date offsets: created >= "-7d"  |  created >= startOfWeek()  |  created >= "2024-01-01"
- startOfQuarter() is Cloud-only — use absolute date literals on Server
- ORDER BY: created, updated, priority, status, key, duedate  (ASC or DESC)
- currentUser() for logged-in user  |  assignee is EMPTY for unassigned
- DURING is only valid after CHANGED/WAS — never use as standalone date range
- issueFunction / resolvedIssuesOf / linkedIssuesOf etc. are ScriptRunner plugin functions — NEVER use them in standard JQL
- To filter by priority: priority in (Minor, Major) — no plugin functions needed
- If no project is specified, do NOT add any project filter at all
- Date arithmetic between two fields is NOT supported: `resolved <= created + 10d` is INVALID
- JQL cannot compute "time to resolve" — use absolute date offsets on a single field only: created >= "-10d"
- For "closed issues": resolution != Unresolved AND resolved is not EMPTY
- The field name is 'resolved' NOT 'closed' — 'closed' does not exist in JQL
- Valid date fields: created, updated, resolved, duedate, lastViewed
"""
from jql_validator import JiraMetadata, fetch_metadata, validate_and_fix, relax_and_retry

load_dotenv()


# ── Base class ───────────────────────────────────────────────────────

class JQLGenerator(ABC):
    validate: bool = True   # set False to skip jql_validator post-processing

    @abstractmethod
    async def generate(self, natural_language: str, profile=None) -> str:
        """Return a JQL string for the given natural language query."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify the backend is reachable. Prints status and returns True/False."""


# ── Prompt builder ───────────────────────────────────────────────────

_MAX_PROJECTS_IN_PROMPT = 20


def _build_prompt(
    natural_language: str,
    metadata: JiraMetadata | None,
    full_reference: bool = False,
    example_store: ExampleStore | None = None,
) -> str:
    today = datetime.now().strftime("%Y-%m-%d")

    if metadata and metadata.project_names:
        # If the user mentions a known project key, show only that one.
        # Otherwise cap to avoid bloating the prompt on large instances.
        query_upper = natural_language.upper()
        mentioned   = {k: v for k, v in metadata.project_names.items() if k in query_upper}
        projects    = mentioned if mentioned else dict(
            list(metadata.project_names.items())[:_MAX_PROJECTS_IN_PROMPT]
        )
        truncated      = len(metadata.project_names) > _MAX_PROJECTS_IN_PROMPT and not mentioned
        projects_block = "\n".join(f"  {k}: {v}" for k, v in projects.items())
        if truncated:
            projects_block += f"\n  ... (showing {_MAX_PROJECTS_IN_PROMPT} of {len(metadata.project_names)})"
        default_project_hint = "If the user does not mention a project, do NOT add a project filter."
    else:
        projects_block       = "  (unavailable)"
        default_project_hint = "Do NOT assume any project key."

    statuses_block   = ", ".join(metadata.statuses)   if metadata and metadata.statuses   else "(unavailable)"
    priorities_block = ", ".join(metadata.priorities) if metadata and metadata.priorities else "(unavailable)"

    reference = JQL_REFERENCE if full_reference else _JQL_QUICK_REFERENCE

    # Retrieve relevant few-shot examples from the configured dataset
    store    = example_store or ExampleStore()
    examples = store.find(natural_language, n=5)
    if examples:
        examples_block = "\n".join(
            f'  Q: "{desc}"\n  JQL: {jql}' for desc, jql in examples
        )
        examples_section = f"\nRelevant JQL examples (from verified Apache Jira queries):\n{examples_block}\n"
    else:
        examples_section = ""

    return f"""You are a Jira API expert.
Today's date is {today}.
{default_project_hint}

Convert this user question into a valid JQL query string.
Return ONLY the raw JQL string — no explanation, no markdown, no code fences.

User question: "{natural_language}"

Available Jira projects (key: name):
{projects_block}

Available statuses : {statuses_block}
Available priorities: {priorities_block}
{examples_section}
{reference}
"""


# ── Local Ollama backend ─────────────────────────────────────────────

class LocalLLMGenerator(JQLGenerator):
    """Generate JQL using a local Ollama model with live Jira context injection."""

    validate = True  # run jql_validator by default

    def __init__(
        self,
        model:           str       = "qwen2.5-coder:7b-instruct",
        base_url:        str       = "http://localhost:11434",
        examples_file:   str | None = None,
        examples_folder: str | None = None,
    ):
        self.model         = model
        self.base_url      = base_url.rstrip("/")
        self.example_store = ExampleStore(path=examples_file, folder=examples_folder)

    async def health_check(self) -> bool:
        print(f"Checking Ollama ({self.base_url}, model: {self.model})...")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                r.raise_for_status()
                models = [m["name"] for m in r.json().get("models", [])]
                if self.model in models:
                    print(f"  Ollama is UP — model '{self.model}' is available.")
                    n = self.example_store.count()
                    if n:
                        print(f"  JQL example dataset: {n} queries loaded from {self.example_store.source_desc()}.")
                    else:
                        print(f"  JQL example dataset: none loaded ({self.example_store.source_desc()}).")
                    return True
                available = ", ".join(models) or "none"
                print(f"  Ollama is UP but model '{self.model}' not found. Available: {available}")
                return False
        except Exception as e:
            print(f"  Ollama is UNREACHABLE: {e}")
            return False

    async def generate(self, natural_language: str, profile=None) -> str:
        # Fetch live Jira metadata for context injection
        metadata: JiraMetadata | None = None
        if profile:
            try:
                metadata = await fetch_metadata(profile)
            except Exception as e:
                print(f"  [warn] Could not fetch Jira metadata for prompt context: {e}")

        prompt = _build_prompt(natural_language, metadata, example_store=self.example_store)

        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=180, write=10, pool=5)) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model"   : self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream"  : False,
                },
            )
            if not response.is_success:
                raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")
            jql = response.json()["message"]["content"].strip()

        jql = _clean_jql(jql)
        print(f"  [JQL generated] {jql}")

        # Validate and auto-correct (enabled by default for local LLMs)
        if self.validate and metadata:
            result = validate_and_fix(jql, metadata)
            if result.changes:
                print(f"  [JQL auto-corrected] {result.changes}")
                print(f"  [JQL final]          {result.fixed_jql}")
            jql = result.fixed_jql

        return jql


# ── Rovo MCP backend ─────────────────────────────────────────────────

class RovoMCPGenerator(JQLGenerator):
    """Generate JQL using the Atlassian Rovo MCP 'search' tool (requires OAuth 2.1)."""

    ROVO_MCP_URL = "https://mcp.atlassian.com/v1/mcp"
    validate     = False  # Rovo output is trusted — skip validator by default

    def __init__(self, bearer_token: str, validate: bool = False):
        self.bearer_token = bearer_token
        self.validate     = validate

    async def health_check(self) -> bool:
        print(f"Checking Rovo MCP server ({self.ROVO_MCP_URL})...")
        try:
            transport = StreamableHttpTransport(
                url=self.ROVO_MCP_URL,
                headers={"Authorization": f"Bearer {self.bearer_token}"},
            )
            async with Client(transport) as client:
                tools = await client.list_tools()
                print(f"  Rovo MCP is UP — {len(tools)} tool(s) available:")
                for t in tools:
                    print(f"    - {t.name}")
                return True
        except Exception as e:
            print(f"  Rovo MCP is UNREACHABLE: {e}")
            return False

    async def generate(self, natural_language: str, profile=None) -> str:
        transport = StreamableHttpTransport(
            url=self.ROVO_MCP_URL,
            headers={"Authorization": f"Bearer {self.bearer_token}"},
        )
        async with Client(transport) as client:
            result = await client.call_tool(
                "search",
                arguments={"query": natural_language},
            )
        for block in result.content:
            if hasattr(block, "text"):
                return _clean_jql(block.text.strip())
        raise RuntimeError("Rovo MCP returned no content.")


# ── Factory ──────────────────────────────────────────────────────────

def get_generator(
    backend:         str | None = None,
    examples_file:   str | None = None,
    examples_folder: str | None = None,
) -> JQLGenerator:
    """
    Return the appropriate JQLGenerator.

    backend:         'local' | 'rovo'  (defaults to JQL_BACKEND env var, then 'local')
    examples_file:   path to a single JQL examples file (.md, .json, .csv)
                     Falls back to JQL_EXAMPLES_FILE env var, then the built-in dataset.
    examples_folder: path to a folder — all .md/.json/.csv files are loaded.
                     Falls back to JQL_EXAMPLES_FOLDER env var.
                     Takes precedence over examples_file when both are given.

    Env vars:
      JQL_BACKEND           — 'local' or 'rovo'
      JQL_LOCAL_MODEL       — Ollama model name  (default: qwen2.5-coder:7b-instruct)
      JQL_OLLAMA_URL        — Ollama base URL    (default: http://localhost:11434)
      JQL_EXAMPLES_FILE     — path to custom JQL examples file
      JQL_EXAMPLES_FOLDER   — path to folder of JQL examples files
      ATLASSIAN_OAUTH_TOKEN — required for 'rovo' backend
    """
    backend         = (backend or os.getenv("JQL_BACKEND", "local")).lower()
    examples_file   = examples_file   or os.getenv("JQL_EXAMPLES_FILE")
    examples_folder = examples_folder or os.getenv("JQL_EXAMPLES_FOLDER")

    if backend == "local":
        return LocalLLMGenerator(
            model           = os.getenv("JQL_LOCAL_MODEL", "qwen2.5-coder:7b-instruct"),
            base_url        = os.getenv("JQL_OLLAMA_URL",  "http://localhost:11434"),
            examples_file   = examples_file,
            examples_folder = examples_folder,
        )

    if backend == "rovo":
        token = os.getenv("ATLASSIAN_OAUTH_TOKEN")
        if not token:
            raise RuntimeError("ATLASSIAN_OAUTH_TOKEN must be set in .env for the Rovo backend.")
        return RovoMCPGenerator(bearer_token=token)

    raise ValueError(f"Unknown backend '{backend}'. Choose 'local' or 'rovo'.")


# ── Helpers ──────────────────────────────────────────────────────────

def _clean_jql(text: str) -> str:
    """Strip markdown code fences if the model wrapped the JQL in them."""
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ── Quick test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from config import get_profile

    parser = argparse.ArgumentParser(description="Test JQL generation")
    parser.add_argument("--backend", choices=["local", "rovo"], default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--query", default="list all open issues")
    args = parser.parse_args()

    async def main():
        profile   = get_profile(args.profile)
        generator = get_generator(args.backend)
        print(f"Backend : {type(generator).__name__}")
        print(f"Profile : {profile.name}")
        print(f"Query   : {args.query}")
        jql = await generator.generate(args.query, profile=profile)
        print(f"JQL     : {jql}")

    asyncio.run(main())

# AtlasMind

AI-powered Jira query assistant. Ask questions in plain English — AtlasMind converts them to JQL using a local Ollama LLM and RAG over annotated examples, then queries Jira and returns the results.

All LLM inference runs locally (Ollama). No data leaves your machine or the Atlassian ecosystem.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.ai) running locally with a code model (e.g. `qwen2.5-coder:7b-instruct`)
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension
- A Jira instance (Atlassian Cloud or Server/Data Center)

## Setup

```bash
# Install dependencies
uv sync

# Copy and fill in your connection profiles
cp profiles.json.example profiles.json

# Set the database URL (or add to .env)
export DATABASE_URL=postgresql://user:password@localhost:5432/jql_vectordb
```

## Running

### CLI mode — single query, table output to stdout

```bash
uv run python main.py --query "list 5 open bugs in KAFKA"
uv run python main.py --query "show issues resolved in ZOOKEEPER last 30 days" --profile work
uv run python main.py --list-profiles
```

### Server mode — FastAPI HTTP server, JSON responses

```bash
uv run python main.py --server
uv run python main.py --server --host 127.0.0.1 --port 9000
```

The server loads the embedding model and annotations once at startup and reuses them across all requests.

**Endpoint:** `GET /query`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `q`       | yes      | Natural language Jira query |
| `profile` | no       | Profile name from `profiles.json` (uses default if omitted) |
| `limit`   | no       | Max results to return (overrides any hint in the query) |

```bash
curl "http://localhost:8000/query?q=list+5+bugs+in+KAFKA"
curl "http://localhost:8000/query?q=open+issues+in+HADOOP&limit=20&profile=work"
```

**Response shape:**

```json
{
  "profile": "work",
  "jira_base_url": "https://issues.apache.org/jira",
  "jql": "project = KAFKA AND issuetype = Bug ORDER BY created DESC",
  "total": 4321,
  "shown": 5,
  "examined": 5,
  "post_filters": [],
  "display_fields": ["status", "assignee", "created"],
  "issues": [
    {
      "key": "KAFKA-1234",
      "status": "Open",
      "assignee": "Jane Smith",
      "created": "2024-03-01",
      "summary": "Consumer lag spikes under high load"
    }
  ]
}
```

Interactive API docs are available at `http://localhost:8000/docs`.

## CLI reference

```
uv run python main.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--query TEXT` | `list all open issues` | Natural language query |
| `--profile NAME` | _(default profile)_ | Profile from `profiles.json` |
| `--annotation-file PATH` | `data/jira-jql-annotated-queries.md` | JQL examples file for RAG |
| `--list-profiles` | — | List configured profiles and exit |
| `--reload-db` | — | Force re-embed annotations into pgvector |
| `--server` | — | Start FastAPI server instead of running a query |
| `--host HOST` | `0.0.0.0` | Server bind host (with `--server`) |
| `--port PORT` | `8000` | Server bind port (with `--server`) |

## Architecture

```
main.py / server.py
    │
    ├── load_annotations_into_db()   embed JQL examples → pgvector
    ├── generate_jql()               NL query → similarity search → Ollama → JQL
    ├── _sanitize_jql()              strip GROUP BY / LIMIT / field arithmetic
    └── atlasmind()                  JQL → Jira REST API → structured result dict
            │
            ├── CLI  → _print_table()   renders table to stdout
            └── API  → FastAPI          serialises dict to JSON response
```

**LLM policy:** Only Ollama (local) and Atlassian Rovo MCP are permitted backends. No external cloud LLMs (OpenAI, Claude API, etc.) — data stays on-premise or within the Atlassian ecosystem.

import asyncio
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────
JIRA_BASE_URL   = "https://sbharatllm.atlassian.net"
ATLASSIAN_EMAIL = "sbharat.llm@gmail.com"
ATLASSIAN_TOKEN = os.getenv("ATLASSIAN_TOKEN")

JQL_QUERY   = "statusCategory != Done ORDER BY created DESC"
MAX_RESULTS = 10


async def atlasmind():
    auth = (ATLASSIAN_EMAIL, ATLASSIAN_TOKEN)
    headers = {"Accept": "application/json"}
    params = {
        "jql":        JQL_QUERY,
        "maxResults": MAX_RESULTS,
        "fields":     ["summary", "status", "assignee", "created"],
    }

    async with httpx.AsyncClient(auth=auth, headers={**headers, "Content-Type": "application/json"}) as client:
        response = await client.post(
            f"{JIRA_BASE_URL}/rest/api/3/search/jql",
            json=params,
        )
        if not response.is_success:
            print(f"Error {response.status_code}: {response.text}")
        response.raise_for_status()
        data = response.json()

    issues = data.get("issues", [])
    total  = data.get("total", 0)

    print(f"JQL: {JQL_QUERY}")
    print(f"Found {total} issue(s), showing {len(issues)}:\n")
    print(f"{'Key':<12} {'Status':<15} {'Assignee':<20} Summary")
    print("-" * 80)
    for issue in issues:
        key      = issue["key"]
        fields   = issue["fields"]
        summary  = fields.get("summary", "")
        status   = fields["status"]["name"]
        assignee = fields.get("assignee") or {}
        assignee = assignee.get("displayName", "Unassigned")
        print(f"{key:<12} {status:<15} {assignee:<20} {summary}")


if __name__ == "__main__":
    asyncio.run(atlasmind())

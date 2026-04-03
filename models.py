"""
models.py -- Pydantic request/response models for the AtlasMind API.
"""

from typing import Literal, Optional
from pydantic import BaseModel


class ChartSpec(BaseModel):
    type: Literal["bar", "pie", "line", "scatter"]
    x_field: str           # field to group by, e.g. "assignee", "status", "issuetype"
    y_field: str           # "count" or a numeric field like "story_points"
    title: str
    color_field: Optional[str] = None   # optional secondary grouping dimension


class QueryRequest(BaseModel):
    query: str
    profile: Optional[str] = None
    limit: Optional[int] = None


class QueryResponse(BaseModel):
    type: str              # "jql" | "general"
    profile: str
    jira_base_url: str
    answer: Optional[str] = None
    jql: Optional[str] = None
    total: int = 0
    shown: int = 0
    examined: int = 0
    post_filters: list[str] = []
    display_fields: list[str] = []
    issues: list[dict] = []
    chart_spec: Optional[ChartSpec] = None
    filters: Optional[dict[str, list[str]]] = None

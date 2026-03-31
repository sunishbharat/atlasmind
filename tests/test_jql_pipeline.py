"""
tests/test_jql_pipeline.py — Unit tests for JQL sanitization and query parsing.

Covers the three most failure-prone areas of the pipeline:
  1. Field arithmetic removal — LLM often generates invalid date arithmetic JQL.
  2. LIMIT clause extraction — LLM appends SQL-style LIMIT which Jira rejects.
  3. Post-filter detection — NL query numbers must not bleed into record limits.
"""

import sys
from pathlib import Path

# Allow importing from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import AsyncMock, MagicMock, patch

from main import _remove_field_arithmetic, _sanitize_jql, _detect_post_filters, _parse_limit, _JQL_TAG, generate_jql


# -- 1. Field arithmetic removal ---------------------------------------

class TestRemoveFieldArithmetic:
    """LLM commonly generates two forms of invalid date arithmetic; both must be stripped."""

    def test_form_a_field_plus_duration(self):
        """resolutiondate >= created + 20d — field + duration is invalid JQL."""
        jql = "project = ZOOKEEPER AND resolutiondate >= created + 20d ORDER BY created DESC"
        result = _remove_field_arithmetic(jql)
        assert "created +" not in result
        assert "resolutiondate >=" not in result
        assert "project = ZOOKEEPER" in result
        assert "ORDER BY created DESC" in result

    def test_form_b_field_minus_field(self):
        """resolutiondate - created > 20d — field arithmetic is invalid JQL."""
        jql = "project = ZOOKEEPER AND (resolutiondate - created > 20d) ORDER BY created DESC"
        result = _remove_field_arithmetic(jql)
        assert "resolutiondate - created" not in result
        assert "project = ZOOKEEPER" in result

    def test_empty_and_parentheses_cleaned_up(self):
        """AND () left behind after clause removal must be stripped."""
        jql = "status = Open AND (resolutiondate - created > 20d)"
        result = _remove_field_arithmetic(jql)
        assert "AND ()" not in result
        assert "AND (  )" not in result
        assert "status = Open" in result

    def test_clean_jql_unchanged(self):
        """Valid JQL with no arithmetic must pass through untouched."""
        jql = "project = KAFKA AND status = Open ORDER BY created DESC"
        assert _remove_field_arithmetic(jql) == jql


# -- 2. LIMIT clause extraction ----------------------------------------

class TestSanitizeJql:
    """LIMIT is a reserved SQL keyword that Jira rejects; it must be stripped and returned."""

    def test_limit_stripped_and_returned(self):
        """LIMIT 10 appended by LLM must be removed from JQL and returned as int."""
        jql = "status = Open ORDER BY created DESC LIMIT 10"
        clean, limit = _sanitize_jql(jql)
        assert "LIMIT" not in clean.upper()
        assert limit == 10

    def test_no_limit_returns_none(self):
        """JQL without LIMIT clause must return None for the limit value."""
        jql = "status = Open ORDER BY created DESC"
        clean, limit = _sanitize_jql(jql)
        assert clean == jql
        assert limit is None

    def test_field_arithmetic_and_limit_both_removed(self):
        """Both date arithmetic and LIMIT must be cleaned in a single pass."""
        jql = "project = FOO AND resolutiondate >= created + 20d ORDER BY created DESC LIMIT 5"
        clean, limit = _sanitize_jql(jql)
        assert "resolutiondate >=" not in clean
        assert "LIMIT" not in clean.upper()
        assert limit == 5
        assert "project = FOO" in clean


# -- 3. Post-filter detection and limit parsing ------------------------

class TestPostFilterAndLimit:
    """Filter-condition numbers (e.g. '20 days') must not bleed into record limits."""

    def test_days_filter_not_parsed_as_limit(self):
        """'took more than 20 days' must not produce limit=20."""
        query = "show bugs that took more than 20 days to fix"
        limit = _parse_limit(query)
        assert limit != 20, "Filter threshold misidentified as record limit"

    def test_explicit_issue_count_parsed(self):
        """'show 5 issues' must produce limit=5."""
        query = "show 5 issues resolved last month"
        assert _parse_limit(query) == 5

    def test_top_n_parsed(self):
        """'top 15 bugs' must produce limit=15."""
        query = "list top 15 bugs assigned to me"
        assert _parse_limit(query) == 15

    def test_post_filter_detected_for_days_to_fix(self):
        """'took more than 20 days' must produce a days_to_fix > 20 post-filter."""
        query = "show bugs that took more than 20 days to fix"
        filters = _detect_post_filters(query)
        assert len(filters) == 1
        pf = filters[0]
        assert pf.field == "days_to_fix"
        assert pf.operator == ">"
        assert pf.threshold == 20

    def test_week_unit_converted_to_days(self):
        """'more than 2 weeks to fix' must produce threshold=14 days."""
        query = "issues that took more than 2 weeks to fix"
        filters = _detect_post_filters(query)
        assert filters[0].threshold == 14

    def test_no_filter_when_no_duration_phrase(self):
        """Queries without time-duration phrases must produce no post-filters."""
        query = "show all open bugs in project KAFKA"
        assert _detect_post_filters(query) == []


# -- 4. Non-JQL (general) query routing --------------------------------

import asyncio

class TestNonJqlRouting:
    """When Ollama responds without the <<JQL>> prefix, generate_jql must
    return is_general=True and the caller must skip the Jira REST API."""

    def _make_client_mock(self, ollama_answer: str):
        """Build a mocked httpx.AsyncClient that returns ollama_answer from POST."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": ollama_answer}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False
        mock_client.post = AsyncMock(return_value=mock_response)
        return mock_client

    def test_general_query_returns_is_general_true(self):
        """A greeting response from Ollama (no <<JQL>> prefix) must set is_general=True."""
        ollama_answer = "Hello! I am doing well, thank you for asking."
        mock_examples = [(1, "some annotation", "project = FOO", 0.5)]

        with patch("main.test_embeddings_jql", return_value=(mock_examples, None)), \
             patch("httpx.AsyncClient", return_value=self._make_client_mock(ollama_answer)):
            text, is_general = asyncio.run(generate_jql("How are you?", model=MagicMock()))

        assert is_general is True
        assert _JQL_TAG not in text
        assert text == ollama_answer

    def test_jql_query_returns_is_general_false(self):
        """A JQL response from Ollama (with <<JQL>> prefix) must set is_general=False and strip the tag."""
        jql = "project = HADOOP AND status = Open ORDER BY created DESC"
        ollama_answer = f"{_JQL_TAG}{jql}"
        mock_examples = [(1, "some annotation", jql, 0.3)]

        with patch("main.test_embeddings_jql", return_value=(mock_examples, None)), \
             patch("httpx.AsyncClient", return_value=self._make_client_mock(ollama_answer)):
            text, is_general = asyncio.run(generate_jql("list open issues in HADOOP", model=MagicMock()))

        assert is_general is False
        assert text == jql

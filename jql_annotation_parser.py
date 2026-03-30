"""
jql_annotation_parser.py — Parser for JQL annotation files used by AtlasMind.

Reads a Markdown-formatted annotation file containing comment/JQL pairs and
returns them as a list of dicts for downstream embedding and storage in pgvector.

Supported format (block-comment style):
    /* Natural language description of the query */
    PROJECT = FOO AND status = "In Progress" ORDER BY created DESC
"""

import re
import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

def parse_jql_annotations(path: str) -> list[dict[str, str]]:
    """Parse a JQL annotation file and return comment/JQL pairs.

    Reads the file at *path* and extracts all block-comment-style annotation
    pairs in the format ``/* comment */\\nJQL``. Each pair becomes a dict with
    keys ``"comment"`` and ``"jql"``.

    Args:
        path: Filesystem path to the annotation file (UTF-8 encoded Markdown).

    Returns:
        list[dict[str, str]]: Ordered list of ``{"comment": ..., "jql": ...}``
        dicts, one per matched annotation block. Returns an empty list if no
        pairs are found.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    logger.info("File loaded: %d characters", len(text))

    pattern = re.compile(r'/\*\s*(.*?)\s*\*/[\r\n]+(?!/\*)([^\r\n]+)', re.DOTALL)
    pairs: list[dict[str, str]] = [
        {"comment": m.group(1).strip(), "jql": m.group(2).strip()}
        for m in pattern.finditer(text)
    ]

    logger.info("Parsed %d comment/JQL pairs", len(pairs))

    for p in pairs[-10:]:
        logger.debug("comment: %s", p["comment"])
        logger.debug("jql:     %s", p["jql"])

    return pairs

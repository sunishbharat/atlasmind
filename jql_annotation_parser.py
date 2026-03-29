
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

def parse_jql_annotations(path:str) -> list[dict[str, str]]:
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

"""Evaluators for TREC, E-Commerce and Injury.

The first valid label in the response must equal the gold label.
"""

import re

from . import Result

_TREC = re.compile(r"\b(ENTY|DESC|HUM|LOC|NUM)\b", re.IGNORECASE)
_ECOMMERCE = re.compile(r"(Clothing\s*&\s*Accessories|Electronics|Household|Books)", re.IGNORECASE)
_INJURY = re.compile(
    r"(Fires\s+and\s+Explosions|Contact\s+with\s+Objects|"
    r"Violence|Transportation|Falls|Exposure|Overexertion)",
    re.IGNORECASE,
)


def _result(predicted: str, gold: str) -> Result:
    return Result(predicted == gold, float(predicted == gold), parse_error=not predicted)


def trec(response, sample):
    m = _TREC.search(response)
    return _result(m.group(1).upper() if m else "", sample.gold_answer.upper())


def ecommerce(response, sample):
    m = _ECOMMERCE.search(response)
    predicted = ""
    if m:
        raw = m.group(1).lower().strip()
        canonical = {"electronics": "Electronics", "household": "Household", "books": "Books"}
        if raw in canonical:
            predicted = canonical[raw]
        elif "clothing" in raw and "accessories" in raw:
            predicted = "Clothing & Accessories"
        else:
            predicted = m.group(1)
    return _result(predicted, sample.gold_answer)


def injury(response, sample):
    m = _INJURY.search(response)
    predicted = ""
    if m:
        raw = m.group(1).lower().strip()
        if "fires" in raw and "explosions" in raw:
            predicted = "Fires and Explosions"
        elif "contact" in raw and "objects" in raw:
            predicted = "Contact with Objects"
        else:
            predicted = raw.capitalize() if raw in {
                "violence", "transportation", "falls", "exposure", "overexertion"
            } else m.group(1)
    return _result(predicted, sample.gold_answer)

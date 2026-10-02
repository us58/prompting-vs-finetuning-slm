"""Evaluators for Replace, Delete and Insert-Before.

Every edit step's expected text must appear in the answer. As in SIFo
(https://github.com/shin-ee-chen/SIFo), answers and expected texts are lowercased, stripped
of punctuation and repeated whitespace, and then matched by substring. The expected text
applies each edit at word boundaries. A step also passes if it matches the plain
str.replace() variant or the expected text stored with the sample.
"""

import json
import re
import string

from . import Result

_PUNCT = set(string.punctuation)


def _normalize(text):
    text = "".join(ch for ch in text.lower().strip() if ch not in _PUNCT)
    return re.sub(r"\s+", " ", text)


def _contains(prediction, expected):
    return _normalize(expected) in _normalize(prediction)


def _parse(response, n):
    """Extracts the answers to Instruction_1 to Instruction_n from a JSON or JSON-like
    response."""
    answers = []
    response = response.replace("assistant\n\n", "")
    response = response.replace("Here are the responses to each instruction:\n\n", "")
    try:
        if "{" in response and "}" in response:
            obj = json.loads(response[response.index("{"):response.rindex("}") + 1])
            for i in range(1, n + 1):
                for key in (f"Instruction_{i}", f"instruction_{i}"):
                    if key in obj:
                        answers.append(str(obj[key]) if obj[key] is not None else "")
                        break
                else:
                    answers.append("")
    except (json.JSONDecodeError, ValueError):
        if "Instruction_1" in response or "instruction_1" in response:
            for part in re.split(r'"?[Ii]nstruction_\d+"?\s*:', response)[1:]:
                part = re.sub(r'[,"\s]+$', "", part.strip())
                answers.append(re.sub(r'^["\s]+', "", part))
    answers += [""] * (n - len(answers))
    return answers[:n]


def _word_pattern(word, flags=0):
    return re.compile(r"\b" + re.escape(word) + r"\b", flags)


def _apply(text, op, substring):
    kind, target = op["operation"], op["target_word"]
    if kind == "delete":
        text = _word_pattern(target).sub("", text)
        return re.sub(r"  +", " ", text).strip()
    if substring:
        if kind == "replace":
            return text.replace(target, op.get("replacement_word"))
        if kind == "insert_before":
            return text.replace(target, op.get("insert_word") + " " + target)
        return text.replace(target, target + " " + op.get("insert_word"))
    pattern = _word_pattern(target, re.IGNORECASE)
    if kind == "replace":
        return pattern.sub(op.get("replacement_word"), text)
    if kind == "insert_before":
        return pattern.sub(lambda m: op.get("insert_word") + " " + m.group(0), text)
    return pattern.sub(lambda m: m.group(0) + " " + op.get("insert_word"), text)


_KINDS = {"delete", "insert_before", "insert_after", "replace"}


def _expected_texts(ctx, substring):
    context, operations = ctx.get("context"), ctx.get("operations")
    if not context or not operations:
        return None
    texts, current = [], context
    for op in operations:
        if not op or op.get("operation") not in _KINDS or "target_word" not in op:
            return None
        current = _apply(current, op, substring)
        texts.append(current)
    return texts or None


def evaluate(response, sample):
    ctx = sample.eval_context
    stored = ctx.get("expected_texts", [])
    word_boundary = _expected_texts(ctx, substring=False)
    substring = _expected_texts(ctx, substring=True)
    expected = word_boundary or stored
    if not expected:
        return Result(False, 0.0)
    candidates = [expected, substring or [], stored]
    answers = _parse(response, len(expected))
    passed = [
        any(i < len(c) and _contains(answer, c[i]) for c in candidates)
        for i, answer in enumerate(answers)
    ]
    return Result(all(passed), sum(passed) / len(expected))

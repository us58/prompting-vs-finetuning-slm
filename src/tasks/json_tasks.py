"""Evaluators for SmartHome, GitAssistant and PII.

Each response is one JSON object, which the evaluator compares to the gold object.
"""

import json
import re
from functools import cache

from . import Result


def _result(correct: bool) -> Result:
    return Result(correct, float(correct))


_UNPARSED = Result(False, 0.0, parse_error=True)


def _extract_json(text):
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def smart_home(response, sample):
    expected = sample.eval_context.get("gold_call")
    if expected is None:
        try:
            expected = json.loads(sample.gold_answer)
        except json.JSONDecodeError:
            return _result(False)
    parsed = _extract_json(response)
    if parsed is None:
        return _UNPARSED
    # Gold intent_unclear calls have no parameters, so a reason that the model adds is not graded.
    params = lambda call: {} if call.get("name") == "intent_unclear" else call.get("parameters")
    return _result(parsed.get("name") == expected.get("name") and params(parsed) == params(expected))


def pii(response, sample):
    expected_text = sample.eval_context.get("redacted_text", "")
    expected = {(e["value"], e["replacement_token"]) for e in sample.eval_context.get("entities", [])}
    parsed = _extract_json(response)
    if parsed is None:
        return _UNPARSED
    text = parsed.get("redacted_text", "")
    text_match = isinstance(text, str) and text.strip() == expected_text.strip()
    entities = parsed.get("entities", [])
    predicted = set()
    for e in entities if isinstance(entities, list) else []:
        if isinstance(e, dict) and "value" in e and "replacement_token" in e:
            v, t = e["value"], e["replacement_token"]
            if isinstance(v, str) and isinstance(t, str):
                predicted.add((v, t))
    entities_match = predicted == expected
    if text_match and not entities_match:
        # The redacted text is already exact, so accept entity values that differ only in
        # their span boundaries, for example "Dr. Aisha Patel" and "Aisha Patel" for [PERSON].
        entities_match = _entities_contain(expected, predicted)
    return _result(text_match and entities_match)


def _entities_contain(expected, predicted):
    if len(expected) != len(predicted):
        return False
    remaining = set(predicted)
    for ev, et in expected:
        for pv, pt in list(remaining):
            if et == pt and (ev in pv or pv in ev):
                remaining.discard((pv, pt))
                break
        else:
            return False
    return not remaining


# The evaluator treats these branch values of git_push and git_pull as the current branch,
# which the schema uses when the branch is omitted.
_BRANCH_SENTINELS = {".", "current"}
_UNICODE_ASCII = [("‑", "-"), ("–", "-"), ("—", "-"), (" ", " "),
                  (" ", " "), ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"')]


@cache
def _tool_defaults(task_description):
    m = re.search(r"\[.*\]", task_description, re.DOTALL)
    if not m:
        return {}
    try:
        tools = json.loads(m.group())
    except json.JSONDecodeError:
        return {}
    defaults = {}
    for tool in tools:
        func = tool.get("function", {})
        props = func.get("parameters", {}).get("properties", {})
        defaults[func.get("name", "")] = {k: v["default"] for k, v in props.items() if "default" in v}
    return defaults


def _ascii(obj):
    if isinstance(obj, str):
        for ch, rep in _UNICODE_ASCII:
            obj = obj.replace(ch, rep)
        return obj
    if isinstance(obj, dict):
        return {k: _ascii(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ascii(v) for v in obj]
    return obj


def _path(p):
    if isinstance(p, str):
        if p.endswith("/."):
            p = p[:-1]
        if p.startswith("./"):
            p = p[2:]
        if p.endswith("/"):
            p = p.rstrip("/")
    return p


def _paths(obj):
    if isinstance(obj, dict):
        return {k: [_path(p) for p in v] if k == "files" and isinstance(v, list) else _paths(v)
                for k, v in obj.items()}
    return obj


def _strip_defaults(call, defaults):
    name = call.get("name", "")
    params = call.get("parameters", {})
    if not isinstance(params, dict):
        return {"name": name, "parameters": params}
    func_defaults = defaults.get(name, {})
    kept = {}
    for k, v in params.items():
        if k in func_defaults and v == func_defaults[k]:
            continue
        if (name in {"git_push", "git_pull"} and k == "branch" and isinstance(v, str)
                and v.strip().lower() in _BRANCH_SENTINELS):
            continue
        kept[k] = v
    return {"name": name, "parameters": kept}


def git_assistant(response, sample):
    try:
        expected = json.loads(sample.gold_answer)
    except json.JSONDecodeError:
        return _result(False)
    s = response.strip()
    # Models sometimes copy the demonstration separator.
    if s.endswith("---"):
        s = s[:-3].strip()
    try:
        predicted = json.loads(s)
    except json.JSONDecodeError:
        return _UNPARSED
    if isinstance(predicted, list):
        if len(predicted) == 1 and isinstance(predicted[0], dict):
            predicted = predicted[0]
        else:
            return _UNPARSED
    if not isinstance(predicted, dict) or not isinstance(predicted.get("name", ""), str):
        return _UNPARSED
    defaults = _tool_defaults(sample.task_description)
    norm = lambda call: _paths(_strip_defaults(_ascii(call), defaults))
    return _result(norm(predicted) == norm(expected))

"""Evaluator for ACEBench Hard and Medium.

It parses calls of the form `[Api(k=v, ...), ...]`, or a JSON list of calls, and checks them
against the gold calls. The checker follows ACEBench (https://github.com/ACEBench/ACEBench),
Copyright (c) Microsoft Corporation, MIT License (licenses/ACEBench-MIT.txt). It keeps the
subset of ACEBench's parameter types that covers these two tasks.
"""

import ast
import json
import re
from collections import Counter

from . import Result


def _bracket_content(text):
    start = text.find("[")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return ""


def _split_top_level(s, opening, closing):
    """Split on commas that are not nested inside any of the given brackets."""
    parts, current, depth = [], "", 0
    for ch in s:
        if ch in opening:
            depth += 1
        elif ch in closing:
            depth -= 1
        elif ch == "," and depth == 0:
            if current.strip():
                parts.append(current.strip())
            current = ""
            continue
        current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def _literal(value_str):
    for candidate in (value_str, re.sub(r"(\w)'(\w)", r"\1\\'\2", value_str)):
        try:
            value = ast.literal_eval(candidate)
            return list(value) if isinstance(value, set) else value
        except (ValueError, SyntaxError):
            continue
    return value_str.strip("'\"")


def _parse_param(param):
    depth = 0
    for i, ch in enumerate(param):
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == "=" and depth == 0:
            return param[:i].strip(), _literal(param[i + 1:].strip())
    return None, None


def _parse_python_calls(calls_str):
    result = []
    for call in _split_top_level(calls_str, "(", ")"):
        m = re.match(r"(\w+)\((.*)\)$", call.strip(), re.DOTALL)
        if not m:
            continue
        params = {}
        try:
            for p in _split_top_level(m.group(2), "({[", ")}]"):
                key, value = _parse_param(p)
                if key:
                    params[key] = value
        except Exception:
            # Skip the call, for example when an unhashable literal raises TypeError.
            continue
        result.append({m.group(1): params})
    return result


def _is_call_list(obj):
    return type(obj) == list and all(type(x) == dict for x in obj)


def parse_calls(response):
    content = _bracket_content(response)
    if content:
        calls = _parse_python_calls(content)
        if calls:
            return calls
    candidates = re.findall(r"\[\s*\{.*?\}\s*\]", response, re.DOTALL)
    candidates.append(response.strip())
    candidates += [m.strip() for m in re.findall(r"```(?:json)?\s*(.*?)```", response, re.DOTALL)]
    for c in candidates:
        try:
            parsed = json.loads(c)
        except json.JSONDecodeError:
            continue
        if _is_call_list(parsed):
            return parsed
    return []


_TYPES = {"string": str, "integer": int, "number": int, "float": float, "boolean": bool,
          "array": list, "object": dict, "dict": dict, "list": list, "any": str}
_NESTED = {"array", "object"}


def _std(s):
    for ch, rep in [("‑", "-"), ("‐", "-"), ("–", "-"), ("—", "-"),
                    (" ", " "), (" ", " ")]:
        s = s.replace(ch, rep)
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", s).lower().replace("'", '"')


def _type_ok(value, answer, expected, nested):
    """Returns (valid, is_variable).

    is_variable is True when the gold value has a different type than the schema. The caller
    then skips the value check.
    """
    answer_type = type(answer) if answer != "" else None
    is_variable = answer_type is not None and answer_type != expected
    if value == "true":
        value = True
    if value == "false":
        value = False
    if type(value) == expected:
        if nested is None:
            return True, is_variable
        if isinstance(value, list) and not value and isinstance(answer, list) and not answer:
            return True, is_variable
        for item in answer:
            if type(item) != list or all(_type_ok(v, item, nested, None)[0] for v in value):
                return True, is_variable
        return False, False
    if answer_type is not None and (type(value) == answer_type or answer == value):
        return True, True
    return False, False


def _dict_ok(output, answer):
    if not isinstance(output, dict) or len(output) != len(answer.keys()):
        return False
    for key, value in output.items():
        if value == "true":
            value = True
        if value == "false":
            value = False
        if key not in answer:
            return False
        expected = answer[key]
        if isinstance(expected, dict):
            if not _dict_ok(value, expected):
                return False
            continue
        std_value = _std(value) if type(value) == str else value
        std_expected = _std(expected) if type(expected) == str else expected
        if str(std_expected) not in str(std_value):
            return False
    return True


def _list_of_dicts_ok(output, answer):
    if len(output) != len(answer):
        return False
    used = set()
    for expected in answer:
        match = next((j for j, d in enumerate(output) if j not in used and _dict_ok(d, expected)), None)
        if match is None:
            return False
        used.add(match)
    return True


def _list_ok(output, answer):
    out = [_std(x) if type(x) == str else x for x in output]
    exp = [_std(x) if type(x) == str else x for x in answer]
    if all(isinstance(x, (str, int, float, bool)) for x in out + exp):
        return sorted(out) == sorted(exp)
    return out == exp


def _call_ok(schema, output, answers, category):
    answer = next(iter(answers.values()))
    output_params = next(iter(output.values()))
    if output_params == {} and schema["parameters"] == {}:
        return True
    if output_params == {} or schema["parameters"] == {}:
        return False
    if answer == schema["parameters"]["properties"]:
        return True
    if answer == {}:
        return False
    name = schema["name"]
    if name not in output:
        return False
    params = output[name]
    if not isinstance(params, dict):
        return False
    props = schema["parameters"]["properties"]
    if any(p not in params for p in schema["parameters"]["required"]):
        return False
    for param, value in params.items():
        if param not in props or param not in answer:
            return False
        type_name = props[param]["type"]
        expected = _TYPES[type_name]
        nested = None
        if type_name in _NESTED:
            items = props[param].get("items", {})
            nested = _TYPES[items["type"]] if "type" in items else (
                str if "string" in type_name else dict)
        if type_name == "float" and type(value) == int:
            value = float(value)
        ok, is_variable = _type_ok(value, answer[param], expected, nested)
        if not ok:
            return False
        if is_variable:
            continue
        if expected == dict:
            ok = _dict_ok(value, answer[param])
        elif expected == list and nested == dict:
            ok = _list_of_dicts_ok(value, answer[param])
        elif expected == str:
            std_value, std_answer = _std(value), _std(answer[param])
            ok = std_value == std_answer if "agent" in category else std_answer in std_value
        elif expected == list:
            ok = _list_ok(value, answer[param])
        if not ok:
            return False
    return True


def _calls_ok(schemas, output, gold, category):
    if len(output) != len(gold):
        return False
    # ACEBench marks repeated calls of a function with a "_<n>" suffix on the gold key.
    names = list(gold)
    answers = [{re.sub(r"_\d+$", "", k): v} for k, v in gold.items()]
    out_counts = Counter(k for call in output for k in call)
    ans_counts = Counter(k for a in answers for k in a)
    if out_counts.keys() != ans_counts.keys() or any(out_counts[k] != ans_counts[k] for k in out_counts):
        return False
    for name, answer in zip(names, answers):
        schema = next((s for s in schemas if s["name"] in name), None)
        answer_name = next(iter(answer))
        if not any(next(iter(call)) == answer_name and _call_ok(schema, call, answer, category)
                   for call in output):
            return False
    return True


def evaluate(response, sample):
    calls = parse_calls(response)
    if not calls:
        return Result(False, 0.0, parse_error=True)
    ctx = sample.eval_context
    correct = _calls_ok(
        ctx.get("function_schema", []), calls, ctx.get("ground_truth", {}),
        ctx.get("category", "normal_single_turn_single_function"),
    )
    return Result(correct, float(correct))

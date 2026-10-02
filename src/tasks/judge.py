"""LLM judge for Text2SQL and Docstring.

gpt-oss-120b rates each prediction against the reference as Good or Bad.
"""

import json

from . import Result

TEXT2SQL_SYSTEM = "You are an expert SQL evaluator that assesses whether predicted SQL queries correctly answer natural language questions."
TEXT2SQL_PROMPT = """Evaluate whether the predicted SQL query correctly answers the natural language question given the schema. A reference query is provided for context, but the predicted query does NOT need to match the reference exactly.

The prediction is good if ALL of the following are true:
1. Valid SQL - Query is syntactically correct and parseable as SQLite. Note: SQLite does NOT require aliases on subqueries in FROM clauses (e.g., `SELECT * FROM (SELECT 1)` is valid).
2. Schema Compliance - All referenced tables and columns exist in the provided schema
3. Correctness - The query would return a correct and reasonable answer to the natural language question. Accept differences in: column selection (e.g. SELECT * vs specific columns), aliases, formatting, logically equivalent expressions, and extra columns beyond what was strictly asked for. Since the model generates SQL from the natural language question and schema only (without seeing actual data), accept reasonable interpretations of string literal values when the exact stored value cannot be determined from the question alone (e.g., casing variants like 'Barista' vs 'barista', or formatting like 'player_win' vs 'player win').
4. No Logical Errors - The query does not contain wrong filters, incorrect joins, wrong aggregation logic, or conditions that would produce incorrect results

The prediction is bad if the query would return incorrect results, uses wrong tables/columns, has invalid syntax, or fundamentally misinterprets the question.

## Schema and Question
{input}

## Reference SQL Query
{gold}

## Predicted SQL Query
{predicted}

## Output format
Return in json format
```json
{{
    "Analysis": "Your analysis here",
    "Answer": "Good"
}}
```
Note: The "Answer" field must be exactly "Good" or "Bad" (with quotes)."""

DOCSTRING_SYSTEM = "You are an expert Python documentation evaluator that assesses whether generated docstrings correctly document Python functions."
DOCSTRING_PROMPT = """Evaluate whether the predicted docstring correctly documents the given Python function. A reference docstring is provided for context, but the predicted docstring does NOT need to match the reference exactly.

CRITERIA FOR "Good":

1. ACCURACY (must pass)
   - Describes what the function actually does
   - All parameters in function signature are documented
   - No hallucinated parameters, exceptions, or functionality
   - Exception types match what's actually raised in code
   - Return value matches actual return type

2. COMPLETENESS (must pass)
   - Documents all parameters
   - Includes Returns section if function returns a value
   - Includes Raises section if function raises exceptions

3. FORMAT & CLARITY (flexible)
   - Different wording OK if semantically equivalent
   - Minor formatting differences acceptable
   - Can be more or less detailed than reference
   - Missing examples is OK

AUTOMATIC "Bad":
- Missing parameters that exist in function
- Documenting parameters that don't exist
- Wrong exception types
- Describes functionality not present in code
- Missing Returns/Raises sections when needed

## Python Function
{input}

## Reference Docstring
{gold}

## Predicted Docstring
{predicted}

## Output format
Return in json format
```json
{{
    "Analysis": "Your analysis here",
    "Answer": "Good"
}}
```
Note: The "Answer" field must be exactly "Good" or "Bad" (with quotes)."""


def _verdict(text):
    if not text or text.startswith("ERROR:"):
        return False
    try:
        data, _ = json.JSONDecoder().raw_decode(text, text.index("{"))
        return data.get("Answer", "").strip().lower() == "good"
    except ValueError:
        pass
    lower = text.lower()
    return '"good"' in lower or "'good'" in lower


class Judge:
    def __init__(self, system, template, gold_key):
        self.system, self.template, self.gold_key = system, template, gold_key

    def prompt(self, response, sample):
        gold = sample.eval_context.get(self.gold_key, sample.gold_answer)
        return self.template.format(input=sample.input, gold=gold, predicted=response.strip())

    def evaluate_batch(self, responses, samples):
        from .. import llm
        completions = llm.judge([self.prompt(r, s) for r, s in zip(responses, samples)], self.system)
        return [Result(v, float(v)) for v in (_verdict(c.text) for c in completions)]


text2sql = Judge(TEXT2SQL_SYSTEM, TEXT2SQL_PROMPT, "gold_sql")
docstring = Judge(DOCSTRING_SYSTEM, DOCSTRING_PROMPT, "gold_docstring")

"""Prompt format and evaluator for each of the 17 tasks.

Accuracy counts `Result.correct`. `Result.score` gives partial credit and is the GEPA metric.
It is the fraction of passed constraints or edit steps, or an F1 score.
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple


class Result(NamedTuple):
    correct: bool
    score: float
    # True when the response could not be parsed. GEPA's LLM feedback then mentions it.
    parse_error: bool = False


from . import acebench, classification, extraction, ifbench, json_tasks, prompts, sifo  # noqa: E402
from . import judge as _judge  # noqa: E402


@dataclass(frozen=True)
class Task:
    format_prompt: Callable
    evaluate: Callable | None = None
    judge: _judge.Judge | None = None

    def evaluate_batch(self, responses, samples):
        if self.judge:
            return self.judge.evaluate_batch(responses, samples)
        return [self.evaluate(r, s) for r, s in zip(responses, samples)]

    def score(self, samples, completions):
        """Result for each sample, or None where the request failed, for example because the
        prompt exceeded the context."""
        ok = [i for i, c in enumerate(completions) if not c.error]
        results = self.evaluate_batch([completions[i].text for i in ok], [samples[i] for i in ok])
        out = [None] * len(samples)
        for i, r in zip(ok, results):
            out[i] = r
        return out


_io = prompts.labeled("Input: ", "\n\nOutput:\n", "")

TASKS = {
    "acebench_hard": Task(prompts.acebench, acebench.evaluate),
    "acebench_medium": Task(prompts.acebench, acebench.evaluate),
    "smart_home": Task(_io, json_tasks.smart_home),
    "git_assistant": Task(prompts.labeled("Request: ", "\nResponse: ", "Request: "), json_tasks.git_assistant),
    "trec": Task(prompts.labeled("Question: ", "\nAnswer: ", "Question: "), classification.trec),
    "ecommerce": Task(prompts.labeled("Product: ", "\nCategory: ", "Product: "), classification.ecommerce),
    "injury": Task(prompts.labeled("Narrative: ", "\nCategory: ", "Narrative: "), classification.injury),
    "currency": Task(_io, extraction.currency),
    "dates": Task(_io, extraction.dates),
    "emoji": Task(prompts.ifbench, ifbench.evaluate),
    "emoji_list": Task(prompts.ifbench, ifbench.evaluate),
    "replace": Task(prompts.sifo, sifo.evaluate),
    "delete": Task(prompts.sifo, sifo.evaluate),
    "insert_before": Task(prompts.sifo, sifo.evaluate),
    "text2sql": Task(_io, judge=_judge.text2sql),
    "docstring": Task(prompts.docstring, judge=_judge.docstring),
    "pii": Task(_io, json_tasks.pii),
}


def get(name: str) -> Task:
    return TASKS[name]

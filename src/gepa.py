"""GEPA prompt optimization in DSPy, with gpt-oss-120b as the reflection and feedback model.

    python -m src.gepa --model Qwen/Qwen3-0.6B-FP8
    python -m src.gepa --model Qwen/Qwen3-0.6B --lora --n 5 10 25 50 100   # LoRA + GEPA
    python -m src.gepa --model Qwen/Qwen3-0.6B-FP8 --max-full-evals 40     # GEPA (2x)

Each run optimizes on one of two folds. `best_fold` selects the fold with the higher score on
all N training samples.
"""

import argparse
import json
import os
import re
import threading
import time
from pathlib import Path

import dspy

from . import data, llm, tasks

NS = [5, 10, 25, 50, 100, 200]
SEEDS = [0, 1]
FOLDS = 2
NUM_THREADS = 25

# Each task group has an initial instruction and descriptions of the task_description, query
# and response fields.
_SIGNATURES = {
    "ifbench": (
        "Read the query carefully, identify every constraint on format, content, and style, then generate a response that satisfies all of them without missing any.",
        "Task instructions and constraints to follow",
        "User query with embedded format/content/style constraints",
        "Complete response that satisfies every constraint mentioned"),
    "acebench": (
        "Analyze the available API schemas carefully, match the user's intent to the correct functions, and fill all required parameters with appropriate values from the query.",
        "API function specifications and output format requirements",
        "User request to fulfill using the available API functions",
        "Function call(s) in the exact format specified, with no extra text"),
    "sifo": (
        "Apply each text modification instruction sequentially to the context. Each step builds on the result of the previous one. Output JSON with the modified text after each step.",
        "Sequential instruction-following task with JSON output format",
        "Context text followed by numbered modification instructions",
        'JSON object mapping each Instruction_N to the modified text after that step, e.g. {"Instruction_1": "modified text", "Instruction_2": "modified text"}'),
    "extraction": (
        "Carefully scan the entire text for all items matching the specified type and format. Output each found item on its own line in the exact format requested.",
        "Extraction task specifying what to find and required output format",
        "Text to extract items from",
        "All extracted items, one per line, in the specified format"),
    "classification": (
        "Read the input carefully and select the single most appropriate category from the allowed labels. Output only the label, nothing else.",
        "Classification task with allowed category labels",
        "Text to classify into one of the allowed categories",
        "Exactly one category label from the allowed list"),
    "pii": (
        "Identify all personally identifiable information in the text and produce a redacted version with a structured list of detected entities.",
        "PII redaction task with output format requirements",
        "Text containing personally identifiable information to redact",
        "JSON with redacted_text and entities list"),
    "docstring": (
        "Read the Python function carefully and generate a complete, accurate Google-style docstring that documents all parameters, return values, and raised exceptions.",
        "Docstring generation task with format requirements",
        "Python function to document",
        "Google-style docstring for the function"),
    "text2sql": (
        "Given a database schema and a natural language question, generate a correct SQL query that answers the question.",
        "Text-to-SQL task with schema and output requirements",
        "Database schema and natural language question",
        "SQL query answering the question"),
    "git_assistant": (
        "Given a natural language description of a desired git operation, respond with the correct git function call as a JSON object.",
        "Git assistant task with available functions and output format requirements",
        "Natural language description of the desired git operation",
        'JSON object with "name" and "parameters" keys representing the git function call'),
    "smart_home": (
        "Given a user command and conversation history, call the appropriate smart home function as a JSON object.",
        "Smart home controller task with available functions and output format requirements",
        "User command with conversation history for context",
        'JSON object with "name" and "parameters" keys representing the smart home function call'),
}
_GROUP = {
    "acebench_hard": "acebench", "acebench_medium": "acebench", "smart_home": "smart_home",
    "git_assistant": "git_assistant", "trec": "classification", "ecommerce": "classification",
    "injury": "classification", "currency": "extraction", "dates": "extraction",
    "emoji": "ifbench", "emoji_list": "ifbench", "replace": "sifo", "delete": "sifo",
    "insert_before": "sifo", "text2sql": "text2sql", "docstring": "docstring", "pii": "pii",
}


def signature(task):
    instructions, td, query, response = _SIGNATURES[_GROUP[task]]
    return dspy.Signature({
        "task_description": (str, dspy.InputField(desc=td)),
        "query": (str, dspy.InputField(desc=query)),
        "response": (str, dspy.OutputField(desc=response)),
    }, instructions)


class Program(dspy.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.predictor = dspy.Predict(signature(task))

    def forward(self, task_description, query, **kwargs):
        if _GROUP[self.task] == "sifo":
            # The SIFo description's JSON format line conflicts with DSPy's output format.
            task_description = re.sub(
                r"\s*Your output should follow this format:\s*\{[^}]*\.{3}\}", "", task_description
            ).strip()
        return self.predictor(task_description=task_description, query=query)


_FEEDBACK_PROMPT = """\
You are an expert evaluator analyzing why a model's response failed to meet requirements.

## Task Description
{task_description}

## Input
{input}

## Model's Response
{response}

## Expected Output
{gold_answer}

## Score
{score:.2f} (0.0 = completely wrong, 1.0 = perfect)
{parse_hint}
Analyze what went wrong. Provide specific, actionable feedback on how the response should be improved. Be concise (2-4 sentences)."""


def _feedback(sample, response, result):
    prompt = _FEEDBACK_PROMPT.format(
        task_description=sample.task_description, input=sample.input, response=response,
        gold_answer=sample.gold_answer or "(no gold answer)", score=result.score,
        parse_hint="\nNote: The response could not be parsed into the expected format." if result.parse_error else "",
    )
    c = llm.chat([prompt], llm.JUDGE_MODEL, llm.JUDGE_URL, max_tokens=512, extra_body=None)[0]
    if c.error:
        raise RuntimeError(c.text)
    return c.text.strip()


def metric(task):
    t = tasks.get(task)

    def score(gold, pred, trace=None, pred_name=None, pred_trace=None):
        result = t.evaluate_batch([pred.response], [gold.sample])[0]
        if pred_name is None:
            return result.score
        if result.correct:
            feedback = f"All checks passed. Score: {result.score:.2f}"
        else:
            feedback = _feedback(gold.sample, pred.response, result)
        return dspy.Prediction(score=result.score, feedback=feedback)

    return score


class TextLM(dspy.LM):
    """Returns plain text when the server also sends reasoning content, and counts tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens = {"input_tokens": 0, "output_tokens": 0}
        self._lock = threading.Lock()  # DSPy calls the LM from many threads

    def __call__(self, *args, **kwargs):
        return [o.get("text", str(o)) if isinstance(o, dict) else o
                for o in super().__call__(*args, **kwargs)]

    def update_history(self, entry):
        # Tokens are counted here because DSPy keeps only the last 10,000 history entries.
        usage = entry.get("usage") or {}
        with self._lock:
            self.tokens["input_tokens"] += usage.get("prompt_tokens", 0) or 0
            self.tokens["output_tokens"] += usage.get("completion_tokens", 0) or 0
        super().update_history(entry)


def _lm(model, base_url, **kwargs):
    return TextLM(f"openai/{model}", api_base=base_url, api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
                  cache=False, **kwargs)


def _tokens(lm, start):
    return {k: v - start[k] for k, v in lm.tokens.items()}


def run(model, task, n, seed, fold, max_full_evals=20):
    target = _lm(model, llm.TARGET_URL, temperature=0.0, max_tokens=llm.MAX_TOKENS,
                 extra_body=llm.NO_THINKING)
    reflection = _lm(llm.JUDGE_MODEL, llm.JUDGE_URL, temperature=1.0, max_tokens=8192,
                     extra_body={"chat_template_kwargs": {"reasoning_effort": "medium"}})
    dspy.configure(lm=target)

    example = lambda s: dspy.Example(sample=s, task_description=s.task_description,
                                     query=s.input).with_inputs("task_description", "query")
    train, val = data.kfold(task, n, seed, FOLDS, fold)
    trainset, valset = [example(s) for s in train], [example(s) for s in val]
    testset = [example(s) for s in data.test_set(task)]
    m = metric(task)

    def evaluate(program, examples):
        start = dict(target.tokens)
        r = dspy.Evaluate(devset=examples, metric=m, num_threads=NUM_THREADS,
                          display_progress=True, max_errors=10**6)(program)
        n_correct = sum(score >= 1.0 - 1e-4 for _, _, score in r.results)
        return {"accuracy": n_correct / len(r.results), "mean_score": r.score}, _tokens(target, start)

    start_target, start_reflection = dict(target.tokens), dict(reflection.tokens)
    optimized = dspy.GEPA(
        metric=m, reflection_lm=reflection, max_full_evals=max_full_evals,
        reflection_minibatch_size=3, candidate_selection_strategy="pareto",
        skip_perfect_score=True, use_merge=False, track_stats=True, num_threads=NUM_THREADS,
    ).compile(Program(task), trainset=trainset, valset=valset or None)
    compute = {"optimization": {"inference_model": _tokens(target, start_target),
                                "reflection_model": _tokens(reflection, start_reflection)}}
    val_scores, compute["val_eval"] = evaluate(optimized, valset or trainset)
    test, compute["test_eval"] = evaluate(optimized, testset)
    all_train, compute["all_train_eval"] = evaluate(optimized, trainset + valset)
    return {
        "model": model, "task": task, "n": n, "seed": seed, "fold": fold,
        "max_full_evals": max_full_evals, "n_train": len(train), "n_val": len(val),
        "instructions": optimized.predictor.signature.instructions,
        "total_metric_calls": optimized.detailed_results.total_metric_calls,
        "val": val_scores, "test": test, "all_train": all_train, "compute": compute,
    }


def best_fold(runs):
    """The fold with the highest mean score on all N training samples, fold 1 on a tie."""
    return max(runs, key=lambda r: (r["all_train"]["mean_score"], r["fold"]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="served model name, for example Qwen/Qwen3-0.6B-FP8")
    p.add_argument("--lora", action="store_true",
                   help="optimize on the LoRA adapter served as <task>__n<N>__s<seed>")
    p.add_argument("--max-full-evals", type=int, default=20)
    p.add_argument("--tasks", nargs="+", default=list(tasks.TASKS), choices=list(tasks.TASKS))
    p.add_argument("--n", nargs="+", type=int, default=NS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--out", type=Path, default=Path("results/gepa"))
    args = p.parse_args()
    variant = f"evals{args.max_full_evals}" + ("_lora" if args.lora else "")
    for task in args.tasks:
        for n in args.n:
            for seed in args.seeds:
                for fold in range(FOLDS):
                    path = args.out / args.model.split("/")[-1] / variant / task / f"n{n}_s{seed}_f{fold}.json"
                    if path.exists():
                        continue
                    target = f"{task}__n{n}__s{seed}" if args.lora else args.model
                    start = time.time()
                    result = run(target, task, n, seed, fold, args.max_full_evals)
                    result["seconds"] = time.time() - start
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(result, indent=1))
                    print(f"{path}: test accuracy={result['test']['accuracy']:.3f}")


if __name__ == "__main__":
    main()

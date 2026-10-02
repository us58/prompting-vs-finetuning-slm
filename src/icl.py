"""In-context learning with All-samples, CV-greedy or RAG demonstrations.

    python -m src.icl --model Qwen/Qwen3-0.6B-FP8 --method all
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from . import data, llm, tasks

NS = [5, 10, 25, 50, 100, 200]
SEEDS = [0, 1]
SHUFFLE_SEED = 0
# All-samples shuffles the draw within these bands, so the demonstrations for N are a
# prefix of those for any larger N. The order of the demonstrations depends on the band
# edges, so the edges 1 to 4 stay even though NS starts at 5.
BANDS = [1, 2, 3, 4, 5, 10, 25, 50, 100, 200]
CV_FOLDS = 5
CV_MAX_EXAMPLES = 30
RAG_K = 10
EMBEDDING_MODEL = "BAAI/bge-m3"


def all_samples(task, n, seed):
    pool, ordered, start = data.train_pool(task, n, seed), [], 0
    for end in [*BANDS, len(pool)]:
        end = min(end, len(pool))
        if end > start:
            band = pool[start:end]
            random.Random(f"{SHUFFLE_SEED}:{start}").shuffle(band)
            ordered += band
            start = end
    return ordered


def rag(task, n, seed, test, embed):
    """For each test query, the RAG_K training samples with the most similar inputs, most
    similar first."""
    pool = data.train_pool(task, n, seed)
    sims = embed([s.input for s in test]) @ embed([s.input for s in pool]).T
    return [[pool[j] for j in np.argsort(-row, kind="stable")[:RAG_K]] for row in sims]


def cv_greedy(task, n, seed, generate):
    """Runs greedy forward selection on each of the 5 folds and ranks the selected samples by
    their summed marginal gain.

    Returns (examples, selection usage). A candidate's score is its validation accuracy, with
    failed requests counted as wrong.
    """
    t = tasks.get(task)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}

    def accuracies(example_sets, val):
        prompts = [t.format_prompt(v, ex) for ex in example_sets for v in val]
        completions = generate(prompts)
        usage["requests"] += len(completions)
        usage["prompt_tokens"] += sum(c.prompt_tokens for c in completions)
        usage["completion_tokens"] += sum(c.completion_tokens for c in completions)
        results = t.score(val * len(example_sets), completions)
        return [sum(bool(r and r.correct) for r in results[i:i + len(val)]) / len(val)
                for i in range(0, len(results), len(val))]

    gains = {}
    for fold in range(CV_FOLDS):
        candidates, val = data.kfold(task, n, seed, CV_FOLDS, fold)
        random.Random(SHUFFLE_SEED).shuffle(candidates)
        current = accuracies([[]], val)[0]
        selected = []
        while len(selected) < CV_MAX_EXAMPLES:
            remaining = [c for c in candidates if c not in selected]
            if not remaining:
                break
            scores = accuracies([selected + [c] for c in remaining], val)
            best = max(range(len(scores)), key=scores.__getitem__)
            if scores[best] <= current:
                break
            selected.append(remaining[best])
            gains[remaining[best].id] = gains.get(remaining[best].id, 0.0) + scores[best] - current
            current = scores[best]
    by_id = {s.id: s for s in data.train_pool(task, n, seed)}
    ranked = sorted(gains, key=gains.get, reverse=True)[:CV_MAX_EXAMPLES]
    return [by_id[i] for i in ranked], usage


def embedder():
    from vllm import LLM

    model = LLM(model=EMBEDDING_MODEL, runner="pooling", max_model_len=8192,
                enforce_eager=True, gpu_memory_utilization=0.9)

    def embed(texts):
        e = np.array([o.outputs.embedding for o in model.embed(texts)])
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        return e / np.where(norms == 0, 1, norms)

    return embed


def run(model, method, task, n, seed, embed=None):
    t, test = tasks.get(task), data.test_set(task)
    generate = lambda prompts: llm.chat(prompts, model)
    selection = None
    if method == "all":
        examples = [all_samples(task, n, seed)] * len(test)
    elif method == "cv_greedy":
        chosen, selection = cv_greedy(task, n, seed, generate)
        examples = [chosen] * len(test)
    else:
        examples = rag(task, n, seed, test, embed)
    completions = generate([t.format_prompt(s, ex) for s, ex in zip(test, examples)])
    results = t.score(test, completions)
    # Failed requests do not count toward accuracy. They are counted in n_errors instead.
    evaluated = [r for r in results if r is not None]
    return {
        "model": model, "method": method, "task": task, "n": n, "seed": seed,
        "accuracy": sum(r.correct for r in evaluated) / len(evaluated) if evaluated else None,
        "n_errors": len(results) - len(evaluated),
        "prompt_tokens": sum(c.prompt_tokens for c in completions),
        "completion_tokens": sum(c.completion_tokens for c in completions),
        "selection": selection,
        "samples": [
            {"id": s.id, "examples": [e.id for e in ex], "response": c.text,
             "correct": r and r.correct, "score": r and r.score,
             "prompt_tokens": c.prompt_tokens, "completion_tokens": c.completion_tokens}
            for s, ex, c, r in zip(test, examples, completions, results)
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="served model name, for example Qwen/Qwen3-0.6B-FP8")
    p.add_argument("--method", required=True, choices=["all", "cv_greedy", "rag"])
    p.add_argument("--tasks", nargs="+", default=list(tasks.TASKS), choices=list(tasks.TASKS))
    p.add_argument("--n", nargs="+", type=int, default=NS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--out", type=Path, default=Path("results/icl"))
    args = p.parse_args()
    embed = embedder() if args.method == "rag" else None
    for task in args.tasks:
        for n in args.n:
            for seed in args.seeds:
                path = args.out / args.model.split("/")[-1] / args.method / task / f"n{n}_s{seed}.json"
                if path.exists():
                    continue
                start = time.time()
                result = run(args.model, args.method, task, n, seed, embed)
                result["seconds"] = time.time() - start
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=1))
                print(f"{path}: accuracy={result['accuracy']}")


if __name__ == "__main__":
    main()

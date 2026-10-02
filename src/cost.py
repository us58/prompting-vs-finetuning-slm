"""Computes C_total(Q) = C_setup + Q * c_query from run results and throughput tables.

    python -m src.cost --results results --out results/costs.csv

The throughput measured on an A6000 converts target-model tokens into GPU hours, which cost
$1.09 each. The setup tokens of CV-greedy and GEPA use the throughput at concurrency 32 with
prefix-cache hits. The LoRA setup cost is its measured training and validation time.
Deployment queries use the throughput at concurrency 1. Reflection tokens of gpt-oss-120b are
priced at API rates, and judge and feedback tokens are not counted. Each deployment query
produces as many output tokens as the task's mean gold answer. The cost model leaves out RAG
because its per-query prompts cannot be prefix-cached.
"""

import argparse
import csv
import glob
import json
import math
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path

from . import data, gepa

GPU_USD_PER_HOUR = 1.09
REFLECTION_USD_PER_MTOK = {"input": 0.15, "output": 0.60}
SETUP_CONCURRENCY, QUERY_CONCURRENCY = 32, 1
N_TEST = data.TEST_SIZE
FAMILIES = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B-Instruct-2507"]


def family(model_dir):
    return model_dir.removesuffix("-FP8")


class Throughput:
    """End-to-end throughput tables from src.throughput, with log-space bilinear lookup."""

    def __init__(self, directory):
        self.tables = {}
        for path in glob.glob(f"{directory}/*.csv"):
            with open(path) as f:
                for r in csv.DictReader(f):
                    key = (Path(path).stem, r["cache"], int(r["concurrency"]))
                    self.tables.setdefault(key, {})[(int(r["input"]), int(r["output"]))] = float(r["e2e_tok_s"])

    def _points(self, fam, profile, cache, concurrency, peers):
        names = FAMILIES if peers else [fam]
        missing = [n for n in names if (f"{n}__{profile}", cache, concurrency) not in self.tables]
        if missing:
            # ICL and GEPA costs compare all three models on their shared measurement grid.
            raise SystemExit(f"missing throughput tables: {', '.join(f'{n}__{profile}.csv' for n in missing)}")
        tables = [self.tables[(f"{n}__{profile}", cache, concurrency)] for n in names]
        valid = set.intersection(*(set(t) for t in tables))
        return self.tables[(f"{fam}__{profile}", cache, concurrency)], valid

    def tok_per_s(self, fam, n_in, n_out, concurrency, cache, profile="base", peers=False):
        """Throughput for requests of n_in input and n_out output tokens.

        The request shape is clamped to the measured grid. With peers, it is clamped to the
        part of the grid that all three models share, so that the models are compared at the
        same operating point.
        """
        table, valid = self._points(fam, profile, cache, concurrency, peers)
        ins, outs = sorted({i for i, _ in valid}), sorted({o for _, o in valid})
        inp = max(ins[0], min(max(n_in, 1), ins[-1]))
        out = max(outs[0], min(max(n_out, 1), outs[-1]))
        i_lo, i_hi = max(v for v in ins if v <= inp), min(v for v in ins if v >= inp)
        lo = lambda o: max(v for v in outs if v <= o)
        hi = lambda o: min(v for v in outs if v >= o)
        if not all(c in valid for c in [(i_lo, lo(out)), (i_lo, hi(out)), (i_hi, lo(out)), (i_hi, hi(out))]):
            # A corner can be missing because of the KV-cache limit. Then use the longest
            # output length up to out that was measured at both neighbouring input lengths,
            # or else the shortest such length.
            both = [o for o in outs if (i_lo, o) in valid and (i_hi, o) in valid]
            out = max((o for o in both if o <= out), default=both[0] if both else outs[0])
        return self._interpolate(table, inp, out)

    @staticmethod
    def _interpolate(table, inp, out):
        ins, outs = sorted({i for i, _ in table}), sorted({o for _, o in table})
        i_lo, i_hi = max(v for v in ins if v <= inp), min(v for v in ins if v >= inp)
        o_lo, o_hi = max(v for v in outs if v <= out), min(v for v in outs if v >= out)

        def lerp(x, x0, x1, y0, y1):
            if x0 == x1:
                return y0
            return y0 + (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0)) * (y1 - y0)

        v0 = lerp(out, o_lo, o_hi, table[(i_lo, o_lo)], table[(i_lo, o_hi)])
        v1 = lerp(out, o_lo, o_hi, table[(i_hi, o_lo)], table[(i_hi, o_hi)])
        return lerp(inp, i_lo, i_hi, v0, v1)


@cache
def gold_tokens(task):
    """Mean gold-answer length on the test set, in Qwen3 tokens."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    lengths = [len(tok.encode(s.gold_answer, add_special_tokens=False)) for s in data.test_set(task)]
    return sum(lengths) / len(lengths)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def _usd(seconds):
    return seconds / 3600 * GPU_USD_PER_HOUR


def per_query(tp, fam, prompt_tokens_per_query, task, profile="base", peers=True):
    """Per-query cost in USD, without and with prefix-cache hits."""
    n_in, n_out = int(prompt_tokens_per_query), int(gold_tokens(task))
    return tuple(_usd((n_in + n_out) / tp.tok_per_s(fam, n_in, n_out, QUERY_CONCURRENCY, cache, profile, peers))
                 for cache in ("no_cache", "full_cache"))


def setup_seconds(tp, fam, n_in, n_out, n_requests):
    """GPU seconds for a batch of setup requests at the throughput of their mean shape."""
    if n_in + n_out <= 0:
        return 0.0
    n_requests = max(int(n_requests), 1)
    shape = (max(int(n_in / n_requests), 1), max(int(n_out / n_requests), 1))
    return (n_in + n_out) / tp.tok_per_s(fam, *shape, SETUP_CONCURRENCY, "full_cache", peers=True)


def _load(pattern):
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(pattern))]


def icl_rows(tp, results):
    groups = defaultdict(list)
    for r in _load(f"{results}/icl/*/*/*/*.json"):
        model = r["model"].split("/")[-1]
        if r["method"] != "rag" and r["accuracy"] is not None and family(model) in FAMILIES:
            groups[(r["method"], model, r["task"], r["n"])].append(r)
    for (method, model, task, n), runs in sorted(groups.items()):
        fam = family(model)
        setup = 0.0
        if method == "cv_greedy":
            sel = [r["selection"] for r in runs]
            setup = _usd(setup_seconds(tp, fam, mean(s["prompt_tokens"] for s in sel),
                                       mean(s["completion_tokens"] for s in sel), mean(s["requests"] for s in sel)))
        pq, pq_cached = per_query(tp, fam, mean(r["prompt_tokens"] for r in runs) / N_TEST, task)
        yield method, model, task, n, mean(r["accuracy"] for r in runs), setup, pq, pq_cached


def gepa_rows(tp, results):
    groups = defaultdict(list)
    for path in sorted(glob.glob(f"{results}/gepa/*/*/*/*.json")):
        model, variant = Path(path).parts[-4:-2]
        if variant.endswith("_lora") or family(model) not in FAMILIES:
            continue
        r = json.loads(Path(path).read_text())
        method = {20: "gepa", 40: "gepa_2x"}.get(r["max_full_evals"])
        if method:  # the paper's budgets only
            groups[(method, model, r["task"], r["n"])].append(r)
    for (method, model, task, n), runs in sorted(groups.items()):
        fam = family(model)
        setups, best = [], []
        for seed in sorted({r["seed"] for r in runs}):
            folds = [r for r in runs if r["seed"] == seed]
            if len(folds) != gepa.FOLDS:
                print(f"skipping {method} {model} {task} n={n} seed={seed}: "
                      f"{len(folds)} of {gepa.FOLDS} folds", file=sys.stderr)
                continue
            # The setup cost covers both folds. Deployment uses the fold with the best score on
            # all training samples.
            def tokens(phase, kind, who="inference_model"):
                return sum((f["compute"][phase][who] if phase == "optimization" else f["compute"][phase])[kind]
                           for f in folds)
            seconds = (setup_seconds(tp, fam, tokens("optimization", "input_tokens"),
                                     tokens("optimization", "output_tokens"),
                                     sum(f["total_metric_calls"] for f in folds))
                       + setup_seconds(tp, fam, tokens("val_eval", "input_tokens"), tokens("val_eval", "output_tokens"),
                                       sum(f["n_val"] or f["n_train"] for f in folds))
                       + setup_seconds(tp, fam, tokens("all_train_eval", "input_tokens"),
                                       tokens("all_train_eval", "output_tokens"),
                                       sum(f["n_train"] + f["n_val"] for f in folds)))
            reflection = (tokens("optimization", "input_tokens", "reflection_model") * REFLECTION_USD_PER_MTOK["input"]
                          + tokens("optimization", "output_tokens", "reflection_model")
                          * REFLECTION_USD_PER_MTOK["output"]) / 1e6
            setups.append(_usd(seconds) + reflection)
            best.append(gepa.best_fold(folds))
        if not best:
            continue
        pq, pq_cached = per_query(tp, fam, mean(b["compute"]["test_eval"]["input_tokens"] for b in best) / N_TEST, task)
        yield method, model, task, n, mean(b["test"]["accuracy"] for b in best), mean(setups), pq, pq_cached


def lora_rows(tp, results):
    groups = defaultdict(list)
    for r in _load(f"{results}/lora/*/*/*/result.json"):
        if family(r["model"].split("/")[-1]) in FAMILIES:
            groups[(r["model"].split("/")[-1], r["task"], r["n"])].append(r)
    for (model, task, n), runs in sorted(groups.items()):
        setup = mean(r["optimization_gpu_hours"] for r in runs) * GPU_USD_PER_HOUR
        prompt = mean(r["prompt_tokens"] for r in runs) / N_TEST
        accuracy = mean(r["test_accuracy"] for r in runs)
        # A merged adapter runs at base-model throughput, and an unmerged one at the
        # throughput measured with the adapter loaded.
        for method, profile in (("lora", "base"), ("lora_adapter", "lora")):
            pq, pq_cached = per_query(tp, family(model), prompt, task, profile, peers=False)
            yield method, model, task, n, accuracy, setup, pq, pq_cached


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("results/costs.csv"))
    args = p.parse_args()
    tp = Throughput(args.results / "throughput")
    rows = [row for method_rows in (icl_rows, gepa_rows, lora_rows) for row in method_rows(tp, args.results)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "model", "task", "n", "accuracy", "setup_usd", "per_query_usd", "per_query_usd_cached"])
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

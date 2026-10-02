"""Task data and the two-stage split shared by every method.

Stage 1 draws a fixed 300-sample test set with seed 42. Stage 2 draws the N training
samples from the remaining pool with the run's seed. Classification tasks use a
class-balanced round-robin draw in both stages.
"""

import hashlib
import json
import lzma
import random
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import tiktoken

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_SEED = 42
TEST_SIZE = 300
STRATIFIED = {"trec", "ecommerce", "injury"}
# load() drops samples whose task_description, input and gold_answer together have more
# cl100k tokens than this limit.
TOKEN_LIMITS = {"trec": 80, "ecommerce": 200, "injury": 150}

# load() drops ACEBench samples whose gold string values do not appear in the input after
# unifying dashes and spaces. Only the listed fields are checked, and None checks every
# string field.
_ACEBENCH_GOLD_FIELDS = {
    "acebench_medium": None,
    "acebench_hard": {"specs", "full_name", "name", "position", "phone"},
}
# load() also drops ACEBench Hard samples whose question asks for a distractor function.
_ACEBENCH_CORE_FUNCTIONS = {
    "HRSystem_createEmployee", "ITAccess_provisionAccount",
    "Benefits_enrollEmployee", "Payroll_configurePayment",
}
_ACEBENCH_DISTRACTOR_KEYWORDS = [
    "badge", "credential", "lobby", "office floor", "server room",
    "executive area", "parking", "access zone", "physical access",
    "training", "course", "onboarding course", "mandatory training", "completion deadline",
    "standing desk", "ergonomic chair", "whiteboard", "projector", "conference phone",
]


@dataclass
class Sample:
    id: str
    benchmark: str
    task_description: str
    input: str
    gold_answer: str
    eval_context: dict = field(default_factory=dict)


@cache
def load(task: str) -> tuple[Sample, ...]:
    """All samples of a task after filtering, in file order."""
    with lzma.open(DATA_DIR / f"{task}.jsonl.xz", "rt", encoding="utf-8") as f:
        samples = [Sample(**json.loads(line)) for line in f if line.strip()]
    samples = [s for s in samples if s.gold_answer]
    if task in TOKEN_LIMITS:
        enc = tiktoken.get_encoding("cl100k_base")
        limit = TOKEN_LIMITS[task]
        samples = [
            s for s in samples
            if len(enc.encode(s.task_description + s.input + s.gold_answer)) <= limit
        ]
    if task in _ACEBENCH_GOLD_FIELDS:
        fields = _ACEBENCH_GOLD_FIELDS[task]
        samples = [s for s in samples if not _gold_missing_from_input(s, fields)]
    if task == "acebench_hard":
        samples = [s for s in samples if not _asks_for_distractor(s)]
    return tuple(samples)


def _normalize_dashes(s: str) -> str:
    for ch, rep in [("‑", "-"), ("‐", "-"), ("–", "-"), ("—", "-"),
                    (" ", " "), (" ", " ")]:
        s = s.replace(ch, rep)
    return s


def _gold_missing_from_input(sample: Sample, fields: set[str] | None) -> bool:
    text = _normalize_dashes(sample.input)

    def missing(d: dict) -> bool:
        for key, value in d.items():
            if isinstance(value, dict):
                if missing(value):
                    return True
            elif isinstance(value, str) and value:
                if fields is not None and key not in fields:
                    continue
                if _normalize_dashes(value) not in text:
                    return True
        return False

    gold = sample.eval_context.get("ground_truth", {})
    return any(isinstance(p, dict) and missing(p) for p in gold.values())


def _asks_for_distractor(sample: Sample) -> bool:
    gold_funcs = set(sample.eval_context.get("ground_truth", {}))
    if not gold_funcs <= _ACEBENCH_CORE_FUNCTIONS:
        return False
    question = sample.input.lower()
    return any(kw in question for kw in _ACEBENCH_DISTRACTOR_KEYWORDS)


def _by_label(samples) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.gold_answer, []).append(s)
    return dict(sorted(groups.items()))


def _round_robin(samples, n: int, seed: int) -> list[Sample]:
    """Draws n samples from the classes in turn. Each class is shuffled independently."""
    groups = _by_label(samples)
    orders = {}
    for label, members in groups.items():
        label_hash = int(hashlib.sha256(label.encode()).hexdigest(), 16) % (2**31)
        order = list(range(len(members)))
        random.Random(seed + label_hash).shuffle(order)
        orders[label] = order
    result, pos = [], dict.fromkeys(groups, 0)
    while len(result) < n and any(pos[l] < len(orders[l]) for l in groups):
        for label in groups:
            if len(result) >= n:
                break
            if pos[label] < len(orders[label]):
                result.append(groups[label][orders[label][pos[label]]])
                pos[label] += 1
    return result


@cache
def _test_and_pool(task: str) -> tuple[tuple[Sample, ...], tuple[Sample, ...]]:
    samples = load(task)
    n_test = min(TEST_SIZE, len(samples))
    if task in STRATIFIED:
        test = _round_robin(samples, n_test, TEST_SEED)
        test_ids = {id(s) for s in test}
        pool = [s for s in samples if id(s) not in test_ids]
    else:
        order = list(range(len(samples)))
        random.Random(TEST_SEED).shuffle(order)
        test = [samples[i] for i in sorted(order[:n_test])]
        pool = [samples[i] for i in sorted(order[n_test:])]
    return tuple(test), tuple(pool)


def test_set(task: str) -> list[Sample]:
    return list(_test_and_pool(task)[0])


def train_pool(task: str, n: int, seed: int) -> list[Sample]:
    """The run's N training samples in draw order. The first k of them equal the draw for k."""
    pool = _test_and_pool(task)[1]
    if n > len(pool):
        raise ValueError(f"{task}: n={n} exceeds the pool of {len(pool)}")
    if task in STRATIFIED:
        return _round_robin(pool, n, seed)
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    return [pool[i] for i in order[:n]]


def _folds(samples: list[Sample], k: int) -> list[list[Sample]]:
    size, extra = divmod(len(samples), k)
    folds, start = [], 0
    for i in range(k):
        end = start + size + (i < extra)
        folds.append(samples[start:end])
        start = end
    return folds


def kfold(task: str, n: int, seed: int, k: int, fold: int) -> tuple[list[Sample], list[Sample]]:
    """(train, val) for one of k folds over the run's N training samples."""
    drawn = train_pool(task, n, seed)
    if task in STRATIFIED:
        groups = _by_label(drawn)
        if min(len(g) for g in groups.values()) < k:
            folds = _folds(drawn, k)
        else:
            per_class = [_folds(g, k) for g in groups.values()]
            folds = [[s for cf in per_class for s in cf[i]] for i in range(k)]
    else:
        # The folds take the drawn samples in file order, not in draw order, so they depend
        # only on which samples were drawn.
        pos = {id(s): i for i, s in enumerate(_test_and_pool(task)[1])}
        folds = _folds(sorted(drawn, key=lambda s: pos[id(s)]), k)
    train = [s for i, f in enumerate(folds) if i != fold for s in f]
    return train, folds[fold]

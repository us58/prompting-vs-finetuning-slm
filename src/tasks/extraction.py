"""Evaluators for Currency and Dates.

A response is correct when its normalized items equal the gold items as a multiset, in any
order. The score is the F1 of the two multisets.
"""

import re
from collections import Counter
from decimal import Decimal, InvalidOperation

from . import Result


def _currencies(text):
    amounts = []
    for match in re.findall(r"\$[\d,]+\.\d{2}", text):
        try:
            amounts.append(Decimal(match.replace("$", "").replace(",", "")))
        except InvalidOperation:
            continue
    return amounts


def _dates(text):
    return re.findall(r"\d{4}-\d{2}-\d{2}", text)


def _multiset_f1(parse):
    def evaluate(response, sample):
        expected_text = sample.eval_context.get("expected_output", sample.gold_answer)
        if not expected_text:
            return Result(False, 0.0)
        expected, extracted = parse(expected_text), parse(response)
        tp = sum((Counter(expected) & Counter(extracted)).values())
        precision = tp / len(extracted) if extracted else 0.0
        recall = tp / len(expected) if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return Result(Counter(expected) == Counter(extracted), f1)

    return evaluate


currency = _multiset_f1(_currencies)
dates = _multiset_f1(_dates)

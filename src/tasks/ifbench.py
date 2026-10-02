"""IFBench constraint checkers with loose evaluation for Emoji and Emoji+List.

The checkers and the sentence splitter are adapted from IFBench
(https://github.com/allenai/IFBench), Copyright 2025 Allen Institute for AI,
Apache License 2.0 (licenses/IFBench-Apache-2.0.txt). The emoji checker adds a per-line
fallback for list answers, and the list checker requires two separator lines with content.
"""

import re
import string

import emoji

from . import Result

_PUNCT = str.maketrans("", "", string.punctuation)


def _variations(response):
    """The 8 loose-evaluation variants: the response and its versions without the first line,
    the last line or both, each with and without markdown asterisks."""
    lines = response.split("\n")
    first = "\n".join(lines[1:]).strip()
    last = "\n".join(lines[:-1]).strip()
    both = "\n".join(lines[1:-1]).strip()
    return [response, response.replace("*", ""), first, last, both,
            first.replace("*", ""), last.replace("*", ""), both.replace("*", "")]


_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def _split_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(_MULTIPLE_DOTS, lambda m: "<prd>" * len(m.group(0)) + "<stop>", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
                  "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>").replace("?", "?<stop>").replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = [s.strip() for s in text.split("<stop>")]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def _ends_with_emoji(stripped):
    second_last = stripped[-2] if len(stripped) > 1 else stripped[-1]
    return emoji.is_emoji(stripped[-1]) or emoji.is_emoji(second_last)


def _emoji_every_sentence(value):
    sentences = _split_sentences(value)
    for i, sentence in enumerate(sentences):
        stripped = sentence.translate(_PUNCT).strip()
        if not stripped:
            return False
        if not _ends_with_emoji(stripped):
            # The splitter may put the emoji at the start of the next sentence.
            if i == len(sentences) - 1:
                return False
            nxt = sentences[i + 1].translate(_PUNCT).strip()
            if not nxt or not emoji.is_emoji(nxt[0]):
                return False
    return True


def _emoji_every_line(value):
    lines = [l.strip() for l in value.split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    for line in lines:
        stripped = line.translate(_PUNCT).strip()
        if len(stripped) < 3:
            continue
        if not _ends_with_emoji(stripped):
            return False
    return True


def _emoji_check(value, kwargs):
    return _emoji_every_sentence(value) or _emoji_every_line(value)


def _list_check(value, kwargs):
    marker = re.escape(kwargs["sep"])
    content_lines = 0
    for line in value.split("\n"):
        if re.search(marker, line) and len(re.sub(marker, "", line).strip()) >= 3:
            content_lines += 1
    return content_lines >= 2


_CHECKERS = {"format:emoji": _emoji_check, "format:list": _list_check}


def evaluate(response, sample):
    ctx = sample.eval_context
    follows = []
    for i, instruction in enumerate(ctx.get("instruction_id_list", [])):
        kwargs = ctx["kwargs"][i] if i < len(ctx.get("kwargs", [])) else {}
        check = _CHECKERS[instruction]
        follows.append(any(v.strip() and check(v, kwargs) for v in _variations(response)))
    if not follows:
        return Result(False, 0.0)
    return Result(all(follows), sum(follows) / len(follows))

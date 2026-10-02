"""Measures the throughput of one served model on one GPU for the cost model.

    python -m src.throughput --model Qwen/Qwen3-0.6B-FP8 --name Qwen3-0.6B__base \\
        --max-context 131072 --max-kv-tokens 379088

The benchmark measures end-to-end throughput, which is input plus output tokens per second
of batch wall time, and the time to first token as the mean over repetitions of each batch's
median. The grid spans input length, output length and concurrency, and every cell is
measured with and without prefix-cache hits. Cells that exceed the context or the server's
KV-cache size, which vLLM prints at startup, are skipped. The results go to
results/throughput/<name>.csv.
"""

import argparse
import asyncio
import csv
import os
import random
import statistics
import time
import uuid
from pathlib import Path

from openai import AsyncOpenAI

from . import llm

INPUT_LENGTHS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                 49152, 65536, 81920, 98304, 114688]
OUTPUT_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
CONCURRENCY = [1, 32]
CACHE_CONDITIONS = ["no_cache", "full_cache"]
N_WARMUP, N_REPS = 2, 10
TOKENIZER = "Qwen/Qwen3-0.6B"  # All Qwen3 models share this tokenizer.
CORPUS = Path("results/throughput/corpus.txt")


def download_corpus(path):
    from datasets import load_dataset
    texts, chars = [], 0
    for article in load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True):
        texts.append(article["text"])
        chars += len(article["text"])
        if chars >= 10_000_000:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(texts), encoding="utf-8")


class Prompts:
    """Prompts cut from Wikipedia text to an exact token length."""

    def __init__(self, tokenizer, corpus, rng):
        self.tok, self.corpus, self.rng = tokenizer, corpus, rng

    def span_ids(self, n):
        for multiplier in (8, 16, 32):
            start = self.rng.randint(0, max(0, len(self.corpus) - n * multiplier))
            ids = self.tok.encode(self.corpus[start:start + n * multiplier], add_special_tokens=False)
            if len(ids) >= n:
                return ids[:n]
        raise ValueError(f"corpus too short for {n} tokens")

    def span(self, n):
        return self.tok.decode(self.span_ids(n), skip_special_tokens=True)

    def no_cache(self, n, count):
        """Prompts that start with a unique UUID, so that they get no prefix-cache hits."""
        prompts = []
        for _ in range(count):
            prefix = self.tok.encode(f"[ID:{uuid.uuid4()}] ", add_special_tokens=False)
            rest = n - len(prefix)
            ids = prefix[:n] if rest < 1 else prefix + self.span_ids(rest)
            prompts.append(self.tok.decode(ids[:n], skip_special_tokens=True))
        return prompts

    def shared_prefix(self, n):
        return self.span(n - min(32, n // 2))

    def full_cache(self, prefix, n, count):
        """Prompts with a shared cached prefix and a short unique suffix."""
        out = []
        for _ in range(count):
            ids = self.tok.encode(prefix + " " + self.span(min(32, n // 2)), add_special_tokens=False)
            out.append(self.tok.decode(ids[:n], skip_special_tokens=True))
        return out


async def _request(client, model, prompt, max_tokens):
    """Returns (start, end, first-token time, input tokens, output tokens), or None if the
    request fails."""
    start, first, usage, chunks = time.perf_counter(), None, None, 0
    try:
        stream = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens,
            temperature=0.0, stream=True, stream_options={"include_usage": True}, timeout=600,
            extra_body={"ignore_eos": True, **llm.NO_THINKING})
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                chunks += 1
                first = first or time.perf_counter()
    except Exception as e:
        print(f"request failed: {e}")
        return None
    # Without a usage chunk, count the streamed chunks as output tokens.
    return (start, time.perf_counter(), first, usage.prompt_tokens if usage else 0,
            (usage.completion_tokens if usage else 0) or chunks)


async def _batch(client, model, prompts, max_tokens):
    results = [r for r in await asyncio.gather(*(_request(client, model, p, max_tokens) for p in prompts)) if r]
    if not results:  # A batch in which every request failed counts as zero throughput.
        return {"e2e": 0.0, "ttft_ms": 0.0}
    wall = max(r[1] for r in results) - min(r[0] for r in results)
    tokens = sum(r[3] + r[4] for r in results)
    ttfts = [(r[2] - r[0]) * 1000 for r in results if r[2] is not None]
    return {"e2e": tokens / wall if wall > 0 else 0.0, "ttft_ms": statistics.median(ttfts) if ttfts else 0.0}


async def measure(model, max_context, max_kv_tokens, out_path):
    from transformers import AutoTokenizer

    if not CORPUS.exists():
        download_corpus(CORPUS)
    tokenizer, corpus = AutoTokenizer.from_pretrained(TOKENIZER), CORPUS.read_text(encoding="utf-8")
    client = AsyncOpenAI(base_url=llm.TARGET_URL, api_key=os.environ.get("LLM_API_KEY", "EMPTY"))
    grid = [(i, o, c) for i in INPUT_LENGTHS for o in OUTPUT_LENGTHS for c in CONCURRENCY
            if i + o <= max_context and (i + o) * c <= max_kv_tokens]
    rows = []
    for cache in CACHE_CONDITIONS:
        prompts = Prompts(tokenizer, corpus, random.Random(42))
        prefixes = {}
        if cache == "full_cache":
            for n in sorted({i for i, _, _ in grid}):
                prefixes[n] = prompts.shared_prefix(n)
            # Each shared prefix is sent once before the measurement, so that the measured
            # requests hit the cache.
            for n in sorted(prefixes):
                await _batch(client, model, prompts.full_cache(prefixes[n], n, 1), 4)
        for inp, out, conc in grid:
            reps = []
            for rep in range(N_WARMUP + N_REPS):
                batch = (prompts.no_cache(inp, conc) if cache == "no_cache"
                         else prompts.full_cache(prefixes[inp], inp, conc))
                r = await _batch(client, model, batch, out)
                if rep >= N_WARMUP:
                    reps.append(r)
            if reps:
                rows.append({"cache": cache, "input": inp, "output": out, "concurrency": conc,
                             "e2e_tok_s": statistics.mean(r["e2e"] for r in reps),
                             "ttft_median_ms": statistics.mean(r["ttft_ms"] for r in reps)})
                print(rows[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="served model or LoRA adapter name")
    p.add_argument("--name", required=True, help="<model family>__<base|lora>, for example Qwen3-0.6B__base")
    p.add_argument("--max-context", type=int, required=True)
    p.add_argument("--max-kv-tokens", type=int, required=True)
    p.add_argument("--out", type=Path, default=Path("results/throughput"))
    args = p.parse_args()
    asyncio.run(measure(args.model, args.max_context, args.max_kv_tokens, args.out / f"{args.name}.csv"))


if __name__ == "__main__":
    main()

"""Chat client for the OpenAI-compatible vLLM servers started by serve.sh."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from openai import APIConnectionError, APITimeoutError, OpenAI

TARGET_URL = os.environ.get("TARGET_URL", "http://localhost:8000/v1")
JUDGE_URL = os.environ.get("JUDGE_URL", "http://localhost:8001/v1")
JUDGE_MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 2048
WORKERS = int(os.environ.get("LLM_WORKERS", "64"))
# With this setting, Qwen3 hybrid models answer without reasoning, and the chat template
# inserts an empty <think></think> block. Qwen3-4B-Instruct-2507's template ignores the flag.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: bool = False


def _client(base_url):
    return OpenAI(base_url=base_url, api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
                  max_retries=5, timeout=3600)


def chat(prompts, model, base_url=TARGET_URL, system=None, temperature=0.0,
         max_tokens=MAX_TOKENS, extra_body=NO_THINKING) -> list[Completion]:
    """One single-turn chat completion per prompt, in prompt order.

    A failed request, for example one with a prompt longer than the context, comes back with
    error=True.
    """
    client = _client(base_url)

    def one(prompt):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, top_p=1.0,
                max_tokens=max_tokens, extra_body=extra_body)
        except (APIConnectionError, APITimeoutError):
            raise  # an unreachable server would otherwise look like a run of failed requests
        except Exception as e:
            return Completion(f"ERROR: {e}", error=True)
        text = r.choices[0].message.content
        usage = r.usage
        if text is None:
            return Completion("ERROR: empty response", usage.prompt_tokens, usage.completion_tokens, True)
        return Completion(text, usage.prompt_tokens, usage.completion_tokens)

    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(one, prompts))


def judge(prompts, system) -> list[Completion]:
    return chat(prompts, JUDGE_MODEL, JUDGE_URL, system=system, max_tokens=8192,
                extra_body={"reasoning_effort": "medium"})

"""LoRA fine-tuning with a grid of 27 configurations and 5-fold cross-validation.

    python -m src.lora train    --model Qwen/Qwen3-0.6B --task trec --n 25 --seed 0
    python -m src.lora evaluate --model Qwen/Qwen3-0.6B --task trec --n 25 --seed 0

`train` fits 27 x 5 adapters on one GPU. `evaluate` scores each adapter on its validation
fold. It selects the configuration with the highest mean validation accuracy and, within it,
the fold with the highest validation accuracy. Then it tests that adapter on the 300 test
samples. Text2SQL and Docstring need the judge server from serve.sh.
"""

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

from . import data, llm, tasks

RANK = 64
FOLDS = 5
SEED = 42
MAX_SEQ_LENGTH = 8192
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATES = [1e-4, 3e-4, 5e-4]
BATCH_SIZES = [2, 4, 8]
EPOCHS = [2, 4, 8]
GRID = [(lr, bs, ep) for ep in EPOCHS for bs in BATCH_SIZES for lr in LEARNING_RATES]


def run_name(lr, bs, epochs, fold):
    return f"lr{lr:g}_bs{bs}_e{epochs}_f{fold}"


def training_text(tokenizer, sample):
    # Unlike format_prompt, this adds no input label such as "Question: " for TREC,
    # E-Commerce, Injury and GitAssistant.
    convo = [{"role": "user", "content": f"{sample.task_description}\n\n{sample.input}"},
             {"role": "assistant", "content": sample.gold_answer}]
    return tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)


def _find(seq, sub, start=0):
    for i in range(start, len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


def response_labels(tokenizer, input_ids):
    """Labels that train only on the assistant answer and its closing <|im_end|>."""
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        tokenize=False, add_generation_prompt=False)
    # Qwen3 hybrid models put an empty think block before the answer.
    marker = "</think>\n\n" if "<think>" in probe else "<|im_start|>assistant\n"
    resp = tokenizer.encode(marker, add_special_tokens=False)
    user = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    labels, pos = [-100] * len(input_ids), 0
    while (r := _find(input_ids, resp, pos)) != -1:
        start = r + len(resp)
        nxt = _find(input_ids, user, start)
        if nxt == -1:
            e = _find(input_ids, im_end, start)
            end = e + len(im_end) if e != -1 else len(input_ids)
        else:
            end = nxt
        labels[start:end] = input_ids[start:end]
        pos = end
    return labels


def train_one(model_id, task, n, seed, fold, lr, bs, epochs, out_dir):
    import numpy as np
    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
    from transformers import DataCollatorForSeq2Seq

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id, max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=RANK, lora_alpha=RANK, target_modules=TARGET_MODULES, lora_dropout=0.0,
        bias="none", use_gradient_checkpointing="unsloth", random_state=SEED,
        max_seq_length=MAX_SEQ_LENGTH)
    start = time.time()  # Model loading is not part of the hyperparameter search cost.
    train, _ = data.kfold(task, n, seed, FOLDS, fold)
    dataset = Dataset.from_dict({"text": [training_text(tokenizer, s) for s in train]})
    trainer = UnslothTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        args=UnslothTrainingArguments(
            output_dir=str(out_dir), dataset_num_proc=1, max_seq_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=bs, gradient_accumulation_steps=1,
            num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="constant",
            warmup_steps=0, optim="adamw_torch_fused", weight_decay=0.1, seed=SEED,
            bf16=True, fp16=False, save_strategy="no", logging_steps=1, report_to="none"),
    )
    # UnslothTrainer may have tokenized the dataset already.
    if "input_ids" in trainer.train_dataset.column_names:
        trainer.train_dataset = trainer.train_dataset.map(
            lambda b: {"labels": [response_labels(tokenizer, ids) for ids in b["input_ids"]]},
            batched=True)
    else:
        trainer.train_dataset = trainer.train_dataset.map(
            lambda b: (t := tokenizer(b["text"], truncation=True, max_length=MAX_SEQ_LENGTH,
                                      add_special_tokens=False))
            | {"labels": [response_labels(tokenizer, ids) for ids in t["input_ids"]]},
            batched=True, remove_columns=["text"])
        trainer.data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return time.time() - start


def run_dir(out, model, task, n, seed):
    return out / model.split("/")[-1] / task / f"n{n}_s{seed}"


def train(model, task, n, seed, out):
    """Each run trains in a fresh process, which frees the GPU when it exits."""
    for i, (lr, bs, epochs) in enumerate(GRID):
        for fold in range(FOLDS):
            d = run_dir(out, model, task, n, seed) / run_name(lr, bs, epochs, fold)
            if not (d / "train.json").exists():
                subprocess.run([sys.executable, "-m", "src.lora", "train-one", "--model", model,
                                "--task", task, "--n", str(n), "--seed", str(seed), "--out", str(out),
                                "--config", str(i), "--fold", str(fold)], check=True)


def train_run(model, task, n, seed, out, config, fold):
    lr, bs, epochs = GRID[config]
    d = run_dir(out, model, task, n, seed) / run_name(lr, bs, epochs, fold)
    d.mkdir(parents=True, exist_ok=True)
    seconds = train_one(model, task, n, seed, fold, lr, bs, epochs, d)
    (d / "train.json").write_text(json.dumps({"seconds": seconds}))
    print(f"{d}: trained in {seconds:.0f}s")


def select(val):
    """Returns (lr, bs, epochs, fold) of the selected adapter from {run_name: val accuracy}."""
    # With math.fsum, configurations with the same fold accuracies tie exactly, whatever the
    # order of the folds.
    means = {cfg: math.fsum(val[run_name(*cfg, f)] for f in range(FOLDS)) / FOLDS for cfg in GRID}
    lr, bs, epochs = min(means, key=lambda c: (-means[c], c[2], c[0], -c[1]))
    fold = max(range(FOLDS), key=lambda f: (val[run_name(lr, bs, epochs, f)], -f))
    return lr, bs, epochs, fold


def evaluate(model, task, n, seed, out):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    t, base = tasks.get(task), run_dir(out, model, task, n, seed)
    engine = LLM(model=model, enable_lora=True, max_lora_rank=RANK,
                 max_model_len=MAX_SEQ_LENGTH, gpu_memory_utilization=0.9)
    params = SamplingParams(temperature=0.0, max_tokens=llm.MAX_TOKENS)

    def generate(adapter_id, name, samples):
        outs = engine.chat([[{"role": "user", "content": t.format_prompt(s)}] for s in samples], params,
                           lora_request=LoRARequest(name, adapter_id, str(base / name)),
                           chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        return [llm.Completion(o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids))
                for o in outs]

    def accuracy(samples, completions):
        return sum(bool(r and r.correct) for r in t.score(samples, completions)) / len(samples)

    val, eval_seconds = {}, 0.0
    for i, (lr, bs, epochs) in enumerate(GRID):
        for fold in range(FOLDS):
            name = run_name(lr, bs, epochs, fold)
            _, val_set = data.kfold(task, n, seed, FOLDS, fold)
            start = time.time()
            val[name] = accuracy(val_set, generate(i * FOLDS + fold + 1, name, val_set))
            eval_seconds += time.time() - start
    lr, bs, epochs, fold = select(val)
    best = run_name(lr, bs, epochs, fold)
    test = data.test_set(task)
    completions = generate(len(GRID) * FOLDS + 1, best, test)
    train_seconds = sum(json.loads((base / name / "train.json").read_text())["seconds"] for name in val)
    result = {
        "model": model, "task": task, "n": n, "seed": seed,
        "selected": {"learning_rate": lr, "batch_size": bs, "epochs": epochs, "fold": fold},
        "adapter": str(base / best), "val_accuracy": val,
        "test_accuracy": accuracy(test, completions),
        # The hyperparameter search cost is the wall time of all 135 training runs plus their
        # validation evaluation.
        "optimization_gpu_hours": (train_seconds + eval_seconds) / 3600,
        "prompt_tokens": sum(c.prompt_tokens for c in completions),
        "completion_tokens": sum(c.completion_tokens for c in completions),
    }
    (base / "result.json").write_text(json.dumps(result, indent=1))
    print(f"{base}: {best} test accuracy={result['test_accuracy']:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["train", "evaluate", "train-one"])
    p.add_argument("--model", required=True, help="bf16 checkpoint, for example Qwen/Qwen3-0.6B")
    p.add_argument("--task", required=True, choices=list(tasks.TASKS))
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, default=Path("results/lora"))
    # train-one gets the configuration as an index into GRID.
    p.add_argument("--config", type=int, help=argparse.SUPPRESS)
    p.add_argument("--fold", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.command == "train-one":
        train_run(args.model, args.task, args.n, args.seed, args.out, args.config, args.fold)
    else:
        (train if args.command == "train" else evaluate)(args.model, args.task, args.n, args.seed, args.out)


if __name__ == "__main__":
    main()

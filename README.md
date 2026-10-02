# An Accuracy and Cost Analysis of Prompting and Fine-Tuning for Small Language Models

Code, task data and test results for the paper. It compares three in-context learning (ICL)
strategies, GEPA prompt optimization and LoRA fine-tuning on 17 narrow tasks, for Qwen3-0.6B,
Qwen3-1.7B and Qwen3-4B-Instruct-2507 with N ∈ {5, 10, 25, 50, 100, 200} training samples,
and converts each method's token and GPU usage into dollar cost.

## Setup

The experiments ran on NVIDIA A6000 (48 GB) GPUs. The pinned packages need Python 3.10
to 3.13. Run all commands from the repository root.

```bash
pip install -r requirements.txt
```

Each server, each in-process vLLM engine (the RAG embedder and `src.lora evaluate`) and
LoRA training needs its own GPUs. The commands below use GPUs 0 and 1 for the judge, GPU 2
for the target server and GPU 3 for the rest.

`src/llm.py` reads four environment variables:

| Variable | Default | Server |
|---|---|---|
| `TARGET_URL` | `http://localhost:8000/v1` | the model under test |
| `JUDGE_URL` | `http://localhost:8001/v1` | gpt-oss-120b |
| `LLM_API_KEY` | `EMPTY` | key for both, if the servers need one |
| `LLM_WORKERS` | `64` | concurrent requests per call of `chat()` |

## Tasks

`data/` holds the 17 tasks as xz-compressed JSONL. `src/data.py` draws a fixed test set of
300 samples with seed 42. For each run seed, it draws the N training samples from the
remaining pool, so every method sees the same samples. Sources and licenses are in
[data/README.md](data/README.md).

| Category | Tasks | Correct when |
|---|---|---|
| Function calling | `acebench_hard`, `acebench_medium`, `smart_home`, `git_assistant` | the calls match the gold calls |
| Classification | `trec`, `ecommerce`, `injury` | the first label in the answer is the gold label |
| Text extraction | `currency`, `dates` | the normalized items equal the gold items |
| Format following | `emoji`, `emoji_list` | every IFBench constraint passes |
| Text modification | `replace`, `delete`, `insert_before` | every edit step is right |
| Code | `text2sql`, `docstring` | gpt-oss-120b judges the answer Good |
| PII redaction | `pii` | redacted text and entities match |

## Run the experiments

`src.icl` and `src.gepa` write one JSON file per run under `results/` and skip runs that
already exist. By default they cover all 17 tasks, all six N and seeds 0 and 1. Use
`--tasks`, `--n` and `--seeds` to run a subset.

1. Start the judge. Text2SQL, Docstring and GEPA need it. It runs gpt-oss-120b on two
   A6000s.

   ```bash
   CUDA_VISIBLE_DEVICES=0,1 ./serve.sh judge
   ```

2. Run ICL. All-samples puts every training sample in the prompt. CV-greedy selects up to
   30 demonstrations by 5-fold greedy forward selection. RAG picks the 10 training samples
   nearest to each query, or all of them when N < 10, with BAAI/bge-m3, which loads
   in-process.

   ```bash
   CUDA_VISIBLE_DEVICES=2 ./serve.sh icl Qwen/Qwen3-0.6B-FP8
   python -m src.icl --model Qwen/Qwen3-0.6B-FP8 --method all
   python -m src.icl --model Qwen/Qwen3-0.6B-FP8 --method cv_greedy
   CUDA_VISIBLE_DEVICES=3 python -m src.icl --model Qwen/Qwen3-0.6B-FP8 --method rag
   ```

   The reference runs with larger models use `Qwen/Qwen3-8B-FP8` and `Qwen/Qwen3-14B-FP8`
   with `--method all`.

3. Run GEPA. It optimizes the instruction on each of two folds. The paper reports the fold
   that scores higher on all N training samples, which `src.gepa.best_fold` selects.

   ```bash
   CUDA_VISIBLE_DEVICES=2 ./serve.sh target Qwen/Qwen3-0.6B-FP8
   python -m src.gepa --model Qwen/Qwen3-0.6B-FP8
   python -m src.gepa --model Qwen/Qwen3-0.6B-FP8 --max-full-evals 40   # GEPA (2x)
   ```

4. Run LoRA. `train` fits 27 hyperparameter configurations on each of 5 folds. `evaluate`
   selects the best configuration and fold by validation accuracy and tests that adapter.
   LoRA uses the bf16 checkpoints, not FP8.

   ```bash
   export CUDA_VISIBLE_DEVICES=3
   tasks=$(python -c "from src import tasks; print(*tasks.TASKS)")
   for task in $tasks; do for n in 5 10 25 50 100 200; do for seed in 0 1; do
     python -m src.lora train    --model Qwen/Qwen3-0.6B --task $task --n $n --seed $seed
     python -m src.lora evaluate --model Qwen/Qwen3-0.6B --task $task --n $n --seed $seed
   done; done; done
   ```

5. Run LoRA + GEPA. It needs the LoRA results of the same model for N up to 100.

   ```bash
   CUDA_VISIBLE_DEVICES=2 ./serve.sh lora Qwen/Qwen3-0.6B
   python -m src.gepa --model Qwen/Qwen3-0.6B --lora --n 5 10 25 50 100
   ```

## Compute the costs

The total cost of Q deployment queries is the setup cost plus Q times the per-query cost.
`src.throughput` measures each model on one GPU, and `src.cost` uses these measurements to
convert token counts into cost at $1.09 per GPU hour.

1. Measure each model once with the ICL server and once with one LoRA adapter loaded.
   On an A6000, Qwen3-4B-Instruct-2507 with LoRA starts only with `--max-model-len 242592`
   added to the `serve.sh lora` command, because its KV cache holds fewer tokens than the
   native 262144 context.

   ```bash
   CUDA_VISIBLE_DEVICES=2 ./serve.sh icl Qwen/Qwen3-0.6B-FP8
   python -m src.throughput --model Qwen/Qwen3-0.6B-FP8 --name Qwen3-0.6B__base \
       --max-context 131072 --max-kv-tokens 379088

   CUDA_VISIBLE_DEVICES=2 ./serve.sh lora Qwen/Qwen3-0.6B --max-loras 1
   python -m src.throughput --model trec__n5__s0 --name Qwen3-0.6B__lora \
       --max-context 40960 --max-kv-tokens 374368
   ```

   `--max-kv-tokens` is the KV-cache size that vLLM prints at startup. The table lists the
   values of the paper's A6000 servers. On other GPUs, take them from the vLLM log.

   | Model | `__base` context | `__base` KV tokens | `__lora` context | `__lora` KV tokens |
   |---|---|---|---|---|
   | Qwen3-0.6B | 131072 | 379088 | 40960 | 374368 |
   | Qwen3-1.7B | 131072 | 367984 | 40960 | 354192 |
   | Qwen3-4B-Instruct-2507 | 262144 | 269632 | 242592 | 242592 |

2. Write the cost table:

   ```bash
   python -m src.cost
   ```

   `results/costs.csv` has one row per method, model, task and N. The `lora` rows assume
   that the adapter is merged into the base weights, and the `lora_adapter` rows assume that
   it is served unmerged.

## Paper results

`paper_results/` holds the test accuracies behind the paper's figures and tables.
`accuracy.csv` has one row per method, model, task, N and seed, for seeds 0 and 1. The
methods are `all`, `rag`, `cv_greedy`, `lora`, `gepa`, `gepa_2x` and `gepa_lora`. The columns
are `method`, `model`, `task`, `n`, `seed`, `accuracy`, `n_errors`, `learning_rate`,
`batch_size`, `epochs` and `fold`. `n_errors` counts the failed requests of an ICL run and is
empty for the other methods. `accuracy` is empty when all 300 requests failed. For `lora`,
the last four columns give the configuration and fold that validation accuracy selected. When
folds tie, the paper's runs took the fold whose result was written first, while `src.lora`
takes the lowest fold, so the two can differ. For
the GEPA methods, `fold` is the fold with the higher score on all N training samples. `all`
covers all five models, `gepa_2x` covers Qwen3-0.6B, and `gepa_lora` covers Qwen3-0.6B and
Qwen3-1.7B up to N=100. The other methods cover Qwen3-0.6B, Qwen3-1.7B and
Qwen3-4B-Instruct-2507. `gepa_instructions.jsonl` holds the optimized instruction of each
reported GEPA run.

## License

The repository is MIT-licensed. `src/tasks/acebench.py` and `src/tasks/ifbench.py` adapt
code from ACEBench (MIT) and IFBench (Apache-2.0). Their licenses are in `licenses/`. The
three classification datasets (`trec`, `ecommerce`, `injury`) keep the licenses of their
sources, which [data/README.md](data/README.md) lists.

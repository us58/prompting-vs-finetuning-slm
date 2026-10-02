#!/usr/bin/env bash
# Starts the vLLM 0.16.0 servers for the experiments, each target model on one A6000 (48 GB).
#
#   ./serve.sh icl Qwen/Qwen3-0.6B-FP8     for ICL and the base-model throughput benchmark
#   ./serve.sh target Qwen/Qwen3-0.6B-FP8  for GEPA, with the native context
#   ./serve.sh lora Qwen/Qwen3-0.6B [args] for LoRA + GEPA and the LoRA throughput benchmark
#   ./serve.sh judge                       gpt-oss-120b as judge and GEPA reflection model, on 2 GPUs
#
# Targets listen on $PORT, by default 8000 as in TARGET_URL of src/llm.py. The judge uses 8001.
# The lora mode serves the bf16 base with the adapters selected under $ADAPTERS (results/lora).
set -euo pipefail
PORT="${PORT:-8000}"
YARN='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'

if [[ "${1:-}" == icl || "${1:-}" == target || "${1:-}" == lora ]]; then
  : "${2:?a model name is required}"
fi

case "${1:-}" in
icl)
  # ICL prompts with up to 200 demonstrations need more than the native 32K context. YaRN
  # extends Qwen3-0.6B, 1.7B, 8B and 14B to 128K, and Qwen3-4B-Instruct-2507 has 256K.
  if [[ "$2" == *Qwen3-4B-Instruct-2507* ]]; then
    exec vllm serve "$2" --port "$PORT" --gpu-memory-utilization 0.9
  fi
  exec vllm serve "$2" --port "$PORT" --gpu-memory-utilization 0.9 \
    --hf-overrides "$YARN" --max-model-len 131072
  ;;
target)
  exec vllm serve "$2" --port "$PORT" --gpu-memory-utilization 0.9
  ;;
lora)
  # Each adapter is served as <task>__n<N>__s<seed>, the name that src.gepa --lora requests.
  mapfile -t modules < <(python - "${ADAPTERS:-results/lora}" "$2" <<'EOF'
import json, sys
from pathlib import Path
for p in sorted(Path(sys.argv[1], sys.argv[2].split("/")[-1]).glob("*/*/result.json")):
    r = json.loads(p.read_text())
    print(f"{r['task']}__n{r['n']}__s{r['seed']}={r['adapter']}")
EOF
)
  if [[ ${#modules[@]} -eq 0 ]]; then
    echo "no LoRA results for $2 under ${ADAPTERS:-results/lora}; run src.lora evaluate first" >&2
    exit 1
  fi
  exec vllm serve "$2" --port "$PORT" --gpu-memory-utilization 0.9 \
    --enable-lora --max-lora-rank 64 --max-loras 32 --lora-modules "${modules[@]}" "${@:3}"
  ;;
judge)
  # gpt-oss-120b needs two A6000s.
  exec vllm serve openai/gpt-oss-120b --port 8001 --tensor-parallel-size 2
  ;;
*)
  sed -n '2,10p' "$0"
  exit 1
  ;;
esac

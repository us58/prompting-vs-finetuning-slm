# Task data

Each `<task>.jsonl.xz` holds one task. Every line is a sample:

| Field | Content |
|---|---|
| `id` | unique sample id |
| `benchmark` | internal name of the dataset version |
| `task_description` | the instruction, the same for every sample of a task |
| `input` | the query |
| `gold_answer` | the reference answer |
| `eval_context` | what the task's evaluator in `src/tasks/` needs |

Before splitting, `src/data.py` drops samples without a gold answer, classification samples
over a token limit, ACEBench samples whose gold values are missing from the input, and
ACEBench Hard samples that ask for a distractor function. The splits depend on the order of
the remaining samples. Do not reorder or trim the files.

## Sources and licenses

Three tasks use published data after filtering ("Data"). Twelve take the task design or
evaluation logic from a published benchmark and use independently generated samples
("Concept"). Currency and Dates are original to this work.

| Tasks | Source | License | Use |
|---|---|---|---|
| `acebench_hard`, `acebench_medium` | [ACEBench](https://github.com/ACEBench/ACEBench) | MIT | Concept |
| `emoji`, `emoji_list` | [IFBench](https://github.com/allenai/IFBench) | Apache-2.0 (code), ODC-BY-1.0 (data) | Concept |
| `replace`, `delete`, `insert_before` | [SIFo](https://github.com/shin-ee-chen/SIFo) | not specified | Concept |
| `trec` | [TREC question classification](https://cogcomp.seas.upenn.edu/Data/QA/QC/) | not specified | Data |
| `ecommerce` | [E-commerce text classification](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification) | CC BY 4.0 | Data |
| `injury` | [CDC occupational injury coding](https://github.com/NASA-Tournament-Lab/CDC-NLP-Occ-Injury-Coding) | CC0 1.0 | Data |
| `smart_home`, `git_assistant`, `pii`, `text2sql`, `docstring` | [Distil Labs benchmarks](https://github.com/distil-labs/inference-efficiency-benchmarks) | not specified | Concept |
| `currency`, `dates` | this work | | |

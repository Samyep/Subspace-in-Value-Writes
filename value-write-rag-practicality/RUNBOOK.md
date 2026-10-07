# Multi-machine runbook

Run all commands from the repository root. The BGE comparison has one shared
selection stage followed by four independent reader jobs. The cross-model
scout experiment has two shared selector files followed by two fixed reader
jobs.

## 1. Validate every clone

```bash
python scripts/validate_bundle.py
python -m pytest -q
```

All clones should report the same Git commit and pass the frozen-artifact checks before GPU work starts.

The provided Slurm files set Hugging Face and Transformers to offline mode.
Before submitting them, confirm that the cache contains the checkpoints named
in `README.md`. A missing checkpoint should be downloaded on a login or other
network-enabled node first.

## 2. Produce the shared reranker selections

A single GPU can score all 400 questions:

```bash
python scripts/score_reranker.py \
  --output results/reranker/bge_v2_m3/selections.jsonl
```

To split this stage across four machines, run one command per machine with `SHARD` set to 0, 1, 2, or 3:

```bash
SHARD=0
python scripts/score_reranker.py \
  --num-shards 4 \
  --shard-index "$SHARD" \
  --output "results/reranker/bge_v2_m3/part-${SHARD}-of-4.jsonl"
```

Collect the four part files on one machine and merge them:

```bash
python scripts/merge_jsonl.py \
  --input results/reranker/bge_v2_m3 \
  --output results/reranker/bge_v2_m3/selections.jsonl \
  --expected-data data/musique_pooled_rag_confirm.jsonl
```

Copy `selections.jsonl` to the same path in every clone. Do not start the reader jobs until the merge reports 400 unique rows.

## 3. Produce the truncated-scout selections

One Llama-3.2-3B pass produces both selector variants:

```bash
python scripts/score_truncated_scout.py \
  --output results/scout/llama_3b_l17/selections.jsonl
```

The run must finish with 400 rows. It also reports Value and Attention top-2
agreement against `reference_results/llama_3b/per_example.jsonl`. Inspect:

```text
results/scout/llama_3b_l17/selections.summary.json
```

The released-run comparison can reflect floating-point differences across
hardware or software versions. It is a reproduction diagnostic and does not
gate cross-reader generation. After the full Llama cost pass finishes, run:

```bash
python scripts/audit_scout_prefix.py
```

This compares the full and truncated passes from the current runtime over all
400 questions and writes `results/scout/llama_3b_l17/prefix_audit.json`.

## 4. Run the four-reader BGE comparison in parallel

Assign one cell to each machine:

```bash
python scripts/run_reranker_reader.py --cell qwen_7b
python scripts/run_reranker_reader.py --cell qwen_14b
python scripts/run_reranker_reader.py --cell qwen3_8b
python scripts/run_reranker_reader.py --cell llama_3b
```

Each command writes one 400-row part file below `results/readers/<cell>/reranker/`. If a reader must be split further, add the same `--num-shards N --shard-index I` pair used above. Merge reader shards with:

```bash
CELL=qwen_7b
python scripts/merge_jsonl.py \
  --input "results/readers/${CELL}/reranker" \
  --output "results/readers/${CELL}/reranker/merged.jsonl" \
  --expected-data data/musique_pooled_rag_confirm.jsonl
```

Collect the four reader directories on the aggregation machine. The scripts are append-only and resumable; rerunning the same command skips completed IDs.

## 5. Run the two fixed cross-model readers in parallel

Qwen2.5-14B is the primary reader and Qwen3-8B is the predeclared confirmation
reader. Both consume the same BGE and truncated-scout selections:

```bash
python scripts/run_scout_reader.py --reader-cell qwen_14b
python scripts/run_scout_reader.py --reader-cell qwen3_8b
```

Each command reruns Full-20, BM25-2, BGE-2, 3B Attention-2, 3B Value-2, and
Oracle-2 under one reader. Condition order rotates with the frozen question
rank. The output directories are:

```text
results/scout_readers/qwen_14b
results/scout_readers/qwen3_8b
```

When machines do not share storage, copy the complete BGE and scout selection
directories to both reader machines first. After generation, copy both reader
directories back to the aggregation machine. Keep the run-config and summary
JSON files with the JSONL rows.

## 6. Run the original four-reader cost audit

Use one machine per reader cell when possible. Run the scoring and generation jobs on the same GPU class and software environment.

```bash
CELL=qwen_7b
python scripts/benchmark_value_scoring.py --cell "$CELL" --max-examples 0
python scripts/benchmark_generation.py --cell "$CELL" --max-examples 0
```

The first command measures the full 20-passage eager-attention scoring pass and records whether Value-2 and Attention-2 reproduce the released selections. The second measures optimized generation for Full-20, Value-2, Attention-2, Retrieval-2, and Reranker-2. It rotates condition order across questions to reduce order effects.

For a smaller cost audit, use the same positive value for `--max-examples` in both commands. If jobs are sharded, merge each of these directories in the same way as the reader outputs:

```text
results/cost/<cell>/scoring
results/cost/<cell>/generation
```

## 7. Aggregate

After collecting all four reader outputs:

```bash
python scripts/aggregate_practicality.py --allow-missing-cost
```

Once all cost outputs are present, run without the flag:

```bash
python scripts/aggregate_practicality.py
```

The final files are:

```text
results/aggregate/practicality_summary.json
results/aggregate/practicality_report.md
```

After both cross-model reader jobs finish, aggregate the scout experiment:

```bash
python scripts/aggregate_scout.py
```

This writes:

```text
results/scout_aggregate/scout_summary.json
results/scout_aggregate/scout_report.md
```

The cost table adds the shared scout pass or BGE scoring pass to the matching
reader-generation pass. Compare wall time only when the runtime signatures in
the JSON show the same GPU and software versions.

The JSON file contains confidence intervals and all component metrics. The Markdown file is a compact table for paper drafting.

## Cluster templates

The `slurm/` files follow the current bgvf single-GPU convention and point to this checkout. Create `logs/` before `sbatch`. The BGE and scout selection jobs must finish before the corresponding reader arrays start. The cost array likewise requires the merged BGE selection file.

```bash
mkdir -p logs
reranker_job=$(sbatch --parsable slurm/bgvf_reranker.slurm)
scout_job=$(sbatch --parsable slurm/bgvf_scout_select.slurm)
sbatch --dependency="afterok:${reranker_job}" slurm/bgvf_reader_array.slurm
sbatch --dependency="afterok:${reranker_job}:${scout_job}" \
  slurm/bgvf_scout_reader_array.slurm
sbatch --dependency="afterok:${reranker_job}" slurm/bgvf_cost_array.slurm
```

On another cluster, update the account, partition, repository path, environment activation, and Hugging Face cache before submission. The plain Python commands above are scheduler-independent.

The bgvf interactive QOS permits only one running job per user. To use all four
GPUs in that job while keeping every model process isolated to one GPU, submit:

```bash
sbatch slurm/bgvf_all_experiments_4gpu.slurm
```

This job runs the two shared selectors first, queues the ten independent reader
and cost tasks across four one-GPU Slurm steps, and writes a separate log for
every step. It finishes by running both aggregators. All underlying Python
programs remain append-only and resumable.

## Frozen long-context extension

The extension protocol is fixed in `FROZEN_EXTENSION_PROTOCOL.md`. The
deterministic top-160 pool and held-out calibration set are included. Validate
them before any GPU run:

```bash
python scripts/validate_extension.py
```

The two preparation scripts remain available for provenance. Rebuilding either
artifact requires the original cached source dataset and evaluation files; see
their `--help` output for the required source paths.

Score the shared selectors, calibrate the one allowed hybrid on held-out
HotpotQA, and run the two readers:

```bash
selector_job=$(sbatch --parsable slurm/bgvf_context_selectors.slurm)
sbatch --dependency="afterok:${selector_job}" slurm/bgvf_context_readers.slurm
```

The selector job writes 400-row BGE and scout files under
`results/context_scaling/`, then evaluates Value weights 0.25, 0.50, and 0.75
against both endpoints on the fixed 200-example calibration set. It launches
one MuSiQue hybrid evaluation only if the selected interior weight improves
macro support recall@2 by at least 1.00 point over the better endpoint.

The reader job uses two shards per reader. Merge and aggregate manually after
a partial or multi-machine run with:

```bash
python scripts/merge_jsonl.py \
  --input results/context_scaling/readers/qwen3_8b \
  --output results/context_scaling/readers/qwen3_8b/merged.jsonl \
  --expected-data data/musique_long_context_pool.jsonl
python scripts/merge_jsonl.py \
  --input results/context_scaling/readers/qwen_14b \
  --output results/context_scaling/readers/qwen_14b/merged.jsonl \
  --expected-data data/musique_long_context_pool.jsonl
python scripts/aggregate_context_scaling.py
```

Every selector and reader command is append-only and resumes from completed
question IDs. Do not change the depths, readers, direction, hybrid weights, or
gate after inspecting MuSiQue outcomes.

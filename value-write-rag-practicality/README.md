# Value-Write RAG practicality experiments

This repository contains the self-contained experiment bundle for the practical RAG follow-up to the value-write study. It evaluates passage selection on a frozen, held-out set of 400 two-hop MuSiQue questions retrieved from a shared 101,958-paragraph corpus.

The experiment bundle has four goals:

1. Compare Value-2 with a strong query-passage cross-encoder, `BAAI/bge-reranker-v2-m3`, under the same top-2 passage budget and the same four reader models.
2. Measure the full inference cost of selecting passages and generating an answer. The cost audit records wall time, input tokens, and peak CUDA memory for the selection and generation passes separately.
3. Test a practical cross-model selector: one Llama-3.2-3B forward pass, truncated after decoder block 17, produces both Value-2 and Attention-2 selections for Qwen2.5-14B and Qwen3-8B readers. The truncation is checked question by question against the released full 3B selector.
4. Measure the operating regime in which the 3B Value scout offsets its own cost. A frozen extension evaluates Full-K, BGE-2, and Value-scout-2 at K in {20, 40, 80, 160}, corresponding to median prompts from about 2.5k to 18.9k tokens, with Qwen3-8B and Qwen2.5-14B readers.

No model weights or activation caches are stored in this repository. The 400-question evaluation file, four frozen directions, and four 400-row reference runs are included so each machine sees exactly the same inputs.

## Included artifacts

- `data/musique_pooled_rag_confirm.jsonl`: frozen 400-question held-out evaluation set, with 20 retrieved passages per question.
- `calibration/*/frozen_direction.npz`: frozen HotpotQA-calibrated directions for the four readers.
- `reference_results/*/per_example.jsonl`: released Full-20, Value-2, Attention-2, Retrieval-2, Random-2, and Oracle-2 outcomes.
- `scripts/score_reranker.py`: cross-encoder scoring and top-2 selection.
- `scripts/run_reranker_reader.py`: answer generation from the reranker-selected passages.
- `scripts/benchmark_value_scoring.py` and `scripts/benchmark_generation.py`: separate scoring and optimized generation measurements for the cost audit.
- `scripts/aggregate_practicality.py`: paired per-reader and shared-question macro summaries with bootstrap confidence intervals.
- `scripts/score_truncated_scout.py`: Llama-3.2-3B decoder-prefix scout, with Value-2 and Attention-2 extracted from the same pass.
- `scripts/audit_scout_prefix.py`: same-runtime comparison between the truncated scout and the full 3B scoring pass.
- `scripts/run_scout_reader.py`: Full-20, BM25-2, BGE-2, scout Attention-2, scout Value-2, and Oracle-2 generation with a fixed reader.
- `scripts/aggregate_scout.py`: paired Qwen2.5-14B primary and Qwen3-8B confirmation summaries, including end-to-end component costs.
- `data/musique_long_context_pool.jsonl`: deterministic top-160 retrieval pool that preserves the original 20 passages exactly and extends them with unused BM25-ranked corpus passages.
- `scripts/analyze_evidence_roles.py`: paired Value/BGE disagreement and first-hop versus terminal-hop retention analysis.
- `scripts/score_long_reranker.py`, `scripts/score_long_scout.py`, and `scripts/run_context_scaling.py`: the frozen long-context selector and reader runs.
- `scripts/calibrate_frozen_hybrid.py`: the predeclared held-out HotpotQA gate for one Value+BGE hybrid. MuSiQue generation runs only if the hybrid exceeds the better calibration endpoint by at least 1.00 support-recall point.
- `scripts/aggregate_context_scaling.py`: paired quality, component timing, end-to-end timing, and crossover summaries.

Every long-running script is resumable. Every output row carries the frozen question ID and rank. Run metadata records portable input identifiers, the model commit when exposed by Transformers, software versions, and the GPU model.

## Frozen extension results

The same-system evidence analysis separates support-chain roles. Averaged over
the four readers, Value-2 has lower annotated support recall than BGE-2 but
higher answer EM. Value-2 retains the terminal evidence hop 9.69 points more
often and the first hop 27.81 points less often. Its macro EM advantage over
BGE-2 is 2.69 points with a paired 95% CI of [0.31, 5.12]. The complete
disagreement counts and conditional outcomes are in
`results/evidence_analysis/evidence_analysis.md`.

The held-out hybrid gate did not pass. On 200 HotpotQA examples, BGE obtained
94.25% macro support recall@2, while the best allowed interior weight (0.25 on
Value) obtained 93.00%. The protocol therefore stopped before any MuSiQue
hybrid generation.

The long-context experiment found the first negative mean end-to-end latency
difference at K=40 for both readers. The shared-question macro scout-minus-Full
difference was -0.080 seconds at K=40 (95% CI [-0.107, -0.053]), -0.271 seconds
at K=80 ([-0.303, -0.241]), and -0.814 seconds at K=160 ([-0.852, -0.776]).
The corresponding macro EM differences were +0.50, -0.25, and -0.63 points;
all quality intervals included zero. The per-reader quality, latency, TTFT,
decode, and selector measurements are in
`results/context_scaling/aggregate/context_scaling_report.md` and its JSON
companion.

## Environment

Python 3.10 or newer is required. Install the CUDA build of PyTorch appropriate for the machine first, then install this project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python scripts/validate_bundle.py
```

The tested cluster environment used Python 3.12, PyTorch 2.10, Transformers 5.11, NumPy 2.4, and scikit-learn 1.8. The code retains compatibility fallbacks for the Transformers 4.x `torch_dtype` argument.

Set `HF_HOME` to a shared model cache if the machines already contain the reader checkpoints:

```bash
export HF_HOME=/path/to/huggingface/cache
export TOKENIZERS_PARALLELISM=false
```

The Slurm templates run offline. Cache every checkpoint needed by a job before
submitting it. In particular, the BGE checkpoint is separate from the four
reader checkpoints:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - <<'PY'
from huggingface_hub import snapshot_download

for model_id in (
    "BAAI/bge-reranker-v2-m3",
    "unsloth/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen3-8B",
):
    snapshot_download(model_id)
PY
```

## Run the experiments

The complete multi-machine sequence, merge commands, and expected output paths are in [RUNBOOK.md](RUNBOOK.md). A minimal data-only check is:

```bash
python scripts/validate_bundle.py
python -m pytest -q
```

A two-question GPU smoke test is:

```bash
python scripts/score_reranker.py \
  --max-examples 2 \
  --output results/reranker/bge_v2_m3/smoke.jsonl

python scripts/run_reranker_reader.py \
  --cell llama_3b \
  --max-examples 2 \
  --reranker-selections results/reranker/bge_v2_m3/smoke.jsonl \
  --output-dir results/smoke/llama_3b

python scripts/score_truncated_scout.py \
  --max-examples 2 \
  --output results/scout/llama_3b_l17/smoke.jsonl

python scripts/run_scout_reader.py \
  --reader-cell qwen_14b \
  --max-examples 2 \
  --scout-selections results/scout/llama_3b_l17/smoke.jsonl \
  --reranker-selections results/reranker/bge_v2_m3/smoke.jsonl \
  --output-dir results/smoke/scout_qwen_14b
```

The full primary experiment uses all 400 questions. The cost scripts also default to all 400; pass `--max-examples 100` for a smaller timing audit.

The long-context extension and its hybrid gate are frozen in
`FROZEN_EXTENSION_PROTOCOL.md`. On the bgvf cluster, the selector/calibration
and reader phases are:

```bash
selector_job=$(sbatch --parsable slurm/bgvf_context_selectors.slurm)
sbatch --dependency="afterok:${selector_job}" slurm/bgvf_context_readers.slurm
```

The first job also evaluates the single predeclared hybrid gate. A failed gate
is a stopping result: it writes the calibration analysis and does not launch a
MuSiQue hybrid run.

The cross-model scout experiment is predeclared with Qwen2.5-14B as the
primary reader and Qwen3-8B as the confirmation reader. Run both readers even
if the primary result is already favorable; this avoids choosing the second
reader after seeing the outcome.

## Evaluation protocol

The evaluation IDs and order were frozen by token length before these follow-up comparisons. Retrieval and direction calibration were fixed without evaluation labels. The cross-encoder scores each of the 20 retrieved passages independently with the question and keeps the two highest-scoring passages. All methods use the same prompt and deterministic greedy answer decoding.

Exact match and token F1 use the standard normalized short-answer comparison. Selection quality reports support recall at 2 and whether both gold support passages were retained. Confidence intervals use 20,000 question-level bootstrap resamples. Macro results first average readers within each question and then resample questions, preserving the paired design.

Wall-clock comparisons are valid when the compared components use the same GPU class and software environment. The aggregator reports the reranker time as a component sum because reranker scoring and reader generation are separate jobs. Keep their runtime metadata with the result files.

The truncated scout retains blocks 0--17 because the frozen Llama direction
uses blocks 14--17. Causal decoder states and attention within that prefix do
not depend on later blocks. Agreement with the released run is recorded as a
cross-environment reproduction diagnostic. The prefix audit compares the
truncated and full passes produced by the same software and hardware runtime,
separating truncation effects from floating-point differences across runs.

For contexts beyond the original 20 passages, the scout reconstructs only the
final-query attention row from pre-RoPE query, key, and value projections. This
preserves the frozen Value score while avoiding a dense sequence-by-sequence
attention tensor. `results/context_scaling/final_query_audit.json` records its
agreement with the original dense implementation on the audit examples.

## Integrity and provenance

Run `python scripts/validate_bundle.py` after every clone. It checks the 400 frozen IDs, passage labels, direction shapes, and all 1,600 released reader rows.

The evaluation data derives from the answerable two-hop MuSiQue validation split (`bdsaglam/musique`, configuration `answerable`). The bundled text and labels remain subject to the source dataset's terms. Model checkpoints are downloaded from their respective Hugging Face repositories and remain subject to their own licenses.

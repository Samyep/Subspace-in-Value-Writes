# Attention Value Writes Reveal Directions for Support Selection

Code, frozen inputs, and released outputs for the controlled support-selection,
causal-intervention, and retrieval-augmented generation experiments in the
paper.

The repository combines three reproducible experiment bundles:

| Experiment bundle | Location | Scope |
|---|---|---|
| Discovery and prompt-local interventions | repository root | Six-model support ranking, span ablation, direction interventions, patching, robustness controls, and the free-form citation bridge |
| Source-disjoint frozen directions | [`source-disjoint-frozen-direction-controls/`](source-disjoint-frozen-direction-controls/) | Six-model intervention with disjoint fitting and evaluation questions, magnitude-matched random controls, and source-clustered confidence intervals |
| Reader-specific RAG | [`value-write-rag-practicality/`](value-write-rag-practicality/) | Held-out MuSiQue passage selection, BGE comparison, evidence-role analysis, cost accounting, and long-context scout scaling |

No model weights or activation caches are stored here. Full reruns require the
listed pretrained checkpoints, access to a CUDA-capable machine, and the source
datasets under their respective terms.

## Main released results

- Across six instruction-tuned models, leading attention value-write
  directions rank annotated support facts above distractors in the controlled
  task. The root artifact bundle contains the prompt-level records and all
  figure inputs.
- The source-disjoint frozen-direction intervention exceeds its matched random
  control in four of six models; the other two comparisons are inconclusive.
- On 400 held-out MuSiQue questions, reader-specific Value-2 reaches 17.12
  macro EM, compared with 14.12 for Full-20 and 14.44 for BGE-2. Its paired
  advantage over BGE-2 is 2.69 EM points with a 95% confidence interval of
  [0.31, 5.12].
- BGE-2 recovers more annotated support passages, while Value-2 retains the
  terminal evidence hop more often. The complete paired disagreement analysis
  is released with per-example records.
- A frozen Llama-3.2-3B value scout becomes faster than full-context generation
  for both tested readers as the retrieval pool grows. At K=160, its two-reader
  macro latency saving is 0.814 seconds, or 44.5% of Full-K time; answer-quality
  intervals include zero at every tested depth.

These summaries describe the released evaluations. See the bundle READMEs for
the exact protocols, model-specific results, and the limits of each comparison.

## Validate the release

Python 3.10 or newer is required.

The root bundle can be checked without rerunning a model:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/validate_artifacts.py
python scripts/summarize_results.py
```

Validate the source-disjoint intervention bundle:

```bash
cd source-disjoint-frozen-direction-controls
python scripts/validate_bundle.py
```

Validate the RAG bundle and regenerate its data-only summaries:

```bash
cd value-write-rag-practicality
python -m pip install -e '.[test]'
python scripts/validate_bundle.py
python scripts/validate_extension.py
python -m pytest -q
```

Use separate environments for full reruns when the checkpoint-specific CUDA or
Transformers versions differ. Each bundle documents its exact run order and
expected output paths.

## Repository layout

- `support_evidence/`: artifact loaders, model-cell metadata, and plotting helpers.
- `scripts/`: validation, summary, prompt inspection, and figure rebuild commands.
- `artifacts/`: frozen controlled-task prompts and per-experiment CSV/JSON outputs.
- `RESULTS.md`: index of the controlled-task results.
- `ARTIFACT_SCHEMA.md`: schema for the root artifact files.
- `DATA_AND_MODELS.md`: datasets, checkpoints, and redistribution notes.
- `REPRODUCTION.md`: root-bundle reproduction commands.
- `source-disjoint-frozen-direction-controls/`: frozen-direction protocol, code,
  audits, directions, per-prompt outcomes, and aggregate results.
- `value-write-rag-practicality/`: RAG protocol, frozen calibration and evaluation
  data, selectors, readers, timing tools, per-example outputs, and aggregate reports.

## Detailed entry points

- [Controlled-task result index](RESULTS.md)
- [Source-disjoint intervention protocol and results](source-disjoint-frozen-direction-controls/README.md)
- [RAG protocol and results](value-write-rag-practicality/README.md)
- [RAG multi-machine runbook](value-write-rag-practicality/RUNBOOK.md)
- [Frozen RAG extension protocol](value-write-rag-practicality/FROZEN_EXTENSION_PROTOCOL.md)

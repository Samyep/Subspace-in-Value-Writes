# Source-disjoint frozen-direction interventions

This directory contains the complete six-model experiment used to test whether a value-write direction fitted on one set of source questions causally affects support selection on different source questions. It complements the prompt-specific interventions in the paper with a frozen shared direction and an approximately magnitude-matched random-displacement control.

The split, model revisions, writer windows, random seeds, numerical rule, and source-clustered analysis were fixed before the complete run. Source ranks 1--200 supply 800 fitting prompts, while ranks 201--400 supply 800 evaluation prompts. No source question occurs on both sides. The first component is selected by variance alone, without using gold labels to choose the component or its sign. Interventions are evaluated only on prompts that the unmodified model answers correctly.

## Main result

`PC1` is the fraction of baseline-correct prompts whose selected support pair becomes incorrect after removing the frozen component. `Random` averages three fixed random-direction replicates that target the same applied edit magnitude at every layer and decoding step. Confidence intervals resample the 200 evaluation source questions and keep prompt variants and random replicates together.

| Model | Baseline-correct prompts | PC1 | Random | PC1 - Random [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| Llama-3.2-3B | 217 | 0.028 | 0.014 | 0.014 [-0.009, 0.038] |
| Llama-3.1-8B | 401 | 0.379 | 0.061 | 0.318 [0.240, 0.398] |
| Qwen2.5-7B | 521 | 0.106 | 0.006 | 0.100 [0.058, 0.146] |
| Qwen2.5-14B | 636 | 0.071 | 0.018 | 0.053 [0.023, 0.086] |
| Mistral-7B-v0.3 | 287 | 0.014 | 0.022 | -0.008 [-0.028, 0.013] |
| Mistral-Nemo-12B | 458 | 0.094 | 0.030 | 0.064 [0.027, 0.105] |

The frozen-axis intervention exceeds the random comparator in four of six models. The Llama-3B and Mistral-7B comparisons are inconclusive. This experiment supports a model-dependent causal effect on controlled support-label selection. It does not establish the same effect in every model or identify a uniquely semantic mechanism.

## Included artifacts

- `experiments/PROTOCOL_v4.md`: final protocol, including the prespecified handling of BF16-representability exceptions.
- `experiments/frozen_direction_intervention_v4.py`: exact runner used for the released six-model outputs.
- `experiments/aggregate_frozen_controls_v4.py`: exact source-clustered aggregation and sensitivity analysis.
- `experiments/frozen_direction_intervention.py` and `experiments/PROTOCOL_pre_numerical_amendment.md`: frozen predecessor retained so the numerical amendment can be audited.
- `data/hotpot_400_prompts.jsonl`: all 400 source questions and four controlled prompt variants per source.
- `results/frozen_direction/main_v4/<model>/per_prompt.jsonl`: all 800 evaluation records per model, including baseline, PC1, and three random outcomes where eligible.
- `results/frozen_direction/main_v4/<model>/frozen_pc1.npz`: the fitted direction for each model.
- `results/frozen_direction/aggregate_v4/aggregate.json`: point estimates, source-bootstrap intervals, error-type splits, replicate rates, and worst-case reassignment bounds.
- `results/frozen_direction/aggregate_v4/numerical_audit.json`: site-level norm-matching diagnostics aggregated over all six models.
- `results/frozen_direction/norm_diagnostic/`: captured tensor and endpoint certificate for the BF16 representability case that motivated V4.
- `audits/`: independent reconstructions of the numerical diagnostics and scientific estimates from the archived records.

The per-prompt records, directions, summaries, manifests, rendered-prompt metadata, numerical certificates, and aggregate outputs are included. The approximately 480 MB of per-prompt fitting-vector caches are omitted because they are deterministic intermediate activations that the runner reconstructs from the released data and pinned checkpoints. Model weights are not included.

## Validate the released bundle

From this directory, run:

```bash
python scripts/validate_bundle.py
```

The validation checks source disjointness, all six per-prompt files, aggregate inputs, released estimates, and independent audit totals.

## Environment

The completed run used Python 3.12.9, PyTorch 2.10.0+cu129, Transformers 5.11.0, and NumPy 2.4.6 on one NVIDIA GH200 GPU. Install a CUDA-enabled PyTorch build for the target machine, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ../value-write-rag-practicality
```

The runner uses the `vw_rag` model loader and value-write extractor from the sibling RAG experiment already present in this repository. The exact helper files used by the controlled support-selection task are retained under `dependencies/`.

The six checkpoints are pinned in `experiments/frozen_direction_intervention_v4.py`. They must already be present in the Hugging Face cache because the frozen runner resolves them with `local_files_only=True`. Use the exact model IDs and revisions in `CELLS` when populating a new cache.

## Reproduce the run

Use a fresh output directory so the released records remain unchanged:

```bash
export HF_HOME=/path/to/huggingface/cache
export TOKENIZERS_PARALLELISM=false
bash scripts/run_all_six.sh
```

`scripts/run_all_six.sh` first runs the saved-tensor GPU regression, then evaluates all six models sequentially, and finally regenerates the numerical audit and scientific aggregate under `results/frozen_direction/reproduction_v4/`. A generic one-GPU Slurm wrapper is provided at `slurm/run_all_six.slurm`; supply site-specific account and partition options through `sbatch`.

For a short pipeline check, run one to ten sources per split with the same explicit paths used in the script:

```bash
python experiments/frozen_direction_intervention_v4.py \
  --cell llama_3b \
  --smoke 2 \
  --output results/frozen_direction/smoke_v4/llama_3b \
  --experiment-root dependencies/controlled_task \
  --rag-root ../value-write-rag-practicality \
  --legacy-root dependencies/legacy \
  --data data/hotpot_400_prompts.jsonl \
  --protocol-file experiments/PROTOCOL_v4.md
```

The experiment used three fixed random directions per eligible prompt. It is deterministic conditional on the pinned checkpoints, recorded software path, device arithmetic, and cached tokenizer assets.

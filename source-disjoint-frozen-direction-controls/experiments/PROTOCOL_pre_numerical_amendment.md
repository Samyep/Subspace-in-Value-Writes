# Path 6 additional intervention controls: fixed protocol

Recorded before running the new control outcomes on 2026-09-30.

The user required the original six-model cohort before any GPU submission or new inference. This supersedes the initial three-cached-model draft. The operators, source split and statistical plan are unchanged.

## Question and scope

Does a direction fitted on different source questions affect controlled support-label selection more than equally large random displacements? Are the resulting errors distinct incorrect fact pairs or repeated labels?

This is an additional source-disjoint evaluation on previously studied source questions, not a newly untouched benchmark. It neither replaces the existing six-model experiments nor establishes unrestricted answer semantics. No component, window, model or sample selection will be changed based on the new outcome.

## Models and sources

- Exactly the original six models: Llama-3.2-3B-Instruct, Hermes-3-Llama-3.1-8B, Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, Mistral-7B-Instruct-v0.3, and Mistral-Nemo-Instruct-2407.
- Zero-based writer windows are 14–17, 14–24, 24–27, 32–47, 19–22 and 19–24 respectively. They are unchanged from the historical six-model experiment.
- The Llama-3B weights are loaded from the cached unsloth mirror after verifying both weight shards against the official Meta metadata. Record any tokenizer/config differences and all exact snapshot commits. The other five checkpoints use the original repositories. No outcome-based model substitutions.
- Source: original `custom_outputs/confirmatory/hotpot_400_prompts.jsonl`; select `active_set` and `working_set`, both lexical packs.
- Fit on ranks 1–200 (200 sources, 800 prompts); evaluate on ranks 201–400 (200 different sources, 800 prompts). Assert zero source-ID overlap and four variants per source before fitting.
- Re-render with each tokenizer's chat template. Apply the inherited 768-token input limit consistently and report any exclusions and counts. No correctness-based fitting filter.

## Frozen direction and operators

1. Extract fact-text-only value writes at the final input token with eager attention and normalized attention inputs. Center the six fact vectors within each prompt.
2. Fit one global variance-selected PC1 to the concatenated centered fitting vectors. No gold labels select a component or sign. Store the direction before reading evaluation outcomes. Use the same unit axis across all held-out prompts, writer layers and decoding steps.
3. Decode two A–F labels with the original restricted greedy decoder. The second pass conditions on the first predicted label followed by a comma. Store all baseline predictions.
4. For every baseline-correct evaluation prompt, remove the frozen PC1 projection from the full attention output at the final position of every writer layer, at both decoding steps. Cache the actually applied displacement norm after casting to the model dtype at each layer and step.
5. Run three fixed random-displacement replicates. Each has an isotropic unit vector derived from a stable deterministic seed of the prompt ID and replicate, held fixed across layers/steps. At each layer/step use the recorded PC1 norm. Report the requested and achieved norms after model-dtype rounding. This is an equal-norm random displacement, not an orthogonal random projection. It matches edit size, not the resulting output norm or downstream state.
6. Add no further components, layer sweeps or tuned amplitudes after inspecting outcomes. A small fixed smoke test checks implementation and uses a separate output directory. Never merge smoke rows into full results.

## Outcomes and statistics

- Primary: frozen-PC1 destruction minus mean destruction across the three random-displacement replicates, on the common baseline-correct cohort.
- Also report both rates, clean accuracy, eligible prompt/source counts and the three individual random-replicate rates.
- Exhaustive output categories: correct; incorrect pair of two distinct labels (one gold or zero gold); repeated label. Report category rates over all baseline-correct prompts, not only within errors.
- Source-clustered paired bootstrap with 20,000 draws and seed 29911. Keep all four variants and fixed random replicates together, recomputing the conditional rate on every resample. The intervals condition on the fitted direction and recorded random seeds. They are per-model intervals, not simultaneous bounds.
- Inspect actual norm-matching errors and intervention coverage; implementation failures are not scientific results. Retain all six complete model outcomes and all predeclared comparisons. No outcome-driven stopping or selection of only positive cells.

## Artifacts

Preserve input/source manifests, model snapshots, direction/checkpoint metadata, exact code and protocol snapshots, predictions, per-layer norm checks, fixed seeds, raw JSONL and aggregation code under Path 6. Path 5 remains unchanged. Whether positive, negative or inconclusive, the completed control suite informs the manuscript's claim boundary; it must not be silently filtered for favorable outcomes.

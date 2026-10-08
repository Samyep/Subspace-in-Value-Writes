# Path 6 additional intervention controls: numerical amendment V4

Recorded on 2026-09-30 after partial inference under the original numerical rule, before reviewing comparative effect estimates. This is a documented numerical amendment, not a claim of preregistration before any inference. The original `PROTOCOL.md`, runner and all complete/partial outputs are retained unchanged.

The user required the original six-model cohort before any GPU submission or new inference, superseding the initial three-cached-model draft. V4 keeps that cohort, the source split, PC1 operator and primary contrast fixed. It adds the stated numerical fallback and flagged-outcome sensitivity analysis.

## Question and scope

Does a direction fitted on different source questions affect controlled support-label selection more than random displacements targeting the same magnitude? Are the resulting errors distinct incorrect fact pairs or repeated labels?

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
5. Run three fixed random-displacement replicates. Each has an isotropic unit vector derived from a stable deterministic seed of the prompt ID and replicate, held fixed across layers/steps. At each layer/step target the recorded PC1 norm. Retain the original candidate whenever its applied BF16 displacement satisfies the original combined rule: absolute norm error at most max(0.005 times target edit norm, 1e-6 times the current control attention-output norm). When the original search fails that rule, search nonnegative FP32 scalar amplitudes to adjacent representable values whose actual displacement norms bracket the target. Evaluate both on the GPU and choose the smaller absolute norm error, choosing the lower amplitude on an exact tie. The fixed random vector is never resampled or changed coordinate by coordinate. Record the bracket, norms, gap, selected amplitude, actual errors and whether the original rule is met. If neither endpoint meets the original rule, retain this certified nearest value and flag the site as quantization-limited. Do not claim that nearest-representable automatically means adequate scientific matching. The pre-rounding vector is isotropic; BF16 rounding can change its realized orientation. This is a random displacement targeting the treatment's edit magnitude, not a random projection or a norm-preserving edit.
6. Add no further components, layer sweeps or tuned amplitudes after inspecting outcomes. A fixed smoke test and the captured numerical regression check implementation in separate directories. Never merge diagnostic or smoke rows into full results. The amended full execution uses a fresh output directory for all six models; do not silently mix old and new identities.

## Outcomes and statistics

- Primary: frozen-PC1 destruction minus mean destruction across the three random-displacement replicates, on the common baseline-correct cohort.
- Also report both rates, clean accuracy, eligible prompt/source counts and the three individual random-replicate rates.
- Exhaustive output categories: correct; incorrect pair of two distinct labels (one gold or zero gold); repeated label. Report category rates over all baseline-correct prompts, not only within errors.
- Source-clustered paired bootstrap with 20,000 draws and seed 29911. Keep all four variants and fixed random replicates together, recomputing the conditional rate on every resample. The intervals condition on the fitted direction and recorded random seeds. They are per-model intervals, not simultaneous bounds.
- Inspect actual norm-matching errors and intervention coverage; implementation failures are not scientific results. Before reading comparative effect estimates from the amended run, produce a numerical-only audit for all six models and obtain an independent assessment of whether the matching is adequate for the proposed wording. Inspect local errors as well as rollout sums-of-squared-edit-norm ratios. A nearest-point certificate alone is not a quality guarantee.
- Preserve original-rule violations as numerical flags. A random rollout is flagged if any site violates the original combined bound, including its existing absolute floor. Report flagged sites, rollouts, prompts and sources by model, and all realized error distributions. Successful refined searches within the original bound are not tolerance violations.
- In addition to the unchanged primary contrast, report a prospective sensitivity analysis that treats each flagged random rollout's destruction indicator as unknown in [0,1]. The lower contrast sets all flagged random outcomes to destroyed; the upper contrast sets them all to correct. With F flagged rollouts and n baseline-correct prompts, the interval width is exactly F/(3*n). Recompute counts and both endpoint statistics in every source-cluster bootstrap draw; use the same 20,000 draws/seed as the primary analysis. Report each endpoint's percentile interval. These are bounds for reassigning only the flagged outcome indicators, not bounds on an unattainable exact-norm intervention, all numerical discrepancies, or semantic specificity. This does not justify otherwise materially mismatched controls.
- Retain all six complete model outcomes and all declared comparisons and sensitivity results. No outcome-driven stopping, model exclusion or selection of only positive cells. If numerical matching is inadequate, qualify or withhold that control claim rather than turn completion into positive evidence.

## Artifacts

Preserve input/source manifests, model snapshots, direction/checkpoint metadata, exact code and protocol snapshots, predictions, per-layer norm checks, fixed seeds, raw JSONL and aggregation code under Path 6. Path 5 remains unchanged. Whether positive, negative or inconclusive, the completed control suite informs the manuscript's claim boundary; it must not be silently filtered for favorable outcomes.

## Numerical amendment after implementation smoke

Before any full evaluation, an implementation smoke run encountered a discrete BF16 rounding step: requested edit norm0.0012223043, nearest searched norm0.0012151701, above the0.5% relative threshold. The fixed numerical floor above was independently reviewed and added to all six models. No source, direction, component, window, random vector, seed or scientific outcome criterion changed. Failed smoke outputs are retained; fresh smoke_v2 outputs are separate. The amendment responds to a numerical exception and is not chosen from accuracy comparisons.

## Tokenizer implementation amendment before full evaluation

The second smoke stopped at an incompatible private tokenizer helper before Mistral inference. The runner now loads both Mistral tokenizers through the public initializer and verifies their native pretokenizers. The Mistral-7B Metaspace configuration and NeMo's existing canonical regex are preserved. The repair, real-prompt preflight and independent comparison of unchanged scientific operators are recorded in `TOKENIZER_IMPLEMENTATION_AMENDMENT.md` and the code review. Both Mistral GPU checks subsequently completed in a subsequent smoke run. The original six-model execution used one frozen runner version and the original protocol; no smoke outcomes entered it. V4 below is a separately identified execution.

## Certified numerical failure during the original full execution

The original full execution completed Llama-3B, Llama-8B and Qwen-7B but stopped at Qwen-14B after 230 evaluation rows. The two Mistrals subsequently completed under the unchanged original rule in a follow-up run. No complete six-model comparison was computed or reported. The failure was not an accuracy criterion.

A diagnostic rerun reproduced the exact failing site without altering any original inputs or outcomes: prompt `5a827a9f55429940e5e1a8cf::working_set_pack_b`, replicate 0, second decoding step, layer 40. Target norm 0.00476321205496788 lies between attainable norms 0.00472657335922122 and 0.004826403688639402 at adjacent FP32 amplitude bit patterns 1009352440 and 1009352441. Neither meets the original combined tolerance 0.00003429914855957031. CPU and GPU endpoint tensors are identical; FP64 diagnostic norms confirm that reduction error does not explain the gap. Certificates and protected-file checks are archived in `results/frozen_direction/norm_diagnostic/capture_qwen14b_norm_site/`.

V4 changes only the response to a certified scalar-resolution failure. It does not enlarge the original quality threshold, change the source split, refit by gold labels, choose another component/window, resample a random vector, change a seed, or choose parameters from comparative outcomes. All six models receive the same amended code in a fresh run. Numerical quality is reviewed before the new comparative estimates. The reported manuscript protocol will describe the final operator and its realized numerical limits; the artifact record preserves when and why this amendment occurred.

# Independent V4 review before the amended full run

Reviewed 2026-09-30, without inspecting comparative scientific estimates from the partial original run. This review permits execution and a numerical audit; it does not conclude that the controls are already numerically adequate or that the manuscript's scientific claims are supported.

The protocol, runner, aggregator, captured tensors, and independent checks were reviewed as one frozen bundle.

## The numerical amendment has a verified reason

I independently loaded the captured failing tensors and reconstructed both endpoint edits. Adjacent FP32 amplitudes with bit patterns 1009352440 and 1009352441 produce norms 0.00472657335922122 and 0.004826403688639402 around target 0.00476321205496788. Both violate the original combined tolerance. CPU reconstruction is byte-identical to the saved GPU tensors, and FP64 norm measurements leave the conclusion unchanged.

The lower endpoint is nearest: its norm is 0.7692% below target, with a site squared-norm ratio of 0.984675. This is a genuine fixed-vector representational gap, not merely insufficient iterations in the original search. The certificate does not predict the frequency or magnitude of exceptions across the complete six-model experiment. Independent results are in `numerical_certificate_independent_checks.json`.

## Runner fidelity

The original successful matcher is preserved verbatim under a new internal name. Fallback executes only after its original combined-bound exception. It reuses the exact already-normalized random vector from that exception, evaluates candidates on the execution device, certifies adjacent scalar values straddling the target, and chooses the nearer endpoint with a fixed lower-amplitude tie rule. It neither resamples a vector nor adjusts coordinates individually.

My AST comparison confirms unchanged source splitting, PCA fitting, projection removal, decoding, random seeds/vectors and tokenizer handling. Twenty-seven independently generated calls that passed the old matcher produce byte-identical tensors and unchanged original diagnostic fields under V4. Replaying the captured failure produces the certified lower tensor and preserves its failure flag. These checks are recorded in `v4_runner_independent_checks.json`.

`original_tolerance_passed`, `certified_fallback_used` and `quantization_limited` distinguish an old success, a refined search that meets the old bound, and a nearest endpoint that still violates it. `used_absolute_tolerance` is correctly false for the last case. Per-site bracket diagnostics and per-rollout quality counts remain available.

## Protocol and aggregation

The protocol accurately states that the numerical amendment follows partial inference and precedes review of comparative effect estimates. It retains all six models, source assignments, PC1 operation, seeds and the primary contrast; it separately declares the fallback and prospective sensitivity analysis. Fresh V4 outputs must not be merged with original-run or smoke outputs.

The numerical-only stage evaluates all six completed models before computing intervention-effect summaries. It recomputes tolerance status from actual norms, checks bracket adjacency, scalar representations, target straddling, nearest-endpoint selection and tie behavior, validates the saved gap and chosen norm, and checks flag counts against model summaries. Both local errors and rollout energy ratios are reported.

I identified and reported an initial aggregation bug: a site violating both bounds would have been incorrectly counted as use of the absolute floor. It is corrected in the reviewed aggregator. The rule is now `passes_original_bound AND exceeds_relative_bound`, consistent with the runner.

The sensitivity statistic sets every flagged random rollout to destroyed for the lower contrast and to correct for the upper contrast. It keeps all prompts, uses the same baseline-correct denominator, and groups sources identically to the primary paired bootstrap. Its width is exactly the number of flagged rollouts divided by three times the eligible prompt count. Flagged outcomes do not select models or prompt exclusions. Endpoint intervals are not simultaneous six-model guarantees or bounds on an ideal exact-norm experiment.

I independently ran all thirteen V4 CPU tests: seven runner tests and six aggregation tests passed. The latter include exact source-resampling checks for the reassignment endpoints. These tests and the certificate support implementation correctness; neither supplies a scientific treatment effect.

## Remaining requirement before scientific interpretation

There is no unresolved methodological blocker to the fresh six-model execution under this explicitly amended protocol. Numerical adequacy is still an open question. The fallback intentionally continues even when its closest possible match is poor; a unit test verifies that such a case is honestly flagged rather than silently called matched. Therefore the complete numerical-only report must be independently assessed before comparative effects are opened.

Inspect flag counts and affected sources, maximum local discrepancies, target sizes, absolute/output-relative errors and rollout energy ratios. Do not judge adequacy solely from the small size of the currently known failure or from total energy. If discrepancies are material, qualify or withhold the magnitude-controlled claim, even if the subsequent worst-case outcome reassignment happens to stay positive. No global/local-axis or semantic-specific claim boundary changes merely because V4 can finish.

## Addendum: frozen prelaunch aggregator

The final aggregator matches the prelaunch implementation apart from the three declared reporting changes. Reversing those changes in memory recovered the previously reviewed implementation. No frozen intervention or scientific output was modified. The check is recorded in `v4_aggregator_addendum_checks.json`.

The two numerical diagnostic corrections are appropriate: output-relative error now uses the runner's denominator floor instead of reporting zero when the output norm is zero; an all-zero target and achieved schedule contributes the declared ratio convention 1 rather than disappearing from the rollout distribution. Neither changes an intervention, scientific outcome, or bootstrap estimate. All six aggregation tests passed again.

The third LaTeX table correctly draws its point range from the worst-case reassignment endpoints and its displayed confidence interval from the lower-endpoint bootstrap distribution. The statistical tests correctly retain source clustering, common eligible denominators, original-bound flags and width `F/(3*n)`. The exact 27-resample test uses inverse-CDF quantiles of the finite resampling distribution, which is appropriate.

One manuscript-reporting requirement remains: the generated table currently shows only the lower-endpoint CI, while the protocol specifies reporting intervals for both endpoints. Both are correctly computed and retained in `aggregate.json`. Include the upper-endpoint interval in the final manuscript or supplementary reporting when those artifacts are prepared. Define `F` as flagged random rollouts, identify the intervals as 95%, and distinguish the reassignment point range from a confidence interval. This is a presentation requirement, not a reason to change the frozen calculation or inspect effects before the numerical audit.

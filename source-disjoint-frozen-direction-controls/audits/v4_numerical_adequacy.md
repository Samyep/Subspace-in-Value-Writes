# Independent numerical adequacy review of the complete V4 suite

Reviewed 2026-09-30 before inspecting comparative intervention effects. **Stage-specific judgment: the realized numerical matching is adequate to proceed with the declared approximately magnitude-matched comparison, provided the local exceptions and prospective reassignment sensitivity analysis are reported.** This judgment concerns numerical quality only. It neither establishes a scientific treatment effect nor supports exact matching, preserved orientation, or semantic specificity.

## Independent verification

I checked all six manifest identities against the frozen V4 code and protocol. Each model retains 800 fitting and 800 evaluation prompts from 200 source questions per split, with no source-ID overlap or length exclusions. I independently recomputed numeric diagnostics from 128,610 intervention-site records in 7,560 random rollouts. I accessed IDs and perturbation metadata, rather than prediction/category fields or comparative effect summaries. High-precision summation reproduces rollout energy ratios up to ordinary final summation rounding.

The independent output, `reviews/v4_numerical_independent_checks.json`, contains per-model manifest and retention checks, every floor-only site, both original-bound failures, their local scales, and aggregate checks. This review does not repeat model forwards; applied norms are audited from the recorded numeric measurements and saved certificates.

## Local matching quality

Relative norm errors below are percentages of the requested edit norm, not percentages of the activation norm. “Floor-only” means a site exceeds 0.5% relative error but meets the original absolute floor; “Flagged” means it violates even that combined rule.

| Model | Sites | Median error (%) | 99th percentile (%) | Maximum (%) | Floor-only sites | Flagged sites |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3B | 5,208 | 0.01372 | 0.37746 | 0.49806 | 0 | 0 |
| Llama-8B | 26,466 | 0.02126 | 0.42219 | 2.59646 | 7 | 0 |
| Qwen-7B | 12,504 | 0.01406 | 0.41015 | 4.02463 | 8 | 0 |
| Qwen-14B | 61,056 | 0.01166 | 0.41445 | 5.47807 | 20 | 2 |
| Mistral-7B | 6,888 | 0.00883 | 0.04814 | 0.08545 | 0 | 0 |
| Mistral-12B | 16,488 | 0.01292 | 0.43503 | 1.44490 | 6 | 0 |

Thus 99.9666% of sites meet the relative 0.5% criterion. Forty-one additional sites meet the predeclared absolute floor, and two require certified nearest-representable exceptions. There are no zero target norms.

The larger relative maxima occur at small targets rather than large absolute mismatches. The largest floor-only relative error is Qwen-14B's target 0.0002104565 versus applied norm 0.0001989275. Its 5.4781% mismatch is an absolute difference of 0.0000115290, or approximately 0.0000003111 of that site's attention-output norm. Across all floor-only sites, the largest error/output-norm ratio is 0.0000009023, below the declared 0.000001 floor. All their target norms are less than 0.0002 of the corresponding output norm, as required for the floor to relax the relative criterion.

The 41 floor-only sites occur in 41 separate rollouts: 7 for Llama-8B, 8 for Qwen-7B, 20 for Qwen-14B and 6 for Mistral-12B. They cover 4, 4, 14 and 3 prompts respectively. They are within the original combined criterion and are not the failures flagged by the prospective reassignment analysis. Their larger relative maxima must nevertheless remain visible in the numerical reporting.

## Both original-bound exceptions

Both occur in Qwen-14B, at writer layer 40, in two different prompts/sources and two different rollouts. The recorded scalar amplitudes are adjacent FP32 values, bracket the target on GPU, and choose the nearest attainable norm. FP64 diagnostic measurements confirm that neither endpoint meets the original bound.

| Site | Target norm | Applied norm | Signed relative error | Absolute error | Error / output norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Second decoding step, replicate 0 | 0.0047632121 | 0.0047265734 | −0.7692% | 0.0000366387 | 0.0000010682 |
| First decoding step, replicate 0 | 0.0049215276 | 0.0049645617 | +0.8744% | 0.0000430341 | 0.0000015546 |

One exception falls below its target and the other above it; neither is a large relative magnitude failure. These observations support retaining them as disclosed approximations, not marking them as exact successes. Their maximum possible influence under the predeclared flagged-outcome reassignment is an interval width of `2/(3 × 636) = 0.001048218`, or 0.10482 percentage points, for Qwen-14B. This width follows from numeric flags and eligible counts alone and uses no scientific outcomes. Its actual endpoints and both endpoint confidence intervals remain to be computed and reported.

## Rollout diagnostics and limits

All rollout sums-of-squared-edit-norm ratios range from 0.9995087 to 1.0008970. This agrees with close aggregate matching, but it is not the sole reason for the adequacy judgment: the local exceptions above were separately inspected. A small total-energy difference cannot guarantee that an individual intervention is behaviorally negligible. Likewise, a small activation-relative error is a scale diagnostic, not a theorem that the model's label choice cannot change.

The original absolute-floor cases and the two certified exceptions are sparse and locally small in absolute scale. They do not resemble gross edit-size mismatches hidden by aggregate statistics. Their realized pattern is consistent with the intended approximate-control design, with its declared limits. The numerical report therefore permits scientific aggregation; it does not preapprove any interpretation of the resulting effects.

## Required wording and disclosures

Use **“random displacements targeting the same edit magnitude”** or **“approximately magnitude-matched random displacements.”** Do not claim that all sites match within 0.5%, that energy is exactly equal, that the resulting activation norms are preserved, or that the realized BF16 displacements retain exactly isotropic orientation.

A concise methods sentence can say:

> We compare projection removal with random displacements targeting the same applied norm at each layer and decoding step. We record numerical matching errors and evaluate the sensitivity of the comparison to the two rollouts requiring nearest-representable exceptions.

The appendix should report per-model relative-error medians, 99th percentiles and maxima, together with floor-only and original-bound-failure counts. State the combined tolerance and the certified nearest-endpoint rule. Report both Qwen-14B reassignment endpoints and both confidence intervals after aggregation; distinguish that range from a confidence interval. These sensitivity bounds concern only reassignment of flagged random outcomes, not an ideal exact-norm intervention or all numerical differences.

All six scientific outcomes must still be retained and interpreted under the existing global-versus-local-axis and support-label-selection claim boundaries. No acceptance-strength or mechanism conclusion is made at this stage.

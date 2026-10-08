# Separate v4 implementation

`frozen_direction_intervention_v4.py` is a separate copy of the frozen runner. It does not modify that runner or its existing results. Its default protocol file is `PROTOCOL_v4.md`, and all six models must use fresh `main_v4` outputs.

The source split, fitting procedure, model snapshots, writer windows, restricted decoder, PC1 intervention, random seeds and fixed random directions are unchanged. CPU AST checks compare these functions directly with the old runner.

## Numerical behavior

The old matcher body is retained verbatim under `_legacy_equal_norm_displacement`. Every successful old call returns exactly the same tensor and original metadata values; v4 only appends explicit quality flags.

Only an original combined-rule failure invokes `_certify_nearest_quantized_edit`. It reuses the exact normalized vector from the failed function frame, without renormalization, reseeding, coordinate correction, or refitting. It evaluates the same FP32 multiply/add followed by the model-dtype cast on the current execution device. A binary search over ordered positive FP32 bit patterns finds adjacent amplitudes whose actual displacement norms straddle the target. The nearer norm is used; an exact tie selects the lower endpoint.

This can continue through a genuine BF16 gap. Such an edit is **not represented as meeting the old tolerance**. `original_tolerance_passed=false` and `quantization_limited=true` preserve the numerical violation. `used_absolute_tolerance` is false if the chosen edit violates even the old absolute floor. If certification merely resolves a search-resolution issue and the nearest edit meets the old rule, `certified_fallback_used=true` but `quantization_limited=false`.

Per-site records retain both endpoint amplitudes and bit patterns, FP32/FP64 norms, actual errors, the straddling assertion, norm gap, execution device, tie rule and original criterion. Per-rollout records report original-rule success and counts of certified and quantization-limited sites. Summaries report the same counts across sites and rollouts. Existing requested/achieved displacement-energy totals remain available for the independent numerical audit.

The method is therefore an approximate norm control with explicitly audited exceptions. Neither nearest-endpoint selection nor successful certification establishes that a large norm mismatch is scientifically acceptable. Numerical quality across all six models must be assessed before interpreting their effects.

## Validation and execution

```bash
python -m unittest -v test_frozen_direction_intervention_v4.py
python frozen_direction_intervention_v4.py --cell qwen_14b --output ../results/frozen_direction/main_v4/qwen_14b
```

Seven CPU tests cover the actual captured GPU failure, the same certified lower endpoint and tensor values, an exactly representable edit, unchanged original successes, no reseeding or vector mutation, explicit rejection flags for a large mismatch, and the fixed lower tie rule. The unchanged scientific functions and original matcher body are also compared structurally. These checks do not replace GPU review of the v4 fallback or the complete all-model numerical audit.

The original failed execution, its diagnostic tensors and its GPU endpoint certificate remain separate under `results/frozen_direction/`. No old and v4 scientific records should be combined.

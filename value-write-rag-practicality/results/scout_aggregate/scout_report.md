# Truncated 3B scout results

The scout retains Llama-3.2-3B blocks 0--17. Value-2 and Attention-2 come from the same forward pass.

## Selection checks

- Value top-2 agreement with the released 3B run: 95.25%.
- Attention top-2 agreement with the released 3B run: 94.75%.
- Maximum absolute Value score difference from the released run: 0.097984647.
- Maximum absolute Attention score difference from the released run: 0.29845434.
- Same-runtime full/truncated Value top-2 agreement: 100.00%.
- Same-runtime full/truncated Attention top-2 agreement: 100.00%.
- Same-runtime maximum absolute Value score difference: 0.
- Same-runtime maximum absolute Attention score difference: 0.

## Answer quality

| Reader | Method | EM | F1 | Support recall@2 | Prompt/full tokens |
|---|---|---:|---:|---:|---:|
| Qwen2.5-14B | Full-20 | 21.25 | 31.10 | -- | 1.000 |
| Qwen2.5-14B | BM25-2 | 8.50 | 17.95 | 37.62 | 0.111 |
| Qwen2.5-14B | BGE-2 | 18.00 | 26.57 | 49.75 | 0.124 |
| Qwen2.5-14B | 3B Attention-2 | 19.75 | 27.73 | 38.38 | 0.143 |
| Qwen2.5-14B | 3B Value-2 | 18.25 | 27.66 | 43.62 | 0.126 |
| Qwen2.5-14B | Oracle-2 | 27.75 | 36.83 | 63.62 | 0.114 |
| Qwen3-8B | Full-20 | 15.75 | 27.38 | -- | 1.000 |
| Qwen3-8B | BM25-2 | 5.25 | 16.14 | 37.62 | 0.112 |
| Qwen3-8B | BGE-2 | 13.00 | 22.97 | 49.75 | 0.126 |
| Qwen3-8B | 3B Attention-2 | 15.50 | 24.64 | 38.38 | 0.145 |
| Qwen3-8B | 3B Value-2 | 16.50 | 25.78 | 43.62 | 0.128 |
| Qwen3-8B | Oracle-2 | 22.25 | 33.67 | 63.62 | 0.115 |

## Paired contrasts

| Reader | Contrast | EM points (95% CI) |
|---|---|---:|
| Qwen2.5-14B | 3B Value-2 minus Full-20 | -3.00 [-6.25, 0.25] |
| Qwen2.5-14B | 3B Value-2 minus 3B Attention-2 | -1.50 [-3.75, 0.75] |
| Qwen2.5-14B | 3B Value-2 minus BGE-2 | 0.25 [-3.25, 3.75] |
| Qwen3-8B | 3B Value-2 minus Full-20 | 0.75 [-2.00, 3.50] |
| Qwen3-8B | 3B Value-2 minus 3B Attention-2 | 1.00 [-1.00, 3.00] |
| Qwen3-8B | 3B Value-2 minus BGE-2 | 3.50 [0.25, 6.75] |

## End-to-end time

Ratios include selection plus answer generation and exclude the first five warm-up questions.

| Reader | Method | Seconds | Time/full ratio | Paired n |
|---|---|---:|---:|---:|
| Qwen2.5-14B | Full-20 | 0.506 | 1.000 | 395 |
| Qwen2.5-14B | BM25-2 | 0.507 | 1.174 | 395 |
| Qwen2.5-14B | BGE-2 | 0.496 | 1.136 | 395 |
| Qwen2.5-14B | 3B Attention-2 | 0.571 | 1.313 | 395 |
| Qwen2.5-14B | 3B Value-2 | 0.546 | 1.245 | 395 |
| Qwen3-8B | Full-20 | 0.417 | 1.000 | 395 |
| Qwen3-8B | BM25-2 | 0.397 | 1.138 | 395 |
| Qwen3-8B | BGE-2 | 0.420 | 1.179 | 395 |
| Qwen3-8B | 3B Attention-2 | 0.497 | 1.406 | 395 |
| Qwen3-8B | 3B Value-2 | 0.473 | 1.344 | 395 |

Scout totals add one shared truncated-Llama scoring pass to selected-context generation. BGE totals add cross-encoder scoring to generation. Direct wall-time comparisons require matching GPU and software signatures, which are recorded in the JSON report.

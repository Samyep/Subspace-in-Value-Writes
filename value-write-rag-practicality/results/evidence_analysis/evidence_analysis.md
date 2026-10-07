# Evidence-role and disagreement analysis

All evidence roles come from MuSiQue `question_decomposition` annotations. The first decomposition support is the first hop and the last support is the terminal hop.

## Main results

| Reader | Selector | EM | Support recall@2 | First-hop hit | Terminal-hop hit | Both supports |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B | VALUE-2 | 12.25 | 35.88 | 43.00 | 28.75 | 10.50 |
| Qwen2.5-7B | BGE-2 | 13.50 | 49.75 | 79.00 | 20.50 | 15.50 |
| Qwen2.5-14B | VALUE-2 | 24.00 | 40.38 | 47.75 | 33.00 | 12.75 |
| Qwen2.5-14B | BGE-2 | 18.00 | 49.75 | 79.00 | 20.50 | 15.50 |
| Qwen3-8B | VALUE-2 | 17.25 | 43.38 | 55.00 | 31.75 | 15.75 |
| Qwen3-8B | BGE-2 | 13.00 | 49.75 | 79.00 | 20.50 | 15.50 |
| Llama-3.2-3B | VALUE-2 | 15.00 | 43.12 | 59.00 | 27.25 | 15.00 |
| Llama-3.2-3B | BGE-2 | 13.25 | 49.75 | 79.00 | 20.50 | 15.50 |

The first hop is available in 348/400 retrieved pools and the terminal hop in 161/400. Conditional retention therefore separates selector behavior from first-stage retrieval availability.

| Reader | Selector | First hit / available | Terminal hit / available | Mean top-2 overlap |
|---|---|---:|---:|---:|
| Qwen2.5-7B | VALUE-2 | 49.43 | 71.43 | 39.12 |
| Qwen2.5-7B | BGE-2 | 90.80 | 50.93 | 39.12 |
| Qwen2.5-14B | VALUE-2 | 54.89 | 81.99 | 46.62 |
| Qwen2.5-14B | BGE-2 | 90.80 | 50.93 | 46.62 |
| Qwen3-8B | VALUE-2 | 63.22 | 78.88 | 50.62 |
| Qwen3-8B | BGE-2 | 90.80 | 50.93 | 50.62 |
| Llama-3.2-3B | VALUE-2 | 67.82 | 67.70 | 51.25 |
| Llama-3.2-3B | BGE-2 | 90.80 | 50.93 | 51.25 |

## Answer disagreements

| Reader | Both correct | Value only | BGE only | Both wrong |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 32 | 17 | 22 | 329 |
| Qwen2.5-14B | 59 | 37 | 13 | 291 |
| Qwen3-8B | 37 | 32 | 15 | 316 |
| Llama-3.2-3B | 36 | 24 | 17 | 323 |

## Shared-question macro contrasts

Positive values favor reader-specific Value-2.

| Metric | Mean difference | 95% CI |
|---|---:|---:|
| EM | 2.69 | [0.31, 5.12] |
| Support recall@2 | -9.06 | [-11.34, -6.75] |
| First-hop retention | -27.81 | [-31.50, -24.19] |
| Terminal-hop retention | 9.69 | [6.38, 13.06] |

## Conditional EM

EM rises sharply when the terminal hop is retained. Full conditional tables by support-hit count and evidence role are stored in the JSON summary and the per-example file.

# Practical RAG follow-up results

## Strong reranker comparison

| Reader | Full-20 EM | Value-2 EM | Attention-2 EM | Reranker-2 EM | Value - reranker |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 10.50 | 12.25 | 12.50 | 13.50 | -1.25 |
| Qwen2.5-14B | 21.00 | 24.00 | 23.25 | 18.00 | +6.00 |
| Qwen3-8B | 15.75 | 17.25 | 13.50 | 13.00 | +4.25 |
| Llama-3.2-3B | 9.25 | 15.00 | 14.25 | 13.25 | +1.75 |
| Macro | 14.12 | 17.12 | 15.88 | 14.44 | +2.69 |

## End-to-end cost audit

- Qwen2.5-7B: Value-2/Full-20 time ratio 2.00; Reranker-2 component/Full-20 ratio 1.26; n=395.
- Qwen2.5-14B: Value-2/Full-20 time ratio 2.02; Reranker-2 component/Full-20 ratio 1.32; n=395.
- Qwen3-8B: Value-2/Full-20 time ratio 1.96; Reranker-2 component/Full-20 ratio 1.23; n=395.
- Llama-3.2-3B: Value-2/Full-20 time ratio 1.46; Reranker-2 component/Full-20 ratio 1.18; n=395.

Value/attention totals sum the reader scoring pass and selected-context generation. Reranker totals are a component sum across the reranker and reader runs; compare wall time directly only when hardware metadata match.

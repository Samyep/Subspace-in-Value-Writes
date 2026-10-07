# Data and Models

## Data

The prompt file is derived from the Hugging Face dataset
`hotpotqa/hotpot_qa`. The released evaluation file contains 200 source
examples, source-example ranks 201-400, rendered under two prompt families
(`active_set`, `working_set`) and two lexical packs (`pack_a`, `pack_b`).
This gives 800 prompt rows.

The repository does not redistribute model weights.

## Main Model Cells

| Cell | Hugging Face model | Writer layers |
|---|---|---|
| `llama_3b` | `meta-llama/Llama-3.2-3B-Instruct` | L14-L17 |
| `llama_8b` | `NousResearch/Hermes-3-Llama-3.1-8B` | L14-L24 |
| `qwen_7b` | `Qwen/Qwen2.5-7B-Instruct` | L24-L27 |
| `qwen_14b` | `Qwen/Qwen2.5-14B-Instruct` | L32-L47 |
| `mistral_7b` | `mistralai/Mistral-7B-Instruct-v0.3` | L19-L22 |
| `mistral_12b` | `mistralai/Mistral-Nemo-Instruct-2407` | L19-L24 |

Some checkpoints require accepting model terms on Hugging Face before reruns.

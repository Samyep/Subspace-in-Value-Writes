# Frozen context-scaling and hybrid protocol

This protocol was fixed before running the extension experiments.  It adds no
new scout model, layer window, or direction search.

## Evidence-role analysis

The frozen 400-question MuSiQue evaluation set is joined to the original
`question_decomposition` annotations.  For every two-hop question, the support
paragraph used by the first decomposition step is the first-hop evidence and
the support paragraph used by the last decomposition step is the terminal-hop
evidence.  The last decomposition answer must match the final answer on all
400 questions.

Reader-specific Value-2 and BGE-2 are compared with:

- answer-correctness disagreement counts;
- top-2 overlap;
- annotated support recall and exact two-support recovery;
- first-hop and terminal-hop retention, both unconditional and conditional on
  the corresponding paragraph being present in the retrieved pool; and
- EM conditional on support-hit count and retained evidence role.

Question-level bootstrap intervals use 20,000 draws.  Macro intervals average
the two or four reader values within each question before resampling questions;
reader-question pairs are never treated as independent observations.

## Long-context retrieval pool

The source corpus is the frozen set of 101,958 unique MuSiQue train and
validation paragraphs.
For each evaluation question, the existing hybrid top-20 must be reproduced
exactly.  The pool is extended by appending unused documents from the same
BM25 ranking.  No document is added, removed, or reordered using evaluation
answers, support labels, or model outcomes.

The fixed retrieval depths are 20, 40, 80, and 160.  They correspond
approximately to median prompts of 2.5k, 5k, 10k, and 20k tokens.  Exact prompt
tokens and support availability are reported at every depth.

## Context-length crossover

Readers:

- `Qwen/Qwen3-8B`, with thinking disabled;
- `Qwen/Qwen2.5-14B-Instruct`.

Conditions at every retrieval depth:

1. Full-K: the reader receives every passage in the prefix;
2. BGE-2: `BAAI/bge-reranker-v2-m3` selects two passages;
3. 3B Value-scout-2: the frozen Llama-3.2-3B prefix scout, truncated after
   block 17, selects two passages with the existing frozen direction.

All readers use greedy decoding with at most 32 new tokens.  The experiment
reports EM, F1, selected support roles, reader prompt and output tokens,
selector latency, reader time to first token, remaining decode latency, reader
total latency, and selector-plus-reader end-to-end latency.  The first five
questions per shard are timing warm-up examples and remain in quality metrics.

The primary efficiency endpoint is the paired mean end-to-end latency of the
3B Value scout minus Full-K.  The crossover is the first fixed depth with a
negative point estimate.  Its paired 95% confidence interval is reported as a
measure of uncertainty.  BGE-2 is a strong semantic-selector reference.

## Frozen hybrid gate

Hybrid calibration uses 200 held-out HotpotQA full-wiki retrieval examples
immediately following the 50 examples used to fit each frozen direction.
These examples are disjoint from direction fitting and from MuSiQue.

For each question and model, Value and BGE scores are z-normalized across the
candidate passages.  A single shared weight is chosen from
`{0.25, 0.50, 0.75}` to maximize the macro support-recall@2 of Qwen3-8B and
Qwen2.5-14B.  Ties are resolved by proximity to 0.50 and then by the smaller
weight.  The score is

```
hybrid = lambda * z(Value) + (1 - lambda) * z(BGE).
```

The MuSiQue hybrid is run exactly once only if the selected interior weight
exceeds the better pure endpoint's HotpotQA macro support-recall@2 by at least
0.01.  If the gate passes, the shared weight is applied unchanged to both
MuSiQue readers at K=20 and compared with their already frozen Value-2 and
BGE-2 results.  If the gate fails, no MuSiQue hybrid generation is run and the
negative calibration result is reported.

## Stopping rule

After these analyses, the two-reader scaling experiment, and the gated hybrid,
no additional scout model, model family, layer, writer window, direction,
score transform, or evaluation subset will be tried for this extension.

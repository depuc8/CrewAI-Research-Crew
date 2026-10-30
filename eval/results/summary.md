# Evaluation results

## Retrieval (local embeddings, 20-paper corpus)

Paper recall@20 chunks (corpus-wide search): **55%**  
Finding recall (per-paper search, top 6 chunks): **65%**

| Question | Paper recall | Finding recall | Missed papers |
|---|---|---|---|
| q1-rag-hallucination | 0% | 50% | 2310.11511, 2401.15884, 2005.11401, 2312.10997 |
| q2-dense-vs-sparse | 75% | 75% | 2112.09118 |
| q3-reasoning-prompting | 100% | 100% | - |
| q4-efficient-finetuning | 50% | 50% | 2305.14314, 2101.00190 |
| q5-factuality-evaluation | 50% | 50% | 2303.08896, 2109.07958 |

## Full reviews

_Not run yet: `python -m eval.run_eval reviews --confirm-spend`._

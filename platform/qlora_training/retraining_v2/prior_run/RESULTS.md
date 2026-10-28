# QLoRA Routing Comparison

| Model | Macro-F1 | Accuracy | p95 ms | RSS MiB | GPU allocated MiB |
|---|---:|---:|---:|---:|---:|
| tfidf | 0.9497 | 95.00% | 0.76 | 42.3 | None |
| frozen | 0.9176 | 91.67% | 33.10 | 1628.6 | None |
| setfit | 0.9334 | 93.33% | 32.43 | 1629.9 | None |
| base_qwen | 0.3864 | 44.17% | 641.88 | 3171.3 | 538.1728515625 |
| qlora | 0.9418 | 94.17% | 800.38 | 3176.8 | 555.6728515625 |

## Per-Domain Recall

| Model | cloud | data | it | other | production | security |
|---|---:|---:|---:|---:|---:|---:|
| tfidf | 95.00% | 100.00% | 95.00% | 90.00% | 90.00% | 100.00% |
| frozen | 90.00% | 95.00% | 100.00% | 90.00% | 90.00% | 85.00% |
| setfit | 90.00% | 95.00% | 100.00% | 100.00% | 90.00% | 85.00% |
| base_qwen | 100.00% | 25.00% | 15.00% | 90.00% | 20.00% | 15.00% |
| qlora | 95.00% | 95.00% | 100.00% | 95.00% | 95.00% | 85.00% |

Per-case predictions and coverage: comparison.json.

Synthetic held-out challenge; not independent human-reviewed real tickets.
Original validation set has been used in previous experiments.
CPU baselines versus CUDA Qwen timings are hardware-specific, not architecture-only speedups.
Qwen uses six length-normalized candidate likelihoods, not free-generation JSON validity.
RSS is sampled total process memory; GPU allocated/reserved are separate, not additive.
Model latency excludes API, queue, retrieval and RAG generation.
No automatic deployment or claim that fine-tuning improves the model.

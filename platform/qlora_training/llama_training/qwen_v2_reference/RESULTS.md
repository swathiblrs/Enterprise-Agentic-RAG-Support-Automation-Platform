# QLoRA Routing Comparison

| Model | Macro-F1 | Accuracy | p95 ms | RSS MiB | GPU allocated MiB |
|---|---:|---:|---:|---:|---:|
| tfidf | 0.7551 | 75.00% | 0.44 | 42.4 | None |
| frozen | 0.8289 | 83.33% | 36.08 | 1630.5 | None |
| setfit | 0.8735 | 87.50% | 31.89 | 1632.1 | None |
| base_qwen | 0.4224 | 46.67% | 640.20 | 3172.7 | 538.1728515625 |
| qlora_v1 | 0.8605 | 86.67% | 777.34 | 3171.4 | 555.6728515625 |
| qlora | 0.9582 | 95.83% | 791.15 | 3170.6 | 555.6728515625 |

## Per-Domain Recall

| Model | cloud | data | it | other | production | security |
|---|---:|---:|---:|---:|---:|---:|
| tfidf | 85.00% | 90.00% | 65.00% | 65.00% | 75.00% | 70.00% |
| frozen | 90.00% | 100.00% | 95.00% | 60.00% | 90.00% | 65.00% |
| setfit | 90.00% | 100.00% | 95.00% | 70.00% | 85.00% | 85.00% |
| base_qwen | 100.00% | 45.00% | 5.00% | 80.00% | 25.00% | 25.00% |
| qlora_v1 | 95.00% | 100.00% | 95.00% | 55.00% | 100.00% | 75.00% |
| qlora | 90.00% | 100.00% | 95.00% | 100.00% | 90.00% | 100.00% |

Per-case predictions and coverage: comparison.json.

Synthetic held-out challenge; not independent human-reviewed real tickets.
Original validation set has been used in previous experiments.
CPU baselines versus CUDA Qwen timings are hardware-specific, not architecture-only speedups.
Qwen uses six length-normalized candidate likelihoods, not free-generation JSON validity.
RSS is sampled total process memory; GPU allocated/reserved are separate, not additive.
Model latency excludes API, queue, retrieval and RAG generation.
No automatic deployment or claim that fine-tuning improves the model.
Version 1 test cases are now development data, not a test metric.
Version 2 training has added boundary examples; fixed baselines have not been retrained on that augmentation.
Both adapters are re-evaluated on identical new held-out rows. Do not compare old and new test percentages directly.
Domain-only training does not establish free-generation JSON validity.

# Llama 3.2 QLoRA: Historical Synthetic Benchmark

| Model | Execution | Accuracy | Macro-F1 | p95 ms |
|---|---|---:|---:|---:|
| base_llama | new_gpu_session | 45.00% | 0.4090 | 1261.56 |
| llama_qlora | new_gpu_session | 91.67% | 0.9170 | 1282.57 |
| tfidf | cached_previous_session_not_rerun | 75.00% | 0.7551 | 0.44 |
| frozen_minilm | cached_previous_session_not_rerun | 83.33% | 0.8289 | 36.08 |
| setfit | cached_previous_session_not_rerun | 87.50% | 0.8735 | 31.89 |
| qwen3_base | cached_previous_session_not_rerun | 46.67% | 0.4224 | 640.20 |
| qwen3_qlora | cached_previous_session_not_rerun | 95.83% | 0.9582 | 791.15 |

Per-domain recall, coverage, confusion matrices and memory: comparison.json.

The 120 synthetic benchmark cases were already inspected during Qwen development.
No independently reviewed real tickets or fresh blind test are included.
References reuse verified old predictions; their latency/memory were not rerun in this session.
Fixed TF-IDF/MiniLM/SetFit artifacts were trained before the QLoRA boundary augmentation.
CPU reference and GPU LLM timings are hardware-specific, not architecture-only speedups.
Six-candidate classification timing excludes queue, API, retrieval and answer generation.
Constrained likelihood classification does not establish free-generation JSON quality.
An adapter cannot be transferred between Qwen and Llama. No active router was changed.

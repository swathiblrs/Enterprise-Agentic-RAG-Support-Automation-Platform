# Domain Routing Model

The application uses the validation-selected frozen MiniLM encoder with a trained Logistic Regression head when explicit routing policies do not identify a domain. The six classes are cloud, data, IT, other, production, and security. Other or insufficient-confidence predictions ask for clarification.

The current experiment compares TF-IDF, frozen MiniLM, and SetFit fine-tuning on the same 1,800-row grouped synthetic dataset: 1,620 training rows, 90 validation rows, and 90 final-test rows. Formatting variants appear only in training. The 1,800 rows represent 720 base scenarios across 144 families and have not been independently human-reviewed.

The frozen model achieved 100% validation macro-F1 and was selected before test evaluation. On the 90-case final test it achieved 96.67% accuracy, 96.57% macro-F1, 96.67% accepted coverage, and 97.70% accuracy among accepted predictions (85/87).

IT recall was 80% (12/15); other classes had 100% recall on their 15 examples each. This small synthetic test does not establish production reliability. TF-IDF had a slightly higher final-test macro-F1, and SetFit did not improve the experiment. Model selection was not changed after observing those test results.

Thresholds come from validation only and are uncalibrated model scores. Explicit service/security policies take precedence. A missing deployment artifact uses the original portable TF-IDF fallback with an explicit marker. Selected-runtime failures abstain.

The model is served through an isolated persistent process with local encoder weights and JSON classifier parameters. The original retrieval embeddings and ChromaDB index are separate. See [training protocol, comparison, limitations, and reproduction](routing_training/README.md) and [raw report](routing_training/reports/grouped-minilm-v1/comparison.json).

# Domain routing fallback

## Purpose

Classify requests missed by the explicit domain policy into security, production, cloud, data, IT or other. It does not override an explicit domain, active security policy, a tied policy decision or a definition-only clarification.

## Implementation

Character TF-IDF (3-5 character n-grams, sublinear term frequency, L2 normalization) and class-balanced Logistic Regression. Training uses scikit-learn in the existing support environment. Coefficients, vocabulary and IDF values are exported to JSON. Runtime inference uses the Python standard library, so the API environment needs no additional ML packages or network downloads.

The exporter asserts runtime probabilities match scikit-learn within 1e-9 on validation and test sentences. The JSON contains a dataset hash. A missing or unreadable artifact causes clarification, not a guessed domain.

## Data and selection

120 manually authored synthetic examples: 72 train, 24 validation, 24 test. Six labels have equal split counts. Texts are checked for exact duplication; this is not a full semantic leakage audit. Historical routing failures informed training examples, so old probes are development/regression data, not an independent test.

Candidate regularization values: 1, 5, 10. Candidate thresholds and score margins are selected using validation accuracy among accepted predictions and coverage, requiring at least six accepted validation examples and at least 90% accepted validation accuracy. The first attempted selection had zero accepted coverage; selection was revised based on that validation failure only. No decisions were based on test scores.

Selected C=5; score threshold=0.3; margin threshold=0.1. Scores are uncalibrated model probabilities, not probabilities that an incident is real. The small validation set does not establish that these thresholds are suitable for production.

## Results

| Split | Forced classification accuracy | Macro-F1 | Accepted coverage | Accuracy among accepted |
|---|---:|---:|---:|---:|
| Validation (24) | 75% | 75.22% | 50% | 91.67% |
| Test (24) | 83.33% | 83.57% | 58.33% | 100% (14/14) |

Do not report 100% overall accuracy: 10/24 test cases were rejected by the acceptance threshold. On the full hybrid router's test, policy plus model gives 18/24 correct domains versus 10/24 for the previous policy alone. These are small synthetic sets without independent human review.

## Reproduce

From this prototype directory:

```sh
'/Users/swathibs/Desktop/projects/Enterprise RAG Support Automation Platform/.venv/bin/python' train_domain_classifier.py
```

This replaces the local JSON model and evaluation report. Restart the API to load the new artifact. No paid APIs are used.

## Known limits

Character-level lexical similarity is not full semantic understanding. Mixed-domain incidents, novel terminology, adversarial text and negation beyond policy handling remain challenging. A larger independently labeled dataset, calibration and per-domain error review are needed before deployment. This domain classifier is separate from the existing PyTorch category/priority model in the support workflow.

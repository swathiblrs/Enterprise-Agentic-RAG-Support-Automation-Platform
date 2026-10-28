# Routing, retrieval and evidence improvements

## Before and after

| Metric | Previous local version | Improved local version |
|---|---:|---:|
| Hybrid domain accuracy on 24 synthetic test sentences | 10/24 (41.67%) | 18/24 (75%) |
| Existing 36 routing regression cases | 36/36 | 36/36 |
| Previously observed 16 wording probes | 12/16 | 16/16 |
| Support retrieval hit rate (20 questions) | 100% | 100% |
| Fixed Precision@3 | 36.67% | 46.67% |
| Expected-source recall | 85% | 100% |
| Support category accuracy | 90% | 90% |
| Legacy team-label agreement | 95% | 95% |
| Priority accuracy | 100% | 100% |
| Incident cases passing | 12/12 | 12/12 |

The 16 wording probes informed training examples and now count as regression cases. The 24-sentence test is separate from training/validation but manually authored, synthetic and not independently reviewed. Six of its cases remain incorrect or uncertain. Do not interpret these results as production accuracy.

## What changed

1. Added a trained character-TF-IDF Logistic Regression fallback when the domain policy has no match. Low scores, small margins, out-of-domain predictions and missing models produce clarification. Explicit domain and security policies remain authoritative.
2. Removed the early classifier's ability to discard explicit query intents. Retrieval now combines classifier hints with query intents, including priority/routing policies, and limits duplicates from a single source.
3. Added evidence origins: user-supplied, locally simulated and rule-derived. Findings are labeled as unverified offline assessments; semantic groundedness is not falsely claimed. The UI shows the evidence-origin counts.

The fixed Precision@3 denominator is always three, including unfilled slots. Legacy precision over the returned source count is 97.5% in this run. Expected sources are document-level labels from the existing support suite; they are not exhaustive relevance judgments for every chunk. The legacy security case still routes to SOC instead of Identity and Access Management by design.

## ML results

See MODEL_CARD.md and domain_model_evaluation.json. Training used 72 examples, validation 24, test 24. Model/threshold selection used validation only. The standalone model's forced test accuracy is 83.33%; the acceptance policy covers 14/24 test cases, all correct in this small set. This is not 100% overall accuracy.

The complete hybrid router is measured separately because its explicit policy can make decisions before the model is invoked. Its 75% domain accuracy is the relevant end-to-end result on these test sentences.

## Verification

The test suite covers model fallback and absence, ambiguity and explicit-domain preservation, policy-document retrieval, evidence provenance, authentication, worker reuse/recovery, payload and citation preservation, and all original smoke scenarios. Raw comparison outputs are saved in evaluation-v3/. Original source repositories remain unchanged; nothing has been merged or pushed.

## Not completed by this change

Shared ingestion/uploads, SSO, live ticket submission and connectors remain future integration work. True claim-level faithfulness, real incident resolution, independent ground-truth evaluation and sustained production load are not established by these offline tests.

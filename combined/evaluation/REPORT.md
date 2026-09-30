# Combined prototype evaluation

Date: 2026-09-29T20:33:07.299815+00:00

## Decision

The combined prototype does not yet outperform the originals. Preserve both original projects and do not merge this version. The core incident engine retains its existing case results when the correct domain is supplied, but automatic routing introduces regressions. A successful six-case smoke test was insufficient to establish quality.

## Method and scope

- Evaluated the running local/offline implementation without changing product logic.
- Used all 20 existing support regression questions and all 12 incident regression cases.
- Added 36 hand-authored routing diagnostics covering synonyms, ambiguity, negation, routine requests and escalations. These are diagnostic labels authored after code inspection, not a blind independent benchmark or representative production sample.
- Re-ran the original support code in an isolated copy including its original populated Chroma index (10 entries). Original source repositories were not modified.
- Compared the original incident engine with its original structured payload against the combined entry point's automatic routing. The original is given the domain; the automatic router solves an additional task. A separate correct-domain control removes that difference.
- Tested both the direct support engine in the prototype and the original indexed engine. Aggregate support scores matched; the missing copied index did not change these aggregate results in this offline run.
- Ran 56 timed authenticated HTTP requests against localhost, plus an excluded warm-up and login. Concurrency was capped at two. No external LLM, cloud tools, or ticket submissions were used.

## Support results (20 cases)

| Metric | Original support | Combined automatic route |
|---|---:|---:|
| Retrieval hit rate | 100.00% | 80.00% |
| Fixed Precision@3 | 36.67% | 26.67% |
| Precision over returned sources (legacy formula) | 96.67% | 80.00% |
| Recall over expected sources | 85.00% | 70.00% |
| Ticket category accuracy | 90.00% | 80.00% |
| Team routing accuracy | 100.00% | 80.00% |
| Priority accuracy | 100.00% | 80.00% |
| Legacy source-presence proxy | 100.00% | 85.00% |
| Legacy prohibited-phrase check | 100.00% | 100.00% |

Combined scores evaluate the final delivered ticket sources, category, priority and assigned team. A clarification with no answer/ticket scores zero for those expected outcomes. If investigation replaces support sources, the original support document labels may no longer match; this is a task-contract regression and does not prove all replacement documents are irrelevant.

The original recall is 85%, not the 100% stated in the resume. The historical 46.67% precision figure also was not reproduced. Current dataset, model, retrieval configuration and metric definition must accompany any future claim.

Formulas:

- Hit rate: fraction of questions with at least one expected source retrieved.
- Fixed Precision@3: expected-source matches in the first three returned sources / 3; missing slots count as misses.
- Legacy precision: expected-source matches / number of sources actually returned. Most original answers returned one source; this explains 96.67% legacy precision versus 36.67% fixed Precision@3.
- Recall: unique expected sources returned / total expected sources, averaged across questions. Labels name expected documents, not exhaustively annotated relevant chunks.
- Category/team/priority accuracy: exact expected label matches / 20. Priority matching ignores case.

Support failures:
- Company-wide authentication failure: clarify, domain None.
- Production VPN outage for many users: investigate, domain production.
- Several users in finance cannot log in: clarify, domain None.
- Security incident affecting company authentication: clarify, domain None.

## Incident regression (12 cases)

| Mode | Passing cases |
|---|---:|
| Original engine with original structured input | 12/12 (100%) |
| Combined automatic routing | 7/12 (58.33%) |
| Combined with correct domain explicitly supplied | 12/12 (100%) |

Passing requires allowed verdict/status, minimum risk score, and expected action/evidence substrings. This is a regression rubric, not proof of correct real-world diagnosis. A minimum risk threshold alone does not penalize overconfident or excessively high risk scores.

Automatic-route failures:
- cloud load balancer unhealthy targets: expected cloud, got None.
- data model drift investigation: expected data, got None.
- production queue backlog: expected production, got None.
- security suspicious oauth persistence: expected security, got it.
- security password spray against many users: expected security, got it.

The new input contract drops fields including tags, user, IP address, service and technique from some original inputs. All explicit-domain cases still passed this small rubric, which does not establish that dropping these fields is harmless.

## Routing and escalation diagnostics (36 cases)

- Domain accuracy: 66.67%.
- Domain macro-F1 (five domains plus unknown): 67.96%.
- Correct path (support/investigate/clarify): 61.11%.
- Escalation precision: 76.47% = TP / (TP + FP).
- Escalation recall: 56.52% = TP / (TP + FN).
- Unnecessary escalation rate among non-incidents: 30.77%.
- Missed escalation rate among expected incidents: 43.48%.
- Counts: TP=13, FP=4, FN=10.

Examples include treating password-spray/OAuth alerts as IT, missing paraphrases such as load-balancer failures, and interpreting negated security words as active incidents. Diagnostic labels assume routine definition questions should not trigger incident investigation; human review of this policy is still needed.

## Performance

| Workload | Concurrency | Requests | p50 ms | p95 ms | Requests/sec | HTTP success |
|---|---:|---:|---:|---:|---:|---:|
| incident_offline | 1 | 20 | 10.83 | 12.41 | 90.416 | 20/20 |
| mixed_50_percent_support | 1 | 8 | 13.43 | 4763.16 | 0.441 | 8/8 |
| incident_offline | 2 | 20 | 21.95 | 27.92 | 89.647 | 20/20 |
| mixed_50_percent_support | 2 | 8 | 16.46 | 5355.62 | 0.799 | 8/8 |

56/56 measured requests returned HTTP 200. That is transport success, not answer correctness. Incident-only runs were short local bursts of around 0.2 seconds; their throughput must not be claimed as sustained production capacity. Mixed runs contain 50% support requests and take longer because each support request launches Python and loads model dependencies. Percentiles use nearest rank; with eight mixed samples p95 is the maximum. The mixed p50 lands among fast incident requests and hides the slow support tail.

The original in-process support engine measured median 88.26 ms over 19 warm calls. This excludes Python startup and HTTP, so it is diagnostic evidence for persistent-worker optimization, not an equivalent end-to-end speed comparison.

## Existing ML model (16 test examples)

| Model | Category accuracy | Category weighted F1 | Priority accuracy | Priority weighted F1 |
|---|---:|---:|---:|---:|
| PyTorch multi-task | 100.00% | 100.00% | 87.50% | 87.40% |
| TF-IDF Logistic Regression | 93.75% | 93.65% | 68.75% | 68.21% |

These are existing classifier measurements, not improvements caused by merging. The baseline routine trains Logistic Regression on train plus validation, while the saved neural checkpoint uses its existing training procedure; this is not a controlled equal-training-data model comparison. The test set has only 16 examples and has not been independently audited for leakage.

## Groundedness, faithfulness and safety

The old source-presence proxy checks for sources and no retrieval fallback. The prohibited-phrase heuristic checks only four phrases. The safe-decision heuristic checks membership in four permitted action strings. None measures semantic entailment, correctness of a diagnosis, or whether escalation is appropriate.

The combined prohibited-phrase heuristic is 100% even though routing and report regression checks fail. Do not write '100% faithfulness' or '100% safe decisions' based on these checks.

True groundedness and faithfulness remain unmeasured: they require claim-level source/evidence annotation or an independently validated judge. Some tool/agent evidence is locally simulated and cannot substantiate live incident findings. Real resolution rate, analyst time saved, user helpfulness, live LLM latency/cost and production capacity are also unmeasured. Legacy support fallback is zero across these 20 original cases; clarification of unknown requests should not be conflated with generation fallback.

## Resume claim audit

- Five operational domains: preserved in the code and explicit-domain tests; automatic routing is not reliably covering them.
- Six tools, five specialist agents, fifteen advertised capabilities: confirmed by local registry introspection. Counts are not execution success rates or proof of real external integration.
- Twelve incident evaluation scenarios: preserved; current auto-mode passes seven.
- Seven support workflows: the old bullet mixes categories and operations; avoid treating it as seven independent domain agents or adding it to the five domains.
- Nine automated steps through postmortem: not demonstrated by the new entry point; postmortem is not exposed in the combined API.
- RAG, LangGraph, LlamaIndex and PyTorch: retained, but evaluated here in offline mode.
- JWT and SQLite: active in the prototype. Original uploads, API-key auth, monitoring, deployment and connectors are copied source, not fully consolidated features of the new app.
- Historical 100% retrieval/routing/priority/groundedness/faithfulness/safe-decision claims: cannot be transferred to the combined project.

## Next changes, in priority order

1. Route using structured domain input plus the trained classifier, with uncertainty handling and explicit multi-domain/negation tests.
2. Preserve complete incident payloads and let high-impact/security intent override routine support paths.
3. Preserve specialized support team assignments and original citations when adding investigation findings; use an explicit policy for any reassignment.
4. Replace per-request subprocess startup with a persistent worker and measure the same workloads again.
5. Consolidate ingestion, authentication and operational endpoints, then evaluate on a larger independently labeled held-out set with semantic answer review.

No product fixes, merges, pushes or resume edits were performed as part of this evaluation.

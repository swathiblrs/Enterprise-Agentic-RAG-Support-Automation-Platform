import hashlib
import json
import math
import statistics
import sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,'/private/tmp')
from measure_combined import OUT, ROOT, means, score_support

summary=json.loads((OUT/'summary.json').read_text())
original=json.loads((OUT/'original_index_baseline.json').read_text())
support=json.loads((OUT/'support_detail.json').read_text())
incidents=json.loads((OUT/'incident_detail.json').read_text())
explicit=json.loads((OUT/'explicit_domain_control.json').read_text())
performance=json.loads((OUT/'http_performance.json').read_text())
baseline=means([score_support(r['case'],r['result']) for r in original['results']])
combined=summary['support_regression']['combined']
summary['original_index_support_baseline']=baseline
summary['explicit_domain_incidents']=explicit['metrics']
summary['model_test']=original['models']
summary['http_performance']=[{k:v for k,v in p.items() if k!='requests'} for p in performance]
summary['created_at']=datetime.now(timezone.utc).isoformat()
summary['resume_capabilities']={'domains':5,'registered_tools':6,'registered_agents':5,'advertised_capabilities':15,'nine_end_to_end_steps':'not verified; new API does not expose postmortems'}
summary['unmeasured']=['semantic groundedness','semantic faithfulness','real investigation correctness','human time saved','real user helpfulness','live LLM performance','production capacity']

def pct(x):return f'{100*x:.2f}%'
rows=[]
labels={'hit':'Retrieval hit rate','precision_at_3':'Fixed Precision@3','precision_returned':'Precision over returned sources (legacy formula)', 'recall':'Recall over expected sources','category':'Ticket category accuracy','team':'Team routing accuracy','priority':'Priority accuracy','legacy_groundedness_proxy':'Legacy source-presence proxy','legacy_faithfulness_proxy':'Legacy prohibited-phrase check'}
for key,label in labels.items():rows.append(f'| {label} | {pct(baseline[key])} | {pct(combined[key])} |')
failures='\n'.join(f'- {r["case"]["question"]}: {r["combined"]["routing"]["path"]}, domain {r["combined"]["routing"]["domain"]}.' for r in support if not r['combined_score']['team'])
incident_failures='\n'.join(f'- {r["name"]}: expected {r["expected_domain"]}, got {r["combined"]["routing"]["domain"]}.' for r in incidents if not r['combined_checks']['pass'])
perfrows='\n'.join(f'| {p["workload"]} | {p["concurrency"]} | {p["n"]} | {p["p50_ms"]:.2f} | {p["p95_ms"]:.2f} | {p["throughput_requests_per_second"]:.3f} | {p["success"]}/{p["n"]} |' for p in performance)
route=summary['routing_diagnostic']
model=original['models']
text=f'''# Combined prototype evaluation

Date: {summary['created_at']}

## Decision

The combined prototype does not yet outperform the originals. Preserve both original projects and do not merge this version. The core incident engine retains its existing case results when the correct domain is supplied, but automatic routing introduces regressions. A successful six-case smoke test was insufficient to establish quality.

## Method and scope

- Evaluated the running local/offline implementation without changing product logic.
- Used all 20 existing support regression questions and all 12 incident regression cases.
- Added 36 hand-authored routing diagnostics covering synonyms, ambiguity, negation, routine requests and escalations. These are diagnostic labels authored after code inspection, not a blind independent benchmark or representative production sample.
- Re-ran the original support code in an isolated copy including its original populated Chroma index ({original['index_count']} entries). Original source repositories were not modified.
- Compared the original incident engine with its original structured payload against the combined entry point's automatic routing. The original is given the domain; the automatic router solves an additional task. A separate correct-domain control removes that difference.
- Tested both the direct support engine in the prototype and the original indexed engine. Aggregate support scores matched; the missing copied index did not change these aggregate results in this offline run.
- Ran 56 timed authenticated HTTP requests against localhost, plus an excluded warm-up and login. Concurrency was capped at two. No external LLM, cloud tools, or ticket submissions were used.

## Support results (20 cases)

| Metric | Original support | Combined automatic route |
|---|---:|---:|
{chr(10).join(rows)}

Combined scores evaluate the final delivered ticket sources, category, priority and assigned team. A clarification with no answer/ticket scores zero for those expected outcomes. If investigation replaces support sources, the original support document labels may no longer match; this is a task-contract regression and does not prove all replacement documents are irrelevant.

The original recall is 85%, not the 100% stated in the resume. The historical 46.67% precision figure also was not reproduced. Current dataset, model, retrieval configuration and metric definition must accompany any future claim.

Formulas:

- Hit rate: fraction of questions with at least one expected source retrieved.
- Fixed Precision@3: expected-source matches in the first three returned sources / 3; missing slots count as misses.
- Legacy precision: expected-source matches / number of sources actually returned. Most original answers returned one source; this explains 96.67% legacy precision versus 36.67% fixed Precision@3.
- Recall: unique expected sources returned / total expected sources, averaged across questions. Labels name expected documents, not exhaustively annotated relevant chunks.
- Category/team/priority accuracy: exact expected label matches / 20. Priority matching ignores case.

Support failures:
{failures}

## Incident regression (12 cases)

| Mode | Passing cases |
|---|---:|
| Original engine with original structured input | 12/12 (100%) |
| Combined automatic routing | 7/12 (58.33%) |
| Combined with correct domain explicitly supplied | 12/12 (100%) |

Passing requires allowed verdict/status, minimum risk score, and expected action/evidence substrings. This is a regression rubric, not proof of correct real-world diagnosis. A minimum risk threshold alone does not penalize overconfident or excessively high risk scores.

Automatic-route failures:
{incident_failures}

The new input contract drops fields including tags, user, IP address, service and technique from some original inputs. All explicit-domain cases still passed this small rubric, which does not establish that dropping these fields is harmless.

## Routing and escalation diagnostics (36 cases)

- Domain accuracy: {pct(route['domain_accuracy'])}.
- Domain macro-F1 (five domains plus unknown): {pct(route['domain_macro_f1'])}.
- Correct path (support/investigate/clarify): {pct(route['path_accuracy'])}.
- Escalation precision: {pct(route['escalation_precision'])} = TP / (TP + FP).
- Escalation recall: {pct(route['escalation_recall'])} = TP / (TP + FN).
- Unnecessary escalation rate among non-incidents: {pct(route['unnecessary_escalation_rate'])}.
- Missed escalation rate among expected incidents: {pct(route['missed_escalation_rate'])}.
- Counts: TP={route['tp']}, FP={route['fp']}, FN={route['fn']}.

Examples include treating password-spray/OAuth alerts as IT, missing paraphrases such as load-balancer failures, and interpreting negated security words as active incidents. Diagnostic labels assume routine definition questions should not trigger incident investigation; human review of this policy is still needed.

## Performance

| Workload | Concurrency | Requests | p50 ms | p95 ms | Requests/sec | HTTP success |
|---|---:|---:|---:|---:|---:|---:|
{perfrows}

56/56 measured requests returned HTTP 200. That is transport success, not answer correctness. Incident-only runs were short local bursts of around 0.2 seconds; their throughput must not be claimed as sustained production capacity. Mixed runs contain 50% support requests and take longer because each support request launches Python and loads model dependencies. Percentiles use nearest rank; with eight mixed samples p95 is the maximum. The mixed p50 lands among fast incident requests and hides the slow support tail.

The original in-process support engine measured median {statistics.median(r['wall_ms'] for r in original['results'][1:]):.2f} ms over 19 warm calls. This excludes Python startup and HTTP, so it is diagnostic evidence for persistent-worker optimization, not an equivalent end-to-end speed comparison.

## Existing ML model (16 test examples)

| Model | Category accuracy | Category weighted F1 | Priority accuracy | Priority weighted F1 |
|---|---:|---:|---:|---:|
| PyTorch multi-task | {pct(model['pytorch']['category_accuracy'])} | {pct(model['pytorch']['category_weighted_f1'])} | {pct(model['pytorch']['priority_accuracy'])} | {pct(model['pytorch']['priority_weighted_f1'])} |
| TF-IDF Logistic Regression | {pct(model['logistic_regression']['category_accuracy'])} | {pct(model['logistic_regression']['category_weighted_f1'])} | {pct(model['logistic_regression']['priority_accuracy'])} | {pct(model['logistic_regression']['priority_weighted_f1'])} |

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
'''
(OUT/'REPORT.md').write_text(text)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
files=[ROOT/'combined.py',ROOT/'support/src/orchestrator.py',ROOT/'support/src/retriever.py',ROOT/'support/tests/eval_questions.json',ROOT/'incident/app/services/investigation_graph.py']
(OUT/'provenance.json').write_text(json.dumps({'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'mode':'offline','date':summary['created_at']},indent=2))
print(text[:4000])

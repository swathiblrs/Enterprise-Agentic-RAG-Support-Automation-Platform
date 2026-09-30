# Local fixes and evaluation

## Changes

- Added weighted domain policy, security precedence, common paraphrases, bounded negation/background handling and clarification for tied domains.
- Retained request metadata and event fields when constructing incident inputs.
- Preserved support citations, specialist IT ownership, existing priority and guidance when attaching investigation findings.
- Kept security ownership explicit: SOC receives security incidents; the previous support team is retained separately.
- Replaced per-request Python startup with a persistent serialized worker, deadline, failure recovery and shutdown cleanup.
- Added the architecture diagram and browser link.

## Results

| Metric | Initial combined | Fixed combined | Original baseline |
|---|---:|---:|---:|
| Incident cases passing | 7/12 | 12/12 | 12/12 |
| Domain routing on 36 regression cases | 66.67% | 100% | Not applicable |
| Support retrieval hit rate | 80% | 100% | 100% |
| Fixed Precision@3 | 26.67% | 36.67% | 36.67% |
| Support source recall | 70% | 85% | 85% |
| Support category accuracy | 80% | 90% | 90% |
| Team agreement with legacy support labels | 80% | 95% | 100% |
| Priority accuracy | 80% | 100% | 100% |

The remaining team mismatch is intentional: `Security incident affecting company authentication` is assigned to SOC rather than the old Identity and Access Management label. That original support team is still present in the output. The old labels were not edited to improve the score.

These 36 routing cases were used to guide the fix and now serve as regression tests. A separate 16-request hand-authored wording probe passed 12/16 (75%). It missed virtual-private-network wording, `authenticate`, unexpected OAuth permissions, and HTTP 502. Those probe results were not used to tune the router further. This is still a policy router with limited language coverage, not independently validated production AI.

Eight automated tests passed, covering the original smoke workflows, auth-before-execution, metadata preservation, citation/team/priority preservation, security assignment, worker reuse and restart after process termination.

## Performance

Same offline localhost workload, two concurrent requests, eight requests split equally between support and incident processing:

| Metric | Initial prototype | Persistent-worker prototype |
|---|---:|---:|
| Mixed workload p95 | 5355.62 ms | 390.99 ms |
| Throughput | 0.80 requests/s | 10.62 requests/s |
| HTTP success across complete benchmark | 56/56 | 56/56 |

This is approximately 92.7% lower p95 and 13.3x throughput for this small warm local workload. It is not a sustained capacity or live-LLM benchmark. Eight samples make nearest-rank p95 equal to the maximum. A preceding sequential batch warmed the new worker; the old implementation could not retain a warm worker between calls. The new cold-start-inclusive sequential mixed batch had p95 3867.90 ms. Hardware/page-cache effects and small sample sizes limit generalization.

## What this establishes

The integration regressions are repaired on the existing cases, and repeated support calls avoid process startup. Retrieval quality is restored to the original level, not improved beyond it. Semantic faithfulness, live evidence accuracy, actual resolution rate and real user benefit are still unmeasured. Do not reuse the old blanket 100% resume claims.

Raw updated results are under `evaluation-v2/`. Initial results remain under `evaluation/`. The original repositories remain untouched. Nothing has been committed, merged or pushed.

# IT Support Workload Evaluation

This independent project uses public IT support guidance to explore helpdesk automation. The 65,000-request workload is a synthetic benchmark and does not represent real users, ticket resolutions, or institutional adoption.

## Knowledge and Development Cases

The support corpus contains 15 documents: five original support/policy guides and ten project-authored summaries of public IT support pages. There are 30 indexed chunks. The source URLs, institution, and review date are recorded in each summary and in `support/public_sources.json`. Public content was reviewed on October 22, 2026.

The support development suite contains 40 synthetic questions spanning NetID, password reset, Duo, eduroam, VPN, OIT HelpDesk, software, printing, Microsoft 365, and Canvas. No student records or actual helpdesk conversations are included. These questions guided development and are not an independently labeled held-out dataset.

## Fresh RAG Checks

Each of the 40 questions was sent once to a fresh offline server before the workload test. All used the support path and missed the answer cache.

| Check | Result |
| --- | --- |
| Expected source present in top-three results | 40/40 (100%) |
| Expected answer term present | 40/40 (100%) |
| Expected official source URL in answer | 40/40 (100%) |
| Fresh-request p95 latency | 389.17 ms |

Source hit counts whether the manually specified document appears in the returned sources. Term matching is a case-insensitive substring check; URL matching checks the official URL. These checks do not measure semantic faithfulness or certify that every sentence is correct. The offline answer path extracts the selected public guidance rather than calling an LLM. Model loading is included in the cold run; p95 does not describe its slowest request.

## 65,000-Request HTTP Workload

After the 40 cold requests, the client sent 65,000 POST requests to the real localhost FastAPI server at concurrency 16. Each request passed JWT verification, domain routing, public FAQ delivery, JSON response serialization, and SQLite case persistence. The client checked question/response correspondence and distinct case IDs.

| Metric | Measured result |
| --- | --- |
| Requests completed successfully | 65,000/65,000 |
| Errors | 0 |
| Distinct FAQ questions | 40 |
| Distinct case IDs | 65,000 |
| Concurrency | 16 in-flight requests |
| Measured duration | 61.47 seconds |
| Throughput | 1,057.34 requests/second |
| p50 / p95 / p99 | 3.06 / 54.00 / 202.59 ms |
| Maximum latency | 2,121.79 ms |
| Warm cache hits | 65,000 (100%) |
| Persisted cases, including 40 cold requests | 65,040 |

Throughput is completed requests divided by measured duration. Percentiles use the nearest-rank method over all 65,000 observed client durations. The workload excludes the preceding 40 cold requests. A separate SQLite database was used for synthetic traffic.

This measures repeated public FAQ delivery. It does not demonstrate 65,000 unique users, simultaneous users, fresh RAG executions, live LLM calls, real tickets resolved, or sustained annual production availability. All requests use one demo identity. Every response is persisted as a separate case. The server, client, storage, and cache run on one local machine.

## Why These Measurements

- Source coverage checks whether support guidance is actually reachable after indexing.
- Answer-term and URL checks catch missing guidance and broken attribution while remaining easy to audit.
- Error count and persisted-case count check completion and storage under load.
- p95/p99 reveal slower requests that an average can hide.
- Throughput and concurrency describe the measured traffic conditions.
- Cache-hit count explains the performance benefit and prevents confusing cached delivery with fresh retrieval speed.

## Reproduce

From this directory, use the independent API/support environments described in the main README. Rebuild the index from `support/` first. Start a fresh server and a fresh synthetic database for each run:

```sh
export SUPPORT_PYTHON="$PWD/.venv-support/bin/python"
export CASES_DB_PATH="/private/tmp/support-workload-$(date +%s).sqlite3"
.venv-api/bin/python -m uvicorn api:api --host 127.0.0.1 --port 8766 --no-access-log
```

In another terminal in the same directory, pass the same database path:

```sh
.venv-api/bin/python benchmark_university.py --requests 65000 --concurrency 16 --database /private/tmp/YOUR-FRESH-DATABASE.sqlite3
```

The runner first writes `evaluation-university/university_quality.json`, then `evaluation-university/workload_summary.json`. Stop the server after measurement. Re-running against an already warm server changes cold-run behavior.

The cache only accepts the 40 exact public questions in `university_eval_questions.json`, holds at most 128 entries, expires after one hour, and returns copied results. Arbitrary user questions bypass it. Restart after corpus updates. This is not semantic caching or a substitute for independently reviewed content.

## Next Validation

Expand paraphrases independently, test unfamiliar support questions, evaluate real LLM answers, test cache misses and sustained arrival bursts, and pilot staff-reviewed suggestions before claiming real helpdesk outcomes.

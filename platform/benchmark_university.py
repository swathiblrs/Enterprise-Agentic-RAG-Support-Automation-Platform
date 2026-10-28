"""Real HTTP benchmark: cold university FAQs, then a repeated-FAQ workload."""
import argparse
import asyncio
import hashlib
import json
import math
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


async def run(args):
    root = Path(__file__).parent
    cases = json.loads((root / "university_eval_questions.json").read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(180, connect=10)
    async with httpx.AsyncClient(base_url=args.url, timeout=timeout,
                                 limits=httpx.Limits(max_connections=args.concurrency)) as client:
        login = await client.post("/auth/login", json={"username": "demo", "password": "local-demo-only"})
        login.raise_for_status()
        client.headers["Authorization"] = "Bearer " + login.json()["access_token"]
        cold = []
        for case in cases:
            started = time.perf_counter()
            response = await client.post("/cases", json={"question": case["question"]})
            response.raise_for_status()
            data = response.json()
            support = data.get("support") or {}
            sources = support.get("sources", [])
            cold.append({**case, "latency_ms": (time.perf_counter()-started)*1000,
                         "retrieved_sources": sources, "path": data["routing"]["path"],
                         "source_hit": case["expected_source"] in sources,
                         "answer_term_found": case["answer_term"].lower() in data["answer"].lower(),
                         "official_url_found": case["expected_url"] in data["answer"],
                         "cache_hit": support.get("delivery", {}).get("public_faq_cache_hit", False)})
        (output / "university_quality.json").write_text(json.dumps(cold, indent=2))
        durations, failures, counts, ids = [], [], Counter(), set()
        start = time.perf_counter()
        async def worker(offset):
            for index in range(offset, args.requests, args.concurrency):
                case = cases[index % len(cases)]
                before = time.perf_counter()
                try:
                    response = await client.post("/cases", json={"question": case["question"]})
                    counts["http_" + str(response.status_code)] += 1
                    response.raise_for_status()
                    data = response.json()
                    if data["question"] != case["question"] or data["id"] in ids:
                        raise ValueError("Mismatched question or duplicate case id")
                    ids.add(data["id"])
                    counts["successful"] += 1
                    counts["cache_hits"] += bool((data.get("support") or {}).get("delivery", {}).get("public_faq_cache_hit"))
                    counts[data["routing"]["path"]] += 1
                except Exception as error:
                    counts["failed"] += 1
                    if len(failures) < 20:
                        failures.append({"request": index, "error": str(error)})
                durations.append((time.perf_counter()-before)*1000)
                if len(durations) % 5000 == 0:
                    print(json.dumps({"completed": len(durations), "elapsed_seconds": round(time.perf_counter()-start,1),
                                      "failed": counts["failed"]}), flush=True)
        await asyncio.gather(*(worker(i) for i in range(args.concurrency)))
        elapsed = time.perf_counter()-start
    with sqlite3.connect(args.database) as db:
        stored = db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "localhost_http_offline_repeated_public_faq",
        "requested": args.requests, "completed": len(durations), "concurrency": args.concurrency,
        "distinct_questions": len(cases), "unique_case_ids": len(ids),
        "elapsed_seconds": elapsed, "requests_per_second": len(durations)/elapsed,
        "latency_ms": {"p50": percentile(durations,.5), "p95": percentile(durations,.95),
                       "p99": percentile(durations,.99), "max": max(durations)},
        "counts": dict(counts), "stored_cases_including_cold_run": stored,
        "cache_ttl_seconds": 3600, "cache_warmup_excluded": True,
        "fresh_rag_quality": {"questions": len(cold), "source_hits": sum(x["source_hit"] for x in cold),
                             "answer_term_hits": sum(x["answer_term_found"] for x in cold),
                             "official_url_hits": sum(x["official_url_found"] for x in cold),
                             "p95_ms": percentile([x["latency_ms"] for x in cold],.95)},
        "dataset_sha256": hashlib.sha256((root/"university_eval_questions.json").read_bytes()).hexdigest(),
        "error_examples": failures,
        "limitations": ["Synthetic repetitions of 40 public FAQs; one demo identity, not 65000 users.",
                        "Warm cache delivery reuses earlier RAG output; not 65000 fresh retrievals or LLM calls.",
                        "Local offline server; no external ITSM submissions or production availability claim.",
                        "Term/source presence checks are development checks, not independent semantic evaluation."]
    }
    (output/"workload_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if counts["failed"] or counts["successful"] != args.requests:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--requests", type=int, default=65000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", default="evaluation-university")
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("requests and concurrency must be positive")
    asyncio.run(run(args))

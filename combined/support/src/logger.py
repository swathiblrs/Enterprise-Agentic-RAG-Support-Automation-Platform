import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from src.config import PERSISTENCE_BACKEND
from src.persistence import write_feedback_log, write_query_log


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "query_logs.jsonl"
FEEDBACK_FILE = LOG_DIR / "feedback_logs.jsonl"


def utc_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_jsonl(file_path: Path, entry: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(exist_ok=True)

    with file_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")


def log_query(entry: Dict[str, Any]) -> None:
    log_entry = {
        "timestamp": utc_timestamp(),
        **entry,
    }

    write_jsonl(LOG_FILE, log_entry)
    if PERSISTENCE_BACKEND == "sqlite":
        write_query_log(log_entry)


def log_feedback(entry: Dict[str, Any]) -> None:
    feedback_entry = {
        "timestamp": utc_timestamp(),
        **entry,
    }

    write_jsonl(FEEDBACK_FILE, feedback_entry)
    if PERSISTENCE_BACKEND == "sqlite":
        write_feedback_log(feedback_entry)


def read_jsonl(file_path: Path) -> list:
    if not file_path.exists():
        return []

    entries = []
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                entries.append(json.loads(line))

    return entries


def read_logs() -> list:
    return read_jsonl(LOG_FILE)


def read_feedback() -> list:
    return read_jsonl(FEEDBACK_FILE)


def calculate_metrics() -> Dict[str, Any]:
    logs = read_logs()
    feedback = read_feedback()

    total_queries = len(logs)
    total_feedback = len(feedback)

    if total_queries == 0:
        return {
            "total_queries": 0,
            "total_feedback": total_feedback,
            "fallback_rate": 0.0,
            "average_latency_ms": 0.0,
            "average_workflow_latency_ms": 0.0,
            "average_answer_stage_latency_ms": 0.0,
            "average_decision_stage_latency_ms": 0.0,
            "average_ticket_draft_stage_latency_ms": 0.0,
            "average_retrieved_chunk_count": 0.0,
            "average_source_count": 0.0,
            "average_overall_confidence": 0.0,
            "agent_decision_counts": {},
            "ticket_category_counts": {},
            "feedback_helpful_rate": calculate_helpful_rate(feedback),
        }

    fallback_count = sum(1 for entry in logs if entry.get("fallback_triggered"))
    latency_values = [entry.get("latency_ms", 0.0) for entry in logs]
    confidence_values = [
        entry.get("confidence", {}).get("overall_confidence", 0.0)
        for entry in logs
    ]
    engineering_metrics = [entry.get("engineering_metrics", {}) for entry in logs]

    return {
        "total_queries": total_queries,
        "total_feedback": total_feedback,
        "fallback_rate": round(fallback_count / total_queries, 4),
        "average_latency_ms": round(sum(latency_values) / total_queries, 2),
        "average_workflow_latency_ms": average_nested_metric(
            engineering_metrics,
            "total_workflow_latency_ms",
        ),
        "average_answer_stage_latency_ms": average_nested_metric(
            engineering_metrics,
            "answer_stage_latency_ms",
        ),
        "average_decision_stage_latency_ms": average_nested_metric(
            engineering_metrics,
            "decision_stage_latency_ms",
        ),
        "average_ticket_draft_stage_latency_ms": average_nested_metric(
            engineering_metrics,
            "ticket_draft_stage_latency_ms",
        ),
        "average_retrieved_chunk_count": average_nested_metric(
            engineering_metrics,
            "retrieved_chunk_count",
        ),
        "average_source_count": average_nested_metric(engineering_metrics, "source_count"),
        "average_overall_confidence": round(sum(confidence_values) / total_queries, 2),
        "agent_decision_counts": count_by_nested_key(logs, "agent_decision", "next_action"),
        "ticket_category_counts": count_by_nested_key(logs, "ticket", "category"),
        "feedback_helpful_rate": calculate_helpful_rate(feedback),
    }


def average_nested_metric(entries: list, key: str) -> float:
    values = [entry.get(key, 0.0) for entry in entries if key in entry]
    if not values:
        return 0.0

    return round(sum(values) / len(values), 2)


def count_by_nested_key(entries: list, parent_key: str, child_key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    for entry in entries:
        value = entry.get(parent_key, {}).get(child_key, "unknown")
        counts[value] = counts.get(value, 0) + 1

    return counts


def calculate_helpful_rate(feedback: list) -> float:
    if not feedback:
        return 0.0

    helpful_count = sum(1 for entry in feedback if entry.get("answer_helpful"))

    return round(helpful_count / len(feedback), 4)

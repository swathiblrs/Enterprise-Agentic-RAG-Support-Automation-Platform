from time import time
from typing import Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from src.logger import log_query
from src.orchestrator import run_support_workflow


app = FastAPI(
    title="Enterprise RAG Support Automation Platform",
    description="RAG-based support automation API with retrieval, answer generation, ticket classification, and query logging.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str


@app.get("/")
def root() -> Dict:
    return {
        "message": "Enterprise RAG Support Automation Platform API",
        "status": "running",
    }


@app.get("/health")
def health_check() -> Dict:
    return {
        "status": "ok",
        "service": "enterprise-rag-support-platform",
    }


@app.post("/ask")
def ask_question(request: AskRequest) -> Dict:
    start_time = time()

    response = run_support_workflow(request.question)

    latency_ms = round((time() - start_time) * 1000, 2)

    api_response = {
        "question": request.question,
        "answer": response.get("answer", ""),
        "sources": response.get("sources", []),
        "ticket": response.get("ticket", {}),
        "fallback_triggered": response.get("fallback_triggered", False),
        "confidence": response.get("confidence", {}),
        "confusion_analysis": response.get("confusion_analysis", {}),
        "agent_decision": response.get("agent_decision", {}),
        "ticket_draft": response.get("ticket_draft", {}),
        "latency_ms": latency_ms,
    }

    log_query(api_response)

    return api_response


@app.get("/logs")
def get_logs() -> Dict:
    log_file = "logs/query_logs.jsonl"

    try:
        with open(log_file, "r", encoding="utf-8") as file:
            logs: List[str] = file.readlines()

        return {
            "total_logs": len(logs),
            "logs": logs[-20:],
        }

    except FileNotFoundError:
        return {
            "total_logs": 0,
            "logs": [],
        }

from uuid import uuid4
from time import time
from typing import Dict

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.document_store import save_uploaded_document
from src.ingest import ingest_documents
from src.integrations import create_ticket, notify_chat, prepare_ticket_payload
from src.logger import calculate_metrics, log_feedback, log_query, read_feedback, read_logs
from src.orchestrator import run_support_workflow
from src.config import DEFAULT_DOMAIN
from src.persistence import database_health, read_table_payloads, write_ticket_draft
from src.security import ROLE_ADMIN, ROLE_SUPPORT_AGENT, authenticate_user, create_access_token, require_auth


app = FastAPI(
    title="Enterprise RAG Support Automation Platform",
    description="RAG-based support automation API with retrieval, answer generation, ticket classification, and query logging.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    domain: str = Field(DEFAULT_DOMAIN, min_length=2, max_length=80)


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=3, max_length=120)
    question: str = Field(..., min_length=3, max_length=2000)
    answer_helpful: bool
    correct_sources: bool
    correct_ticket_routing: bool
    correct_priority: bool
    comments: str = Field("", max_length=2000)


class DocumentUploadRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    content_base64: str = Field(..., min_length=1)
    domain: str = Field(DEFAULT_DOMAIN, min_length=2, max_length=80)
    source_type: str = Field("upload", min_length=2, max_length=80)
    reindex: bool = False


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=80)
    password: str = Field(..., min_length=1, max_length=200)


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


@app.get("/ready")
def readiness_check() -> Dict:
    database = database_health()

    return {
        "status": "ok" if database["status"] == "ok" else "degraded",
        "service": "enterprise-rag-support-platform",
        "checks": {
            "database": database,
        },
    }


@app.post("/auth/login")
def login(request: LoginRequest) -> Dict:
    user = authenticate_user(request.username, request.password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_access_token(user["sub"], user["role"])

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
    }


@app.post("/ask")
def ask_question(
    request: AskRequest,
    _: Dict = Depends(require_auth([ROLE_SUPPORT_AGENT])),
) -> Dict:
    request_id = str(uuid4())
    start_time = time()

    response = run_support_workflow(request.question, domain=request.domain)
    ticket_payload = prepare_ticket_payload(request_id, response)
    ticket_integration = create_ticket(ticket_payload)
    chat_integration = notify_chat(ticket_payload)

    latency_ms = round((time() - start_time) * 1000, 2)

    api_response = {
        "request_id": request_id,
        "question": request.question,
        "domain": request.domain,
        "answer": response.get("answer", ""),
        "sources": response.get("sources", []),
        "ticket": response.get("ticket", {}),
        "answer_generation_mode": response.get("answer_generation_mode", "unknown"),
        "fallback_triggered": response.get("fallback_triggered", False),
        "confidence": response.get("confidence", {}),
        "confusion_analysis": response.get("confusion_analysis", {}),
        "agent_decision": response.get("agent_decision", {}),
        "ticket_draft": response.get("ticket_draft", {}),
        "integrations": {
            "itsm": ticket_integration,
            "chat": chat_integration,
        },
        "latency_ms": latency_ms,
    }

    log_query(api_response)
    write_ticket_draft(request_id, ticket_payload, ticket_integration)

    return api_response


@app.get("/logs")
def get_logs(_: Dict = Depends(require_auth([ROLE_ADMIN]))) -> Dict:
    logs = read_logs()

    return {
        "total_logs": len(logs),
        "logs": logs[-20:],
    }


@app.post("/feedback")
def submit_feedback(
    request: FeedbackRequest,
    _: Dict = Depends(require_auth([ROLE_SUPPORT_AGENT])),
) -> Dict:
    if hasattr(request, "model_dump"):
        feedback_entry = request.model_dump()
    else:
        feedback_entry = request.dict()

    log_feedback(feedback_entry)

    return {
        "status": "received",
        "request_id": request.request_id,
    }


@app.get("/feedback")
def get_feedback(_: Dict = Depends(require_auth([ROLE_ADMIN]))) -> Dict:
    feedback = read_feedback()

    return {
        "total_feedback": len(feedback),
        "feedback": feedback[-20:],
    }


@app.get("/metrics")
def get_metrics(_: Dict = Depends(require_auth([ROLE_ADMIN]))) -> Dict:
    return calculate_metrics()


@app.get("/tickets")
def get_ticket_drafts(_: Dict = Depends(require_auth([ROLE_ADMIN]))) -> Dict:
    tickets = read_table_payloads("ticket_drafts", limit=20)

    return {
        "total_returned": len(tickets),
        "tickets": tickets,
    }


@app.post("/documents/upload")
def upload_document(
    request: DocumentUploadRequest,
    _: Dict = Depends(require_auth([ROLE_ADMIN])),
) -> Dict:
    try:
        document = save_uploaded_document(
            request.filename,
            request.content_base64,
            domain=request.domain,
            source_type=request.source_type,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    reindexed = False
    if request.reindex:
        ingest_documents()
        reindexed = True

    return {
        "status": "uploaded",
        "document": document,
        "reindexed": reindexed,
    }

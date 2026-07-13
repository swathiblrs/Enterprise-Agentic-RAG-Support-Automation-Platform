import base64
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "enterprise-rag-support-platform"


def test_ask_endpoint_vpn_question():
    payload = {
        "question": "My VPN is not working after I reset my password"
    }

    response = client.post("/ask", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == payload["question"]
    assert "request_id" in data
    assert "answer" in data
    assert "sources" in data
    assert "ticket" in data
    assert "answer_generation_mode" in data
    assert "fallback_triggered" in data
    assert "confidence" in data
    assert "agent_decision" in data
    assert "ticket_draft" in data
    assert "latency_ms" in data

    assert data["ticket"]["category"] == "VPN Connectivity"
    assert data["ticket"]["priority"] == "Medium"
    assert data["ticket"]["assigned_team"] == "Network Support"
    assert data["ticket_draft"]["assigned_team"] == "Network Support"
    assert data["agent_decision"]["assigned_team"] == "Network Support"


def test_ask_endpoint_mfa_question():
    payload = {
        "question": "I cannot approve Duo push notifications"
    }

    response = client.post("/ask", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["ticket"]["category"] == "Multi-Factor Authentication"
    assert data["ticket"]["priority"] == "Medium"
    assert data["ticket"]["assigned_team"] == "Identity and Access Management"


def test_ask_endpoint_account_locked_question():
    payload = {
        "question": "My account is locked"
    }

    response = client.post("/ask", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["ticket"]["category"] == "Account Access"
    assert data["ticket"]["priority"] == "Medium"
    assert data["ticket"]["assigned_team"] == "Identity and Access Management"


def test_ask_endpoint_critical_question():
    payload = {
        "question": "Company-wide authentication failure"
    }

    response = client.post("/ask", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["ticket"]["priority"] == "Critical"
    assert data["ticket"]["assigned_team"] == "Identity and Access Management"
    assert data["agent_decision"]["next_action"] == "create_urgent_ticket_draft"


def test_feedback_endpoint_accepts_workflow_feedback():
    payload = {
        "request_id": "test-request-id",
        "question": "My VPN is not working",
        "answer_helpful": True,
        "correct_sources": True,
        "correct_ticket_routing": True,
        "correct_priority": True,
        "comments": "Looks good",
    }

    response = client.post("/feedback", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "received"
    assert data["request_id"] == payload["request_id"]


def test_metrics_endpoint_returns_observability_summary():
    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert "total_queries" in data
    assert "fallback_rate" in data
    assert "average_latency_ms" in data
    assert "agent_decision_counts" in data


def test_document_upload_endpoint_accepts_text_documents():
    content = base64.b64encode(b"# Test KB\n\nTemporary support document.").decode("utf-8")
    payload = {
        "filename": "temporary_test_kb.md",
        "content_base64": content,
        "reindex": False,
    }

    response = client.post("/documents/upload", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "uploaded"
    assert data["document"]["filename"] == payload["filename"]
    assert data["reindexed"] is False

    uploaded_path = Path(data["document"]["path"])
    if uploaded_path.exists():
        uploaded_path.unlink()


def test_document_upload_endpoint_rejects_unsupported_files():
    content = base64.b64encode(b"not supported").decode("utf-8")
    payload = {
        "filename": "malware.exe",
        "content_base64": content,
        "reindex": False,
    }

    response = client.post("/documents/upload", json=payload)

    assert response.status_code == 400

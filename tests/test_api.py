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
    assert "answer" in data
    assert "sources" in data
    assert "ticket" in data
    assert "fallback_triggered" in data
    assert "confidence" in data
    assert "agent_decision" in data
    assert "ticket_draft" in data
    assert "latency_ms" in data

    assert data["ticket"]["category"] == "VPN Connectivity"
    assert data["ticket"]["priority"] == "Medium"
    assert data["ticket"]["assigned_team"] == "Network Support"
    assert data["ticket_draft"]["assigned_team"] == "Network Support"


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

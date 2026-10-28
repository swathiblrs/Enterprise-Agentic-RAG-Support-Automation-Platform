from typing import Any, Dict

import requests

from src.config import (
    CHAT_INTEGRATION_MODE,
    CHAT_WEBHOOK_URL,
    ITSM_INTEGRATION_MODE,
    ITSM_WEBHOOK_URL,
)
from src.logger import utc_timestamp


def prepare_ticket_payload(request_id: str, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
    ticket = workflow_state.get("ticket", {})
    ticket_draft = workflow_state.get("ticket_draft", {})
    decision = workflow_state.get("agent_decision", {})

    return {
        "request_id": request_id,
        "title": ticket_draft.get("title", ticket.get("summary", "")),
        "description": ticket_draft.get("description", workflow_state.get("answer", "")),
        "category": ticket_draft.get("category", ticket.get("category", "General IT Support")),
        "priority": ticket_draft.get("priority", ticket.get("priority", "Low")),
        "assigned_team": ticket_draft.get("assigned_team", ticket.get("assigned_team", "Service Desk")),
        "next_action": decision.get("next_action", "answer_and_create_ticket_draft"),
        "source_references": ticket_draft.get("source_references", workflow_state.get("sources", [])),
        "suggested_steps": ticket_draft.get("suggested_steps", []),
        "confidence": workflow_state.get("confidence", {}),
    }


def create_ticket(ticket_payload: Dict[str, Any]) -> Dict[str, Any]:
    if ITSM_INTEGRATION_MODE == "disabled":
        return integration_result("disabled", "ITSM ticket creation disabled.", ticket_payload)

    if ITSM_INTEGRATION_MODE == "webhook":
        return post_webhook(ITSM_WEBHOOK_URL, ticket_payload, "itsm")

    ticket_id = f"MOCK-{ticket_payload.get('request_id', '')[:8]}"
    return integration_result(
        "mock_created",
        "Mock ITSM ticket prepared. Configure ITSM_WEBHOOK_URL for real ticket creation.",
        ticket_payload,
        external_id=ticket_id,
    )


def notify_chat(ticket_payload: Dict[str, Any]) -> Dict[str, Any]:
    notification = {
        "text": (
            f"{ticket_payload.get('priority')} support issue routed to "
            f"{ticket_payload.get('assigned_team')}: {ticket_payload.get('title')}"
        ),
        "ticket": ticket_payload,
    }

    if CHAT_INTEGRATION_MODE == "disabled":
        return integration_result("disabled", "Chat notification disabled.", notification)

    if CHAT_INTEGRATION_MODE == "webhook":
        return post_webhook(CHAT_WEBHOOK_URL, notification, "chat")

    return integration_result(
        "mock_sent",
        "Mock chat notification prepared. Configure CHAT_WEBHOOK_URL for real notifications.",
        notification,
    )


def post_webhook(url: str, payload: Dict[str, Any], target: str) -> Dict[str, Any]:
    if not url:
        return integration_result("error", f"{target} webhook URL is not configured.", payload)

    try:
        response = requests.post(url, json=payload, timeout=10)
        return integration_result(
            "sent" if response.status_code < 400 else "error",
            f"{target} webhook returned HTTP {response.status_code}.",
            payload,
            status_code=response.status_code,
        )
    except Exception as error:
        return integration_result("error", f"{target} webhook failed: {error}", payload)


def integration_result(status: str, message: str, payload: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    return {
        "status": status,
        "message": message,
        "timestamp": utc_timestamp(),
        "payload": payload,
        **extra,
    }

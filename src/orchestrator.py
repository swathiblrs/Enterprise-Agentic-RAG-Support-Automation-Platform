from typing import Dict, List

from src.config import CLARIFY_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD
from src.generator import answer_question


SUPPORT_SIGNALS = {
    "VPN Connectivity": ["vpn", "network", "connect", "connection"],
    "Account Access": ["password", "locked", "login", "sign in", "account"],
    "Multi-Factor Authentication": ["duo", "mfa", "authentication", "push"],
}


def run_support_workflow(question: str) -> Dict:
    """
    Coordinates the support workflow around the RAG answer.

    The API calls this orchestrator instead of directly calling the generator so
    the project can track confidence, decisions, escalation, and ticket drafts.
    """
    state = build_initial_state(question)

    response = answer_question(question)
    state.update(
        {
            "answer": response.get("answer", ""),
            "sources": response.get("sources", []),
            "retrieved_chunks": response.get("retrieved_chunks", []),
            "ticket": response.get("ticket", {}),
            "fallback_triggered": response.get("fallback", False),
        }
    )

    state["confidence"] = calculate_confidence(state)
    state["confusion_analysis"] = detect_confusion(question)
    state["agent_decision"] = decide_next_action(state)
    state["ticket_draft"] = build_ticket_draft(state)

    return state


def build_initial_state(question: str) -> Dict:
    return {
        "question": question,
        "answer": "",
        "sources": [],
        "retrieved_chunks": [],
        "ticket": {},
        "fallback_triggered": False,
        "confidence": {},
        "confusion_analysis": {},
        "agent_decision": {},
        "ticket_draft": {},
        "errors": [],
    }


def calculate_confidence(state: Dict) -> Dict:
    retrieved_chunks = state.get("retrieved_chunks", [])
    ticket = state.get("ticket", {})
    confusion = detect_confusion(state.get("question", ""))

    retrieval_confidence = calculate_retrieval_confidence(retrieved_chunks)
    classification_confidence = calculate_classification_confidence(
        state.get("question", ""),
        ticket.get("category", "General IT Support"),
        confusion,
    )

    if state.get("fallback_triggered"):
        overall_confidence = min(retrieval_confidence, classification_confidence, 0.35)
    else:
        overall_confidence = round((retrieval_confidence * 0.6) + (classification_confidence * 0.4), 2)

    return {
        "retrieval_confidence": retrieval_confidence,
        "classification_confidence": classification_confidence,
        "overall_confidence": overall_confidence,
    }


def calculate_retrieval_confidence(retrieved_chunks: List[Dict]) -> float:
    if not retrieved_chunks:
        return 0.0

    top_score = max(float(chunk.get("score", 0.0)) for chunk in retrieved_chunks)
    source_count = len({chunk.get("source") for chunk in retrieved_chunks if chunk.get("source")})

    score_confidence = min(top_score, 1.0)
    source_bonus = min(source_count * 0.05, 0.15)

    return round(min(score_confidence + source_bonus, 1.0), 2)


def calculate_classification_confidence(question: str, category: str, confusion: Dict) -> float:
    question_lower = question.lower()
    signals = SUPPORT_SIGNALS.get(category, [])
    matching_signals = sum(1 for signal in signals if signal in question_lower)

    if category == "General IT Support":
        base_confidence = 0.55
    else:
        base_confidence = 0.65 + min(matching_signals * 0.1, 0.25)

    if confusion.get("multi_intent"):
        base_confidence -= 0.15

    return round(max(min(base_confidence, 1.0), 0.0), 2)


def detect_confusion(question: str) -> Dict:
    question_lower = question.lower()
    matched_categories = []

    for category, signals in SUPPORT_SIGNALS.items():
        if any(signal in question_lower for signal in signals):
            matched_categories.append(category)

    return {
        "multi_intent": len(matched_categories) > 1,
        "matched_categories": matched_categories,
    }


def decide_next_action(state: Dict) -> Dict:
    confidence = state.get("confidence", {})
    ticket = state.get("ticket", {})
    confusion = state.get("confusion_analysis", {})
    overall_confidence = confidence.get("overall_confidence", 0.0)

    if state.get("fallback_triggered") or overall_confidence < LOW_CONFIDENCE_THRESHOLD:
        return {
            "next_action": "escalate_to_human",
            "reason": "Low confidence or insufficient retrieved knowledge.",
            "assigned_team": ticket.get("assigned_team", "Service Desk"),
        }

    if confusion.get("multi_intent") and overall_confidence < CLARIFY_CONFIDENCE_THRESHOLD:
        return {
            "next_action": "ask_clarifying_question",
            "reason": "Multiple support intents were detected with moderate confidence.",
            "assigned_team": ticket.get("assigned_team", "Service Desk"),
        }

    if ticket.get("priority") == "Critical":
        return {
            "next_action": "create_urgent_ticket_draft",
            "reason": "Critical priority issue detected.",
            "assigned_team": ticket.get("assigned_team", "Service Desk"),
        }

    return {
        "next_action": "answer_and_create_ticket_draft",
        "reason": "Sufficient confidence to answer and prepare a support ticket draft.",
        "assigned_team": ticket.get("assigned_team", "Service Desk"),
    }


def build_ticket_draft(state: Dict) -> Dict:
    ticket = state.get("ticket", {})
    sources = state.get("sources", [])
    decision = state.get("agent_decision", {})

    return {
        "title": ticket.get("summary", state.get("question", "")),
        "description": state.get("answer", ""),
        "category": ticket.get("category", "General IT Support"),
        "priority": ticket.get("priority", "Low"),
        "assigned_team": ticket.get("assigned_team", "Service Desk"),
        "source_references": sources,
        "next_action": decision.get("next_action", "answer_and_create_ticket_draft"),
        "suggested_steps": build_suggested_steps(ticket.get("category", "General IT Support")),
    }


def build_suggested_steps(category: str) -> List[str]:
    if category == "VPN Connectivity":
        return [
            "Verify network connectivity and VPN client status.",
            "Confirm whether the issue started after a password reset.",
            "Escalate to Network Support if reconnection fails.",
        ]

    if category == "Account Access":
        return [
            "Verify account lockout or password reset status.",
            "Confirm whether the user can sign in to other enterprise services.",
            "Escalate to Identity and Access Management if access remains blocked.",
        ]

    if category == "Multi-Factor Authentication":
        return [
            "Verify MFA device registration and push notification status.",
            "Confirm the user has network access and can receive MFA prompts.",
            "Escalate to Identity and Access Management for MFA reset or re-enrollment.",
        ]

    return [
        "Collect affected system, error message, and business impact.",
        "Confirm whether one user or multiple users are affected.",
        "Route to the Service Desk for triage.",
    ]

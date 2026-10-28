from typing import Dict

from src.ml_ticket_model import predict_ticket_ml
from src.torch_ticket_model import predict_ticket_torch


CATEGORY_CONFIDENCE_THRESHOLD = 0.3
PRIORITY_CONFIDENCE_THRESHOLD = 0.3

CATEGORY_RETRIEVAL_INTENTS = {
    "VPN Connectivity": ["vpn"],
    "Account Access": ["account"],
    "Multi-Factor Authentication": ["mfa"],
    "General IT Support": [],
}


def classify_ticket(question: str) -> Dict:
    question_lower = question.lower()
    ml_prediction = predict_ticket_torch(question) or predict_ticket_ml(question)

    rule_category = classify_category_with_rules(question_lower)
    category = choose_category(rule_category, ml_prediction)
    assigned_team = assigned_team_for_category(category)
    rule_priority = predict_priority(question_lower)
    priority = choose_priority(rule_priority, ml_prediction)

    return {
        "summary": generate_summary(question),
        "category": category,
        "priority": priority,
        "assigned_team": assigned_team,
        "classification_method": "hybrid_ml_nlp",
        "ml_model": (ml_prediction or {}).get("model", "unavailable"),
        "ml_category": (ml_prediction or {}).get("category", ""),
        "ml_priority": (ml_prediction or {}).get("priority", ""),
        "ml_category_confidence": (ml_prediction or {}).get("category_confidence", 0.0),
        "ml_priority_confidence": (ml_prediction or {}).get("priority_confidence", 0.0),
    }


def classify_for_retrieval(question: str) -> Dict:
    """
    Runs a lightweight early classification pass before RAG retrieval.

    This is used only to guide the retriever toward the most likely support
    domain/source area. Final ticket category, priority, and routing are still
    produced later by classify_ticket().
    """
    question_lower = question.lower()
    ml_prediction = predict_ticket_torch(question) or predict_ticket_ml(question)
    rule_category = classify_category_with_rules(question_lower)
    predicted_category = choose_category(rule_category, ml_prediction)
    category_confidence = (ml_prediction or {}).get("category_confidence", 0.0)

    if predicted_category != "General IT Support":
        category_confidence = max(category_confidence, 0.55)

    retrieval_intents = CATEGORY_RETRIEVAL_INTENTS.get(predicted_category, [])
    mode = "targeted" if retrieval_intents and category_confidence >= CATEGORY_CONFIDENCE_THRESHOLD else "broad"

    return {
        "category": predicted_category,
        "confidence": round(category_confidence, 4),
        "retrieval_intents": retrieval_intents if mode == "targeted" else [],
        "retrieval_mode": mode,
        "fallback_reason": "" if mode == "targeted" else "No confident support domain was detected; using broad retrieval.",
        "model": (ml_prediction or {}).get("model", "rules_only"),
    }


def classify_category_with_rules(question_lower: str) -> str:
    category_scores = {
        "VPN Connectivity": 0,
        "Account Access": 0,
        "Multi-Factor Authentication": 0,
        "General IT Support": 1,
    }

    # VPN-related signals
    if any(keyword in question_lower for keyword in ["vpn", "network", "connect", "connection"]):
        category_scores["VPN Connectivity"] += 3

    # Account-related signals
    if any(keyword in question_lower for keyword in ["password", "locked", "login", "log in", "sign in", "account", "sso"]):
        category_scores["Account Access"] += 2

    # MFA-related signals
    if any(keyword in question_lower for keyword in ["duo", "mfa", "authentication", "push"]):
        category_scores["Multi-Factor Authentication"] += 3

    category = max(category_scores, key=category_scores.get)
    return category


def choose_category(rule_category: str, ml_prediction: Dict) -> str:
    if not ml_prediction:
        return rule_category

    ml_category = ml_prediction.get("category", rule_category)
    ml_confidence = ml_prediction.get("category_confidence", 0.0)

    if rule_category == "General IT Support" and ml_confidence >= CATEGORY_CONFIDENCE_THRESHOLD:
        return ml_category

    if ml_category == rule_category:
        return ml_category

    if ml_confidence >= 0.55:
        return ml_category

    return rule_category


def choose_priority(rule_priority: str, ml_prediction: Dict) -> str:
    if not ml_prediction:
        return rule_priority

    ml_priority = ml_prediction.get("priority", rule_priority)
    ml_confidence = ml_prediction.get("priority_confidence", 0.0)

    # Rule-based escalation signals are retained as guardrails for high-risk cases.
    if priority_rank(rule_priority) >= priority_rank("High"):
        return rule_priority

    if ml_confidence >= PRIORITY_CONFIDENCE_THRESHOLD:
        return ml_priority

    return rule_priority


def priority_rank(priority: str) -> int:
    return {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Critical": 4,
    }.get(priority, 1)


def assigned_team_for_category(category: str) -> str:
    team_mapping = {
        "VPN Connectivity": "Network Support",
        "Account Access": "Identity and Access Management",
        "Multi-Factor Authentication": "Identity and Access Management",
        "General IT Support": "Service Desk",
    }

    return team_mapping.get(category, "Service Desk")


def predict_priority(question_lower: str) -> str:
    if is_informational_routing_question(question_lower):
        return "Low"

    # Critical issues affect production systems, company-wide services, or security.
    if any(
        keyword in question_lower
        for keyword in [
            "outage",
            "down",
            "production",
            "company-wide",
            "company wide",
            "security breach",
            "security incident",
        ]
    ):
        return "Critical"

    # High priority issues affect multiple users, a department, or a team.
    if any(
        keyword in question_lower
        for keyword in [
            "multiple users",
            "many users",
            "several users",
            "department",
            "team blocked",
            "business-critical",
            "business critical",
        ]
    ):
        return "High"

    # Medium priority issues block one user or prevent normal work.
    if any(
        keyword in question_lower
        for keyword in [
            "cannot",
            "can't",
            "unable",
            "not working",
            "blocked",
            "failed",
            "locked",
            "error",
            "issue",
            "stopped",
            "broken",
            "not responding",
            "no response",
            "lost",
        ]
    ):
        return "Medium"

    return "Low"


def is_informational_routing_question(question_lower: str) -> bool:
    informational_starts = [
        "which support team",
        "which team",
        "who handles",
        "where should",
        "how should this be routed",
    ]

    return any(question_lower.startswith(prefix) for prefix in informational_starts)


def generate_summary(question: str) -> str:
    question = question.strip()

    if len(question) <= 80:
        return question

    return question[:77] + "..."


if __name__ == "__main__":
    test_questions = [
        "My VPN is not working after I reset my password",
        "I cannot approve Duo push notifications",
        "My account is locked",
        "Company-wide authentication failure",
        "Multiple users cannot access VPN",
    ]

    for question in test_questions:
        print("\nQuestion:", question)
        print(classify_ticket(question))

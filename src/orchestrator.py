import asyncio
from time import perf_counter
from typing import Dict, List

try:
    from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

    LLAMA_INDEX_WORKFLOWS_AVAILABLE = True
except Exception:
    Event = object
    StartEvent = object
    StopEvent = object
    Workflow = object

    def step(func):
        return func

    LLAMA_INDEX_WORKFLOWS_AVAILABLE = False

from src.config import CLARIFY_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD
from src.generator import answer_question


SUPPORT_SIGNALS = {
    "VPN Connectivity": ["vpn", "network", "connect", "connection"],
    "Account Access": ["password", "locked", "login", "sign in", "account"],
    "Multi-Factor Authentication": ["duo", "mfa", "authentication", "push"],
}


def run_support_workflow(question: str, domain: str = "it_support") -> Dict:
    """
    Coordinates the support workflow around the RAG answer.

    LlamaIndex Workflows is used when installed. A compatible sequential runner
    keeps local tests and offline demos deterministic if the optional workflow
    package is not available in the current environment.
    """
    if LLAMA_INDEX_WORKFLOWS_AVAILABLE:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(run_support_workflow_async(question, domain))

        return run_sequential_support_workflow(
            question,
            domain,
            workflow_engine="llamaindex_workflows_sync_fallback",
        )

    return run_sequential_support_workflow(
        question,
        domain,
        workflow_engine="llamaindex_workflows_compatible_fallback",
    )


async def run_support_workflow_async(question: str, domain: str = "it_support") -> Dict:
    workflow = SupportAutomationWorkflow(timeout=30, verbose=False)
    return await workflow.run(question=question, domain=domain)


def run_sequential_support_workflow(
    question: str,
    domain: str,
    workflow_engine: str,
) -> Dict:
    state = build_initial_state(question, domain)
    state["workflow_engine"] = workflow_engine

    workflow_start = perf_counter()
    run_answer_stage(state)
    run_decision_stage(state)
    run_ticket_stage(state)
    finalize_workflow_metrics(state, workflow_start)

    return state


if LLAMA_INDEX_WORKFLOWS_AVAILABLE:
    class AnswerGeneratedEvent(Event):
        state: Dict

    class DecisionReadyEvent(Event):
        state: Dict

    class TicketDraftReadyEvent(Event):
        state: Dict

    class SupportAutomationWorkflow(Workflow):
        @step
        async def generate_answer(self, ev: StartEvent) -> AnswerGeneratedEvent:
            state = build_initial_state(ev.question, ev.domain)
            state["workflow_engine"] = "llamaindex_workflows"
            state["_workflow_start"] = perf_counter()
            run_answer_stage(state)
            return AnswerGeneratedEvent(state=state)

        @step
        async def score_and_decide(self, ev: AnswerGeneratedEvent) -> DecisionReadyEvent:
            state = ev.state
            run_decision_stage(state)
            return DecisionReadyEvent(state=state)

        @step
        async def prepare_ticket(self, ev: DecisionReadyEvent) -> TicketDraftReadyEvent:
            state = ev.state
            run_ticket_stage(state)
            return TicketDraftReadyEvent(state=state)

        @step
        async def finish(self, ev: TicketDraftReadyEvent) -> StopEvent:
            state = ev.state
            workflow_start = state.pop("_workflow_start", perf_counter())
            finalize_workflow_metrics(state, workflow_start)
            return StopEvent(result=state)

else:
    class SupportAutomationWorkflow:
        def __init__(self, timeout: int = 30, verbose: bool = False):
            self.timeout = timeout
            self.verbose = verbose

        async def run(self, question: str, domain: str = "it_support") -> Dict:
            return run_sequential_support_workflow(
                question,
                domain,
                workflow_engine="llamaindex_workflows_compatible_fallback",
            )


def run_answer_stage(state: Dict) -> None:
    start = perf_counter()
    question = state.get("question", "")
    domain = state.get("domain", "it_support")
    response = answer_question(question, domain=domain)
    state.update(
        {
            "answer": response.get("answer", ""),
            "sources": response.get("sources", []),
            "retrieved_chunks": response.get("retrieved_chunks", []),
            "ticket": response.get("ticket", {}),
            "answer_generation_mode": response.get("answer_generation_mode", "unknown"),
            "fallback_triggered": response.get("fallback", False),
        }
    )
    record_stage_latency(state, "answer_stage_latency_ms", start)
    state["engineering_metrics"]["retrieved_chunk_count"] = len(state.get("retrieved_chunks", []))
    state["engineering_metrics"]["source_count"] = len(state.get("sources", []))


def run_decision_stage(state: Dict) -> None:
    start = perf_counter()
    state["confidence"] = calculate_confidence(state)
    state["confusion_analysis"] = detect_confusion(state.get("question", ""))
    state["agent_decision"] = decide_next_action(state)
    record_stage_latency(state, "decision_stage_latency_ms", start)


def run_ticket_stage(state: Dict) -> None:
    start = perf_counter()
    state["ticket_draft"] = build_ticket_draft(state)
    record_stage_latency(state, "ticket_draft_stage_latency_ms", start)


def record_stage_latency(state: Dict, metric_name: str, start: float) -> None:
    state["engineering_metrics"][metric_name] = round((perf_counter() - start) * 1000, 2)


def finalize_workflow_metrics(state: Dict, workflow_start: float) -> None:
    metrics = state["engineering_metrics"]
    metrics["total_workflow_latency_ms"] = round((perf_counter() - workflow_start) * 1000, 2)
    metrics["workflow_engine"] = state.get("workflow_engine", "unknown")
    metrics["llamaindex_workflows_available"] = LLAMA_INDEX_WORKFLOWS_AVAILABLE


def build_initial_state(question: str, domain: str) -> Dict:
    return {
        "question": question,
        "domain": domain,
        "answer": "",
        "sources": [],
        "retrieved_chunks": [],
        "ticket": {},
        "answer_generation_mode": "unknown",
        "fallback_triggered": False,
        "confidence": {},
        "confusion_analysis": {},
        "agent_decision": {},
        "ticket_draft": {},
        "workflow_engine": "unknown",
        "engineering_metrics": {
            "answer_stage_latency_ms": 0.0,
            "decision_stage_latency_ms": 0.0,
            "ticket_draft_stage_latency_ms": 0.0,
            "total_workflow_latency_ms": 0.0,
            "retrieved_chunk_count": 0,
            "source_count": 0,
            "workflow_engine": "unknown",
            "llamaindex_workflows_available": LLAMA_INDEX_WORKFLOWS_AVAILABLE,
        },
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

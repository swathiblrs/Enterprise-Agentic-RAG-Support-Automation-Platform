from typing import Dict, List

from src.config import LLM_MODEL_NAME, OPENAI_API_KEY, USE_LLM_GENERATION
from src.retriever import retrieve_relevant_chunks
from src.ticket_classifier import classify_ticket


def answer_question(question: str) -> Dict:
    """
    Generates a grounded support answer using retrieved knowledge chunks
    and adds ticket classification metadata.
    """
    retrieved_chunks = retrieve_relevant_chunks(question, top_k=3)
    ticket = classify_ticket(question)

    if not retrieved_chunks:
        return {
            "answer": (
                "I could not find enough information in the knowledge base to answer this question. "
                "Please contact the Service Desk for further assistance."
            ),
            "sources": [],
            "retrieved_chunks": [],
            "ticket": ticket,
            "answer_generation_mode": "fallback",
            "fallback": True,
        }

    answer, generation_mode = generate_grounded_answer(question, retrieved_chunks)

    sources = []
    for chunk in retrieved_chunks:
        if chunk["source"] not in sources:
            sources.append(chunk["source"])

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "ticket": ticket,
        "answer_generation_mode": generation_mode,
        "fallback": False,
    }


def generate_grounded_answer(question: str, retrieved_chunks: List[Dict]) -> tuple:
    """
    Creates a grounded support answer from retrieved context.
    """
    if USE_LLM_GENERATION and OPENAI_API_KEY:
        llm_answer = generate_llm_grounded_answer(question, retrieved_chunks)

        if llm_answer:
            return llm_answer, "llm"

    return generate_rule_based_answer(question, retrieved_chunks), "rule_based"


def generate_llm_grounded_answer(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Uses an LLM to answer only from retrieved context.

    This path is optional so the project remains testable without external API
    access. If the LLM call fails, the caller falls back to deterministic logic.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return ""

    context_blocks = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"Source {index}: {chunk['source']}\n{chunk['text']}"
        )

    prompt = f"""
You are an enterprise IT support assistant.

Answer the user's question using only the retrieved knowledge-base context.
If the context is not enough, say that the Service Desk should review it.
Do not invent policy, tools, URLs, phone numbers, or escalation paths.
Include a short "Sources" line listing the source file names used.

User question:
{question}

Retrieved context:
{chr(10).join(context_blocks)}
"""

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=0,
            api_key=OPENAI_API_KEY,
        )
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as error:
        print(f"LLM generation unavailable: {error}")
        return ""


def generate_rule_based_answer(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Creates a deterministic grounded support answer from retrieved context.
    """
    question_lower = question.lower()
    context_text = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
    context_lower = context_text.lower()

    if "vpn" in question_lower:
        return (
            "Based on the knowledge base, this appears to be a VPN connectivity issue. "
            "Please verify your internet connection, confirm that your VPN client is updated, "
            "restart the VPN application, and try signing in again. "
            "If the issue started after a password reset, make sure you are using the updated password "
            "and reconnecting through the VPN client."
        )

    if any(keyword in question_lower for keyword in ["password", "locked", "login", "sign in", "account"]):
        return (
            "Based on the knowledge base, this appears to be an account access issue. "
            "Please try resetting your password through the approved password reset process. "
            "If your account is locked, wait for the lockout period to expire or contact the Identity "
            "and Access Management team for account unlock support."
        )

    if any(keyword in question_lower for keyword in ["duo", "mfa", "authentication", "push"]):
        return (
            "Based on the knowledge base, this appears to be a multi-factor authentication issue. "
            "Please check your Duo/MFA device, ensure push notifications are enabled, verify network access, "
            "and try approving the request again. If the issue continues, contact the Identity and Access "
            "Management team."
        )

    if any(keyword in context_lower for keyword in ["critical", "high", "priority", "outage"]):
        return (
            "Based on the retrieved knowledge base content, this request may require priority review. "
            "Please include the number of affected users, business impact, and whether this is blocking "
            "production or normal work."
        )

    return (
        "Based on the retrieved knowledge base content, this request should be reviewed by the Service Desk. "
        "Please provide the error message, affected system, number of impacted users, and steps already tried."
    )


if __name__ == "__main__":
    test_questions = [
        "My VPN is not working after I reset my password",
        "I cannot approve Duo push notifications",
        "My account is locked",
        "Multiple users cannot access VPN",
        "Company-wide authentication failure",
    ]

    for question in test_questions:
        print("\nQuestion:", question)
        result = answer_question(question)
        print("Answer:", result["answer"])
        print("Sources:", result["sources"])
        print("Ticket:", result["ticket"])
        print("Fallback:", result["fallback"])

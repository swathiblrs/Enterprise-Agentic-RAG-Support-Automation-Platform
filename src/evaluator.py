import json
from pathlib import Path
from time import time

from src.orchestrator import run_support_workflow


EVAL_FILE = Path("tests/eval_questions.json")


def load_eval_questions():
    with EVAL_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def contains_expected_source(actual_sources, expected_sources):
    return any(source in actual_sources for source in expected_sources)


def calculate_precision_at_k(actual_sources, expected_sources):
    if not actual_sources:
        return 0.0

    relevant = sum(1 for source in actual_sources if source in expected_sources)
    return relevant / len(actual_sources)


def calculate_recall_at_k(actual_sources, expected_sources):
    if not expected_sources:
        return 1.0

    retrieved_expected = sum(1 for source in expected_sources if source in actual_sources)
    return retrieved_expected / len(expected_sources)


def evaluate_groundedness(response):
    return bool(response["sources"]) and not response["fallback_triggered"]


def evaluate_faithfulness(response):
    answer = response["answer"].lower()

    unsupported_phrases = [
        "guaranteed",
        "definitely resolved",
        "without review",
        "ignore policy",
    ]

    return not any(phrase in answer for phrase in unsupported_phrases)


def run_evaluation():
    questions = load_eval_questions()

    total = len(questions)

    retrieval_correct = 0
    category_correct = 0
    team_correct = 0
    priority_correct = 0
    safe_decision_count = 0
    grounded_count = 0
    faithful_count = 0
    total_precision = 0
    total_recall = 0
    total_confidence = 0
    total_latency = 0

    print("\nRunning evaluation...\n")

    for item in questions:
        question = item["question"]

        start_time = time()
        response = run_support_workflow(question)
        latency_ms = round((time() - start_time) * 1000, 2)

        total_latency += latency_ms

        actual_sources = response["sources"]
        actual_ticket = response["ticket"]
        confidence = response["confidence"]
        agent_decision = response["agent_decision"]

        retrieval_ok = contains_expected_source(
            actual_sources,
            item["expected_sources"],
        )

        category_ok = actual_ticket["category"] == item["expected_category"]
        team_ok = actual_ticket["assigned_team"] == item["expected_team"]
        priority_ok = actual_ticket["priority"] == item["expected_priority"]
        precision_at_k = calculate_precision_at_k(actual_sources, item["expected_sources"])
        recall_at_k = calculate_recall_at_k(actual_sources, item["expected_sources"])
        grounded = evaluate_groundedness(response)
        faithful = evaluate_faithfulness(response)
        safe_decision = agent_decision["next_action"] in [
            "answer_and_create_ticket_draft",
            "ask_clarifying_question",
            "create_urgent_ticket_draft",
            "escalate_to_human",
        ]

        retrieval_correct += int(retrieval_ok)
        category_correct += int(category_ok)
        team_correct += int(team_ok)
        priority_correct += int(priority_ok)
        safe_decision_count += int(safe_decision)
        grounded_count += int(grounded)
        faithful_count += int(faithful)
        total_precision += precision_at_k
        total_recall += recall_at_k
        total_confidence += confidence["overall_confidence"]

        print(f"Question: {question}")
        print(f"Actual Sources: {actual_sources}")
        print(f"Expected Sources: {item['expected_sources']}")
        print(f"Actual Ticket: {actual_ticket}")
        print(f"Retrieval Correct: {retrieval_ok}")
        print(f"Category Correct: {category_ok}")
        print(f"Team Correct: {team_ok}")
        print(f"Priority Correct: {priority_ok}")
        print(f"Precision@K: {precision_at_k:.2f}")
        print(f"Recall@K: {recall_at_k:.2f}")
        print(f"Grounded: {grounded}")
        print(f"Faithful: {faithful}")
        print(f"Agent Decision: {agent_decision}")
        print(f"Overall Confidence: {confidence['overall_confidence']}")
        print(f"Latency: {latency_ms} ms")
        print("-" * 70)

    print("\nEvaluation Summary")
    print("=" * 70)
    print(f"Total Questions: {total}")
    print(f"Retrieval Accuracy: {retrieval_correct / total:.2%}")
    print(f"Category Accuracy: {category_correct / total:.2%}")
    print(f"Team Routing Accuracy: {team_correct / total:.2%}")
    print(f"Priority Accuracy: {priority_correct / total:.2%}")
    print(f"Average Precision@K: {total_precision / total:.2%}")
    print(f"Average Recall@K: {total_recall / total:.2%}")
    print(f"Grounded Answer Rate: {grounded_count / total:.2%}")
    print(f"Faithfulness Heuristic Rate: {faithful_count / total:.2%}")
    print(f"Safe Agent Decision Rate: {safe_decision_count / total:.2%}")
    print(f"Average Overall Confidence: {total_confidence / total:.2f}")
    print(f"Average Latency: {total_latency / total:.2f} ms")


if __name__ == "__main__":
    run_evaluation()

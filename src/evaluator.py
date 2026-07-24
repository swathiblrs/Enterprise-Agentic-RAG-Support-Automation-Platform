import json
from pathlib import Path
from math import log2
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


def calculate_top_1_accuracy(actual_sources, expected_sources):
    if not actual_sources:
        return 0.0

    return 1.0 if actual_sources[0] in expected_sources else 0.0


def calculate_mrr(actual_sources, expected_sources):
    for index, source in enumerate(actual_sources, start=1):
        if source in expected_sources:
            return 1 / index

    return 0.0


def calculate_ndcg_at_k(actual_sources, expected_sources):
    if not actual_sources:
        return 0.0

    dcg = 0.0
    for index, source in enumerate(actual_sources, start=1):
        if source in expected_sources:
            dcg += 1 / log2(index + 1)

    ideal_relevant_count = min(len(expected_sources), len(actual_sources))
    if ideal_relevant_count == 0:
        return 1.0

    ideal_dcg = sum(1 / log2(index + 1) for index in range(1, ideal_relevant_count + 1))

    return dcg / ideal_dcg if ideal_dcg else 0.0


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
    total_top_1_accuracy = 0
    total_mrr = 0
    total_ndcg = 0
    total_confidence = 0
    total_latency = 0
    total_workflow_latency = 0
    total_answer_stage_latency = 0
    total_retrieved_chunks = 0
    total_source_count = 0

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
        engineering_metrics = response.get("engineering_metrics", {})

        retrieval_ok = contains_expected_source(
            actual_sources,
            item["expected_sources"],
        )

        category_ok = actual_ticket["category"] == item["expected_category"]
        team_ok = actual_ticket["assigned_team"] == item["expected_team"]
        priority_ok = actual_ticket["priority"] == item["expected_priority"]
        precision_at_k = calculate_precision_at_k(actual_sources, item["expected_sources"])
        recall_at_k = calculate_recall_at_k(actual_sources, item["expected_sources"])
        top_1_accuracy = calculate_top_1_accuracy(actual_sources, item["expected_sources"])
        mrr = calculate_mrr(actual_sources, item["expected_sources"])
        ndcg_at_k = calculate_ndcg_at_k(actual_sources, item["expected_sources"])
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
        total_top_1_accuracy += top_1_accuracy
        total_mrr += mrr
        total_ndcg += ndcg_at_k
        total_confidence += confidence["overall_confidence"]
        total_workflow_latency += engineering_metrics.get("total_workflow_latency_ms", latency_ms)
        total_answer_stage_latency += engineering_metrics.get("answer_stage_latency_ms", latency_ms)
        total_retrieved_chunks += engineering_metrics.get("retrieved_chunk_count", len(response.get("retrieved_chunks", [])))
        total_source_count += engineering_metrics.get("source_count", len(actual_sources))

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
        print(f"Top-1 Source Accuracy: {top_1_accuracy:.2f}")
        print(f"MRR: {mrr:.2f}")
        print(f"nDCG@K: {ndcg_at_k:.2f}")
        print(f"Grounded: {grounded}")
        print(f"Faithful: {faithful}")
        print(f"Agent Decision: {agent_decision}")
        print(f"Overall Confidence: {confidence['overall_confidence']}")
        print(f"Workflow Engine: {response.get('workflow_engine', 'unknown')}")
        print(f"Workflow Latency: {engineering_metrics.get('total_workflow_latency_ms', latency_ms)} ms")
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
    print(f"Top-1 Source Accuracy: {total_top_1_accuracy / total:.2%}")
    print(f"Mean Reciprocal Rank: {total_mrr / total:.2%}")
    print(f"Average nDCG@K: {total_ndcg / total:.2%}")
    print(f"Grounded Answer Rate: {grounded_count / total:.2%}")
    print(f"Faithfulness Heuristic Rate: {faithful_count / total:.2%}")
    print(f"Safe Agent Decision Rate: {safe_decision_count / total:.2%}")
    print(f"Average Overall Confidence: {total_confidence / total:.2f}")
    print(f"Average Latency: {total_latency / total:.2f} ms")
    print(f"Average Workflow Latency: {total_workflow_latency / total:.2f} ms")
    print(f"Average Answer Stage Latency: {total_answer_stage_latency / total:.2f} ms")
    print(f"Average Retrieved Chunks: {total_retrieved_chunks / total:.2f}")
    print(f"Average Source Count: {total_source_count / total:.2f}")


if __name__ == "__main__":
    run_evaluation()

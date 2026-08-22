import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional


TRAINING_DATA_FILE = Path("data/ticket_training_data.json")
MULTITASK_DATA_FILE = Path("data/ticket_multitask_dataset.jsonl")


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def load_training_examples() -> List[Dict[str, str]]:
    multitask_examples = load_multitask_training_examples()
    if multitask_examples:
        return multitask_examples

    if not TRAINING_DATA_FILE.exists():
        return []

    with TRAINING_DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_multitask_training_examples() -> List[Dict[str, str]]:
    if not MULTITASK_DATA_FILE.exists():
        return []

    examples = []
    with MULTITASK_DATA_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            example = json.loads(line)
            if example.get("split") in {"train", "validation"}:
                examples.append(
                    {
                        "text": example["text"],
                        "category": example["category"],
                        "priority": example["priority"],
                    }
                )

    return examples


@lru_cache(maxsize=1)
def get_ticket_models() -> Optional[Dict]:
    examples = load_training_examples()
    if not examples:
        return None

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError:
        return None

    texts = [normalize_text(example["text"]) for example in examples]
    categories = [example["category"] for example in examples]
    priorities = [example["priority"] for example in examples]

    category_model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words="english",
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    priority_model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words="english",
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    category_model.fit(texts, categories)
    priority_model.fit(texts, priorities)

    return {
        "category_model": category_model,
        "priority_model": priority_model,
        "training_examples": len(examples),
    }


def predict_ticket_ml(question: str) -> Optional[Dict]:
    models = get_ticket_models()
    if not models:
        return None

    normalized_question = normalize_text(question)
    category_model = models["category_model"]
    priority_model = models["priority_model"]

    category = str(category_model.predict([normalized_question])[0])
    priority = str(priority_model.predict([normalized_question])[0])

    return {
        "category": category,
        "priority": priority,
        "category_confidence": max_probability(category_model, normalized_question),
        "priority_confidence": max_probability(priority_model, normalized_question),
        "model": "tfidf_logistic_regression",
        "training_examples": models["training_examples"],
    }


def max_probability(model, text: str) -> float:
    if not hasattr(model, "predict_proba"):
        return 0.0

    probabilities = model.predict_proba([text])[0]
    return round(float(max(probabilities)), 4)


def evaluate_ticket_models(eval_items: List[Dict]) -> Dict:
    predictions = []
    category_true = []
    category_pred = []
    priority_true = []
    priority_pred = []

    for item in eval_items:
        prediction = predict_ticket_ml(item["question"])
        if not prediction:
            continue

        predictions.append(prediction)
        category_true.append(item["expected_category"])
        category_pred.append(prediction["category"])
        priority_true.append(item["expected_priority"])
        priority_pred.append(prediction["priority"])

    if not predictions:
        return {
            "available": False,
            "model": "unavailable",
            "training_examples": 0,
            "category_accuracy": 0.0,
            "category_weighted_f1": 0.0,
            "priority_accuracy": 0.0,
            "priority_weighted_f1": 0.0,
        }

    from sklearn.metrics import accuracy_score, f1_score

    models = get_ticket_models() or {}

    return {
        "available": True,
        "model": "tfidf_logistic_regression",
        "training_examples": models.get("training_examples", 0),
        "category_accuracy": round(accuracy_score(category_true, category_pred), 4),
        "category_weighted_f1": round(
            f1_score(category_true, category_pred, average="weighted", zero_division=0),
            4,
        ),
        "priority_accuracy": round(accuracy_score(priority_true, priority_pred), 4),
        "priority_weighted_f1": round(
            f1_score(priority_true, priority_pred, average="weighted", zero_division=0),
            4,
        ),
    }

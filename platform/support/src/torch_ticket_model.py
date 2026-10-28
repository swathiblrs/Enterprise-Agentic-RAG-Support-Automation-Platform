import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn


DATASET_FILE = Path("data/ticket_multitask_dataset.jsonl")
CHECKPOINT_FILE = Path("models/ticket_multitask.pt")
CATEGORY_LABELS = [
    "Account Access",
    "General IT Support",
    "Multi-Factor Authentication",
    "VPN Connectivity",
]
PRIORITY_LABELS = ["Low", "Medium", "High", "Critical"]
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
HASHING_EMBEDDING_DIM = 384
CATEGORY_CONFIDENCE_THRESHOLD = 0.35
PRIORITY_CONFIDENCE_THRESHOLD = 0.35


def load_multitask_examples(split: Optional[str] = None) -> List[Dict[str, str]]:
    if not DATASET_FILE.exists():
        return []

    examples = []
    with DATASET_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            example = json.loads(line)
            if split is None or example.get("split") == split:
                examples.append(example)

    return examples


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


class EmbeddingEncoder:
    def __init__(self, prefer_sentence_transformer: bool = True):
        self.backend = "hashing_vectorizer"
        self.model = None
        self.vectorizer = None

        if prefer_sentence_transformer:
            self.model = self.load_sentence_transformer()
            if self.model is not None:
                self.backend = "sentence_transformer"
                return

        from sklearn.feature_extraction.text import HashingVectorizer

        self.vectorizer = HashingVectorizer(
            n_features=HASHING_EMBEDDING_DIM,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )

    def load_sentence_transformer(self):
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            try:
                return SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, local_files_only=True)
            except TypeError:
                return None
        except Exception:
            return None

    def encode(self, texts: List[str]) -> np.ndarray:
        normalized_texts = [normalize_text(text) for text in texts]

        if self.backend == "sentence_transformer":
            embeddings = self.model.encode(
                normalized_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.astype("float32")

        embeddings = self.vectorizer.transform(normalized_texts).toarray()
        return embeddings.astype("float32")


class TicketEmbeddingDataset:
    def __init__(self, examples: List[Dict[str, str]], encoder: EmbeddingEncoder):
        import torch
        from torch.utils.data import TensorDataset

        self.examples = examples
        self.category_to_id = {label: index for index, label in enumerate(CATEGORY_LABELS)}
        self.priority_to_id = {label: index for index, label in enumerate(PRIORITY_LABELS)}

        embeddings = encoder.encode([example["text"] for example in examples])
        category_labels = [self.category_to_id[example["category"]] for example in examples]
        priority_labels = [self.priority_to_id[example["priority"]] for example in examples]

        self.dataset = TensorDataset(
            torch.tensor(embeddings, dtype=torch.float32),
            torch.tensor(category_labels, dtype=torch.long),
            torch.tensor(priority_labels, dtype=torch.long),
        )
        self.input_dim = embeddings.shape[1]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]


class MultiTaskTicketClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.category_head = nn.Linear(hidden_dim // 2, len(CATEGORY_LABELS))
        self.priority_head = nn.Linear(hidden_dim // 2, len(PRIORITY_LABELS))

    def forward(self, embeddings):
        features = self.shared(embeddings)
        return self.category_head(features), self.priority_head(features)


def class_weights(examples: List[Dict[str, str]], label_key: str, labels: List[str]):
    import torch

    counts = {label: 0 for label in labels}
    for example in examples:
        counts[example[label_key]] += 1

    total = sum(counts.values())
    weights = []
    for label in labels:
        count = max(counts[label], 1)
        weights.append(total / (len(labels) * count))

    return torch.tensor(weights, dtype=torch.float32)


def train_multitask_model(
    epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    patience: int = 6,
    checkpoint_path: Path = CHECKPOINT_FILE,
    prefer_sentence_transformer: bool = True,
) -> Dict:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    train_examples = load_multitask_examples("train")
    validation_examples = load_multitask_examples("validation")

    if not train_examples or not validation_examples:
        raise ValueError("Train and validation examples are required.")

    torch.manual_seed(42)
    encoder = EmbeddingEncoder(prefer_sentence_transformer=prefer_sentence_transformer)
    train_dataset = TicketEmbeddingDataset(train_examples, encoder)
    validation_dataset = TicketEmbeddingDataset(validation_examples, encoder)

    device = select_device()
    model = MultiTaskTicketClassifier(train_dataset.input_dim).to(device)

    category_loss = nn.CrossEntropyLoss(
        weight=class_weights(train_examples, "category", CATEGORY_LABELS).to(device)
    )
    priority_loss = nn.CrossEntropyLoss(
        weight=class_weights(train_examples, "priority", PRIORITY_LABELS).to(device)
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = run_epoch(model, train_loader, category_loss, priority_loss, optimizer, device)
        model.eval()
        validation_loss = run_epoch(model, validation_loader, category_loss, priority_loss, None, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "validation_loss": round(validation_loss, 4),
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                input_dim=train_dataset.input_dim,
                embedding_backend=encoder.backend,
                history=history,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    metrics = evaluate_checkpoint(checkpoint_path, split="test")
    metrics.update(
        {
            "best_epoch": best_epoch,
            "best_validation_loss": round(best_validation_loss, 4),
            "device": device,
            "embedding_backend": encoder.backend,
            "checkpoint": str(checkpoint_path),
        }
    )
    return metrics


def run_epoch(model, loader, category_loss, priority_loss, optimizer, device: str) -> float:
    import torch

    total_loss = 0.0
    total_examples = 0

    for embeddings, category_labels, priority_labels in loader:
        embeddings = embeddings.to(device)
        category_labels = category_labels.to(device)
        priority_labels = priority_labels.to(device)

        if optimizer:
            optimizer.zero_grad()

        category_logits, priority_logits = model(embeddings)
        loss = category_loss(category_logits, category_labels) + priority_loss(
            priority_logits,
            priority_labels,
        )

        if optimizer:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu()) * embeddings.size(0)
        total_examples += embeddings.size(0)

    return total_loss / max(total_examples, 1)


def save_checkpoint(
    checkpoint_path: Path,
    model,
    input_dim: int,
    embedding_backend: str,
    history: List[Dict],
) -> None:
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": 128,
            "dropout": 0.25,
            "category_labels": CATEGORY_LABELS,
            "priority_labels": PRIORITY_LABELS,
            "embedding_backend": embedding_backend,
            "history": history,
            "category_confidence_threshold": CATEGORY_CONFIDENCE_THRESHOLD,
            "priority_confidence_threshold": PRIORITY_CONFIDENCE_THRESHOLD,
        },
        checkpoint_path,
    )


def load_checkpoint(checkpoint_path: Path = CHECKPOINT_FILE) -> Optional[Dict]:
    if not checkpoint_path.exists():
        return None

    import torch

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=select_device(),
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=select_device())
    model = MultiTaskTicketClassifier(
        checkpoint["input_dim"],
        hidden_dim=checkpoint.get("hidden_dim", 128),
        dropout=checkpoint.get("dropout", 0.25),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(select_device())
    model.eval()
    checkpoint["model"] = model
    return checkpoint


def predict_ticket_torch(question: str, checkpoint_path: Path = CHECKPOINT_FILE) -> Optional[Dict]:
    checkpoint = load_checkpoint(checkpoint_path)
    if not checkpoint:
        return None

    import torch

    encoder = EmbeddingEncoder(
        prefer_sentence_transformer=checkpoint.get("embedding_backend") == "sentence_transformer"
    )
    if encoder.backend != checkpoint.get("embedding_backend"):
        return None

    embedding = encoder.encode([question])
    device = select_device()
    input_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

    with torch.no_grad():
        category_logits, priority_logits = checkpoint["model"](input_tensor)
        category_probabilities = torch.softmax(category_logits, dim=1).cpu().numpy()[0]
        priority_probabilities = torch.softmax(priority_logits, dim=1).cpu().numpy()[0]

    category_index = int(category_probabilities.argmax())
    priority_index = int(priority_probabilities.argmax())
    category_confidence = round(float(category_probabilities[category_index]), 4)
    priority_confidence = round(float(priority_probabilities[priority_index]), 4)

    return {
        "category": CATEGORY_LABELS[category_index],
        "priority": PRIORITY_LABELS[priority_index],
        "category_confidence": category_confidence,
        "priority_confidence": priority_confidence,
        "model": "pytorch_multitask_ticket_classifier",
        "embedding_backend": checkpoint.get("embedding_backend", "unknown"),
        "category_threshold_met": category_confidence >= CATEGORY_CONFIDENCE_THRESHOLD,
        "priority_threshold_met": priority_confidence >= PRIORITY_CONFIDENCE_THRESHOLD,
    }


def evaluate_checkpoint(checkpoint_path: Path = CHECKPOINT_FILE, split: str = "test") -> Dict:
    examples = load_multitask_examples(split)
    predictions = []

    for example in examples:
        prediction = predict_ticket_torch(example["text"], checkpoint_path)
        if prediction:
            predictions.append((example, prediction))

    if not examples or len(predictions) != len(examples):
        return empty_metrics("pytorch_multitask_ticket_classifier")

    return classification_metrics(
        [example["category"] for example, _ in predictions],
        [prediction["category"] for _, prediction in predictions],
        [example["priority"] for example, _ in predictions],
        [prediction["priority"] for _, prediction in predictions],
        model_name="pytorch_multitask_ticket_classifier",
        examples=len(examples),
    )


def evaluate_logistic_regression_baseline(split: str = "test") -> Dict:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError:
        return empty_metrics("tfidf_logistic_regression")

    train_examples = load_multitask_examples("train") + load_multitask_examples("validation")
    test_examples = load_multitask_examples(split)

    if not train_examples or not test_examples:
        return empty_metrics("tfidf_logistic_regression")

    category_model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    priority_model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ]
    )

    train_texts = [normalize_text(example["text"]) for example in train_examples]
    category_model.fit(train_texts, [example["category"] for example in train_examples])
    priority_model.fit(train_texts, [example["priority"] for example in train_examples])

    test_texts = [normalize_text(example["text"]) for example in test_examples]
    return classification_metrics(
        [example["category"] for example in test_examples],
        list(category_model.predict(test_texts)),
        [example["priority"] for example in test_examples],
        list(priority_model.predict(test_texts)),
        model_name="tfidf_logistic_regression",
        examples=len(test_examples),
    )


def classification_metrics(
    category_true: List[str],
    category_pred: List[str],
    priority_true: List[str],
    priority_pred: List[str],
    model_name: str,
    examples: int,
) -> Dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "available": True,
        "model": model_name,
        "examples": examples,
        "category_accuracy": round(float(accuracy_score(category_true, category_pred)), 4),
        "category_weighted_f1": round(
            float(f1_score(category_true, category_pred, average="weighted", zero_division=0)),
            4,
        ),
        "priority_accuracy": round(float(accuracy_score(priority_true, priority_pred)), 4),
        "priority_weighted_f1": round(
            float(f1_score(priority_true, priority_pred, average="weighted", zero_division=0)),
            4,
        ),
    }


def empty_metrics(model_name: str) -> Dict:
    return {
        "available": False,
        "model": model_name,
        "examples": 0,
        "category_accuracy": 0.0,
        "category_weighted_f1": 0.0,
        "priority_accuracy": 0.0,
        "priority_weighted_f1": 0.0,
    }

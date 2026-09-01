import os
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "vectorstore" / "chroma"

COLLECTION_NAME = "support_kb"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DOMAIN = os.getenv("DEFAULT_DOMAIN", "it_support")

INTENT_SOURCE_MAP = {
    "vpn": "vpn_troubleshooting_kb.md",
    "account": "password_reset_kb.md",
    "mfa": "mfa_duo_kb.md",
    "priority": "priority_matrix.md",
    "routing": "ticket_routing_rules.md",
}

INTENT_KEYWORDS = {
    "vpn": ["vpn", "network", "connect", "connection", "disconnect", "remote"],
    "account": ["password", "locked", "login", "log in", "sign in", "account", "sso"],
    "mfa": ["duo", "mfa", "authentication", "push", "phone", "device"],
    "priority": [
        "multiple users",
        "many users",
        "several users",
        "department",
        "company-wide",
        "company wide",
        "outage",
        "production",
        "security incident",
        "security breach",
    ],
    "routing": ["route", "routing", "assigned", "team", "support group"],
}


embedding_model = None


def get_embedding_model() -> Optional[SentenceTransformer]:
    """
    Lazily loads the embedding model so API imports and unit tests do not
    require network/model initialization before a retrieval request runs.
    """
    global embedding_model

    if embedding_model is not None:
        return embedding_model

    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception as error:
        print(f"Dense retrieval unavailable: {error}")
        return None

    return embedding_model


def load_knowledge_base_documents(domain: str = DEFAULT_DOMAIN) -> List[Dict]:
    """
    Loads knowledge base files from the data folder.
    These documents are used for keyword/BM25 retrieval.
    """
    documents = []

    for file_path in DATA_DIR.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in [".md", ".txt", ".pdf"]:
            continue

        document_domain = infer_domain(file_path)

        if domain and domain != "all" and document_domain != domain:
            continue

        text = extract_text(file_path)

        documents.append(
            {
                "source": str(file_path.relative_to(DATA_DIR)),
                "domain": document_domain,
                "source_type": infer_source_type(file_path),
                "text": text,
            }
        )

    return documents


def extract_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    return file_path.read_text(encoding="utf-8")


def infer_domain(file_path: Path) -> str:
    relative_path = file_path.relative_to(DATA_DIR)

    if len(relative_path.parts) > 1:
        return relative_path.parts[0]

    return DEFAULT_DOMAIN


def infer_source_type(file_path: Path) -> str:
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return "pdf"

    if extension == ".md":
        return "markdown"

    if extension == ".txt":
        return "text"

    return "unknown"


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25 keyword retrieval.
    """
    normalized = text.lower()
    for character in ["\n", "_", "-", ".", "/", "(", ")", ",", ":"]:
        normalized = normalized.replace(character, " ")

    return normalized.split()


def detect_query_intents(question: str) -> List[str]:
    question_lower = question.lower()
    intents = []

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            intents.append(intent)

    # If VPN and account signals both appear, keep VPN as the primary support
    # source because password-reset VPN failures usually need VPN remediation.
    if "vpn" in intents and "account" in intents:
        intents.remove("account")

    return intents


def expected_sources_for_intents(intents: List[str]) -> List[str]:
    sources = []

    for intent in intents:
        source = INTENT_SOURCE_MAP.get(intent)
        if source and source not in sources:
            sources.append(source)

    return sources


def get_chroma_collection():
    """
    Connects to the local ChromaDB vector store.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def dense_retrieve(question: str, top_k: int = 5, domain: str = DEFAULT_DOMAIN) -> List[Dict]:
    """
    Retrieves relevant chunks using vector similarity search from ChromaDB.
    """
    model = get_embedding_model()

    if model is None:
        return []

    collection = get_chroma_collection()
    query_embedding = model.encode(question).tolist()

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if domain and domain != "all":
        query_kwargs["where"] = {"domain": domain}

    results = collection.query(**query_kwargs)

    retrieved_chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        retrieved_chunks.append(
            {
                "text": document,
                "source": metadata.get("source", "unknown"),
                "domain": metadata.get("domain", domain),
                "source_type": metadata.get("source_type", "unknown"),
                "retrieval_method": "dense",
                "score": 1 / (1 + distance),
            }
        )

    return retrieved_chunks


def bm25_retrieve(question: str, top_k: int = 5, domain: str = DEFAULT_DOMAIN) -> List[Dict]:
    """
    Retrieves relevant documents using BM25 keyword search.
    """
    documents = load_knowledge_base_documents(domain=domain)

    if not documents:
        return []

    tokenized_docs = [tokenize(doc["text"]) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    tokenized_question = tokenize(question)
    scores = bm25.get_scores(tokenized_question)

    ranked_results = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    retrieved_chunks = []

    for document, score in ranked_results[:top_k]:
        if score <= 0:
            continue

        retrieved_chunks.append(
            {
                "text": document["text"],
                "source": document["source"],
                "domain": document["domain"],
                "source_type": document["source_type"],
                "retrieval_method": "bm25",
                "score": float(score),
            }
        )

    return retrieved_chunks


def merge_results(results: List[Dict]) -> List[Dict]:
    """
    Merges dense and BM25 results and removes duplicate sources/text.
    """
    seen = set()
    merged = []

    for result in results:
        unique_key = (result["source"], result["text"][:120])

        if unique_key not in seen:
            seen.add(unique_key)
            merged.append(result)

    return merged


def rerank_results(
    question: str,
    results: List[Dict],
    top_k: int = 3,
    retrieval_intents: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Lightweight reranking based on keyword overlap and retrieval score.
    This keeps the project simple while showing production-style reranking logic.
    """
    question_tokens = set(tokenize(question))
    query_intents = retrieval_intents or detect_query_intents(question)
    intent_sources = expected_sources_for_intents(query_intents)

    reranked = []

    for result in results:
        chunk_tokens = set(tokenize(result["text"]))
        source_tokens = set(tokenize(result.get("source", "")))
        keyword_overlap = len(question_tokens.intersection(chunk_tokens))
        source_overlap = len(question_tokens.intersection(source_tokens))
        intent_boost = 8 if result.get("source") in intent_sources else 0
        priority_boost = 3 if "priority" in query_intents and result.get("source") == "priority_matrix.md" else 0

        rerank_score = result["score"] + keyword_overlap + (source_overlap * 2) + intent_boost + priority_boost

        reranked.append(
            {
                **result,
                "rerank_score": rerank_score,
            }
        )

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    if intent_sources:
        intent_ranked = [
            result for result in reranked
            if result.get("source") in intent_sources
        ]

        if intent_ranked:
            return intent_ranked[: max(1, min(len(intent_sources), top_k))]

    return reranked[:top_k]


def retrieve_relevant_chunks(
    question: str,
    top_k: int = 3,
    domain: str = DEFAULT_DOMAIN,
    retrieval_intents: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Main retrieval function used by the RAG pipeline.

    It performs:
    1. Dense vector retrieval from ChromaDB
    2. Keyword retrieval using BM25
    3. Result merging
    4. Lightweight reranking
    """
    dense_results = dense_retrieve(question, top_k=5, domain=domain)
    bm25_results = bm25_retrieve(question, top_k=5, domain=domain)

    merged_results = merge_results(dense_results + bm25_results)
    reranked_results = rerank_results(
        question,
        merged_results,
        top_k=top_k,
        retrieval_intents=retrieval_intents,
    )

    return reranked_results


if __name__ == "__main__":
    test_question = "My VPN is not working after I reset my password"

    results = retrieve_relevant_chunks(test_question)

    print("\nQuestion:", test_question)
    print("\nRetrieved Chunks:")

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Source:", result["source"])
        print("Method:", result["retrieval_method"])
        print("Score:", round(result["score"], 4))
        print("Rerank Score:", round(result["rerank_score"], 4))
        print("Preview:", result["text"][:250].replace("\n", " "))

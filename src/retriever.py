from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "vectorstore" / "chroma"

COLLECTION_NAME = "support_kb"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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


def load_knowledge_base_documents() -> List[Dict]:
    """
    Loads markdown knowledge base files from the data folder.
    These documents are used for keyword/BM25 retrieval.
    """
    documents = []

    for file_path in DATA_DIR.glob("*.md"):
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "source": file_path.name,
                "text": text,
            }
        )

    return documents


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25 keyword retrieval.
    """
    return text.lower().replace("\n", " ").split()


def get_chroma_collection():
    """
    Connects to the local ChromaDB vector store.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def dense_retrieve(question: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieves relevant chunks using vector similarity search from ChromaDB.
    """
    model = get_embedding_model()

    if model is None:
        return []

    collection = get_chroma_collection()
    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved_chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        retrieved_chunks.append(
            {
                "text": document,
                "source": metadata.get("source", "unknown"),
                "retrieval_method": "dense",
                "score": 1 / (1 + distance),
            }
        )

    return retrieved_chunks


def bm25_retrieve(question: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieves relevant documents using BM25 keyword search.
    """
    documents = load_knowledge_base_documents()

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


def rerank_results(question: str, results: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Lightweight reranking based on keyword overlap and retrieval score.
    This keeps the project simple while showing production-style reranking logic.
    """
    question_tokens = set(tokenize(question))

    reranked = []

    for result in results:
        chunk_tokens = set(tokenize(result["text"]))
        keyword_overlap = len(question_tokens.intersection(chunk_tokens))

        rerank_score = result["score"] + keyword_overlap

        reranked.append(
            {
                **result,
                "rerank_score": rerank_score,
            }
        )

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    return reranked[:top_k]


def retrieve_relevant_chunks(question: str, top_k: int = 3) -> List[Dict]:
    """
    Main retrieval function used by the RAG pipeline.

    It performs:
    1. Dense vector retrieval from ChromaDB
    2. Keyword retrieval using BM25
    3. Result merging
    4. Lightweight reranking
    """
    dense_results = dense_retrieve(question, top_k=5)
    bm25_results = bm25_retrieve(question, top_k=5)

    merged_results = merge_results(dense_results + bm25_results)
    reranked_results = rerank_results(question, merged_results, top_k=top_k)

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

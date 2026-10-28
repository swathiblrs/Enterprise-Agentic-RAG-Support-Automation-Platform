"""Verify the existing retrieval collection against its original pretrained encoder."""
import hashlib
import json
import os
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parent.parent
client=chromadb.PersistentClient(path=str(ROOT/"support/vectorstore/chroma"))
collection=client.get_collection("support_kb")
data=collection.get(include=["documents","embeddings","metadatas"])
encoder=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",local_files_only=True,device="cpu")
expected=encoder.encode(data["documents"],show_progress_bar=False)
stored=np.asarray(data["embeddings"])
error=float(np.max(np.abs(stored-expected)))
digest=hashlib.sha256()
for index in sorted(range(len(data["ids"])),key=lambda i:data["ids"][i]):
    digest.update(data["ids"][index].encode())
    digest.update(data["documents"][index].encode())
    digest.update(json.dumps(data["metadatas"][index],sort_keys=True).encode())
    digest.update(np.asarray(stored[index],dtype="float32").tobytes())
report={"collection":"support_kb","chunks":len(data["ids"]),
        "encoder":"sentence-transformers/all-MiniLM-L6-v2","embedding_dim":int(stored.shape[1]),
        "max_abs_embedding_difference":error,"matches_original_retrieval_encoder":bool(error<1e-5),
        "collection_content_sha256":digest.hexdigest(),
        "note":"Chroma SQLite metadata can change on access; this checks stored content/vectors, not file byte identity."}
path=ROOT/"routing_training/reports/grouped-minilm-v1/retrieval_verification.json"
path.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if not report["matches_original_retrieval_encoder"]:raise SystemExit(1)

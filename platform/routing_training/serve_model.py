"""JSON-lines embedding inference; classification heads are JSON, never pickle."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
with contextlib.redirect_stdout(sys.stderr):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    manifest=json.loads(Path(sys.argv[1]).read_text())
    head_path=(ROOT/manifest["head"]).resolve()
    head_path.relative_to(ROOT)
    if hashlib.sha256(head_path.read_bytes()).hexdigest()!=manifest["head_sha256"]:
        raise ValueError("Classifier checksum mismatch")
    head=json.loads(head_path.read_text())
    if manifest.get("encoder"):
        encoder_path=(ROOT/manifest["encoder"]).resolve()
        encoder_path.relative_to(ROOT)
        encoder_source=str(encoder_path)
    elif manifest.get("encoder_model")== "sentence-transformers/all-MiniLM-L6-v2":
        encoder_source=manifest["encoder_model"]
    else:
        raise ValueError("No supported local encoder source")
    encoder=SentenceTransformer(encoder_source,revision=manifest.get("encoder_revision"),local_files_only=True,device="cpu")
    encoder.max_seq_length=int(manifest.get("max_seq_length",64))
    weights=np.asarray(head["coefficients"])
    bias=np.asarray(head["intercept"])

for line in sys.stdin:
    try:
        request=json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            features=encoder.encode([request["text"]],normalize_embeddings=True,
                                    convert_to_numpy=True,show_progress_bar=False)
        scores=features[0]@weights.T+bias
        probabilities=np.exp(scores-scores.max())
        probabilities/=probabilities.sum()
        response={"ok":True,"probabilities":probabilities.tolist()}
    except Exception:
        response={"ok":False}
    print(json.dumps(response),flush=True)

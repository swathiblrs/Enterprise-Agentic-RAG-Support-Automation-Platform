"""Validation-only timing: same CPU and JSON head for all model candidates."""
import json
from pathlib import Path
import sys
import time
import math

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from domain_classifier import probabilities

run_id=sys.argv[1]
out=ROOT/"routing_models"/run_id
report_dir=ROOT/"routing_training"/"reports"/run_id
rows=[json.loads(line) for line in (ROOT/"routing_training/data/validation.jsonl").read_text().splitlines()]
torch.set_num_threads(4)
report={}
for name,head_file,encoder_folder in [
        ("tfidf","tfidf.json",None),("frozen","frozen_head.json","frozen_encoder"),
        ("setfit","setfit_head.json","setfit_encoder")]:
    head=json.loads((out/head_file).read_text())
    if encoder_folder:
        encoder=SentenceTransformer(str(out/encoder_folder),device="cpu",local_files_only=True)
        weights=np.array(head["coefficients"]);bias=np.array(head["intercept"])
        def infer(text):
            features=encoder.encode([text],normalize_embeddings=True,show_progress_bar=False)
            scores=features[0]@weights.T+bias
            exps=np.exp(scores-scores.max())
            return exps/exps.sum()
    else:
        def infer(text):return probabilities(text,head)
    for row in rows[:3]:infer(row["text"])
    durations=[]
    for row in rows[:50]:
        started=time.perf_counter()
        infer(row["text"])
        durations.append((time.perf_counter()-started)*1000)
    durations.sort()
    report[name]={"device":"cpu","samples":50,"batch_size":1,"ipc_included":False,
                  "warm_p50_ms":durations[24],"warm_p95_ms":durations[math.ceil(.95*50)-1]}
(report_dir/"standardized_latency.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

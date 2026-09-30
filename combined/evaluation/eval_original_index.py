import contextlib
import json
import os
import sys
import time
from pathlib import Path
root=Path('/private/tmp/original-support-evaluation')
os.chdir(root)
sys.path.insert(0,str(root))
os.environ.update(USE_LLM_GENERATION='false', OPENAI_API_KEY='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
with contextlib.redirect_stdout(sys.stderr):
    from src.orchestrator import run_support_workflow
    from src.retriever import get_chroma_collection
    from src.torch_ticket_model import evaluate_checkpoint, evaluate_logistic_regression_baseline
    cases=json.loads((root/'tests/eval_questions.json').read_text())
    collection=get_chroma_collection()
    results=[]
    for case in cases:
        start=time.perf_counter();result=run_support_workflow(case['question'])
        results.append({'case':case,'result':result,'wall_ms':(time.perf_counter()-start)*1000})
    models={'pytorch':evaluate_checkpoint(),'logistic_regression':evaluate_logistic_regression_baseline()}
Path('/private/tmp/combined-evaluation/original_index_baseline.json').write_text(json.dumps({'index_count':collection.count(),'results':results,'models':models},indent=2))
print('Original indexed baseline completed',len(results),'cases; index count',collection.count())

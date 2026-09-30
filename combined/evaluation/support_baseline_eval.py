import contextlib
import json
import os
import sys
import time
from pathlib import Path

root=Path('/Users/swathibs/Desktop/projects/Combined Agentic RAG Prototype/support')
os.chdir(root)
sys.path.insert(0,str(root))
os.environ.update(USE_LLM_GENERATION='false', OPENAI_API_KEY='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
with contextlib.redirect_stdout(sys.stderr):
    from src.orchestrator import run_support_workflow
    cases=json.loads((root/'tests/eval_questions.json').read_text())
    results=[]
    for case in cases:
        start=time.perf_counter()
        result=run_support_workflow(case['question'])
        results.append({'case':case,'result':result,'wall_ms':(time.perf_counter()-start)*1000})
        print('baseline',len(results),file=sys.stderr,flush=True)
Path('/private/tmp/support_baseline_results.json').write_text(json.dumps(results,indent=2))

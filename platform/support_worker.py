"""Persistent JSON-lines worker using the support project's environment."""
import contextlib
import json
import os
import sys
from pathlib import Path

root = Path(__file__).parent / 'support'
os.chdir(root)
sys.path.insert(0, str(root))
with contextlib.redirect_stdout(sys.stderr):
    from src.orchestrator import run_support_workflow

for line in sys.stdin:
    try:
        request = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            result = run_support_workflow(request['question'])
        response = {'ok': True, 'result': result}
    except Exception as error:
        print(f'Support request failed: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        response = {'ok': False, 'error': 'Support workflow failed'}
    print(json.dumps(response), flush=True)

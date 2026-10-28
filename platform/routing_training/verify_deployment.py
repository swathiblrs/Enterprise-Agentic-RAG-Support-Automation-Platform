"""Check selected runtime against validation predictions, without reading test labels."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import routing_model

parser=argparse.ArgumentParser()
parser.add_argument("run_id")
args=parser.parse_args()
report_dir=ROOT/"routing_training"/"reports"/args.run_id
routing_model.MANIFEST=report_dir/"deployment.json"
routing_model.load_manifest.cache_clear()
selection=json.loads((report_dir/"selection.json").read_text())
cases={row["id"]:row for row in (json.loads(line) for line in
       (ROOT/"routing_training/data/validation.jsonl").read_text().splitlines())}
expected=selection["candidates"][selection["winner"]]["validation"]["cases"]
failures=[]
durations=[]
try:
    for index,row in enumerate(expected):
        start=time.perf_counter()
        prediction=routing_model.predict(cases[row["id"]]["text"])
        durations.append((time.perf_counter()-start)*1000)
        if prediction.get("domain")!=row["predicted"] or prediction.get("accepted")!=row["accepted"]:
            failures.append({"id":row["id"],"expected":row,"actual":prediction})
    result={"validation_cases":len(expected),"mismatches":failures,
            "cold_first_request_ms":durations[0],"warm_mean_with_ipc_ms":sum(durations[1:])/max(1,len(durations)-1)}
    (report_dir/"runtime_verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if failures:raise SystemExit(1)
finally:
    routing_model.embedding_worker.close()

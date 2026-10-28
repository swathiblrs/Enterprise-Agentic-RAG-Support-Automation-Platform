"""Development/regression check for the deployed hybrid router, not a fresh test."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from domain_router import route
from routing_model import embedding_worker

rows=json.loads((ROOT/"evaluation-v3/routing_cases.json").read_text())
results=[]
try:
    for row in rows:
        fields={key:value for key,value in row.items() if key not in ("actual","expected_domain","expected_path")}
        defaults={"domain":"auto","severity":"medium","logs":[],"events":[],"raw_events":[],"metrics":{},
                  "source":"manual","service":None,"tags":[],"title":None,"description":None}
        defaults.update(fields)
        decision=route(SimpleNamespace(**defaults))
        results.append({"question":row["question"],"expected_domain":row["expected_domain"],
                        "expected_path":row["expected_path"],"actual_domain":decision["domain"],
                        "actual_path":decision["path"],
                        "domain_correct":decision["domain"]==row["expected_domain"],
                        "path_correct":decision["path"]==row["expected_path"]})
    report={"scope":"Historical development/regression cases; not a held-out generalization metric.",
            "cases":len(results),"domain_correct":sum(r["domain_correct"] for r in results),
            "path_correct":sum(r["path_correct"] for r in results),"results":results}
    (ROOT/"routing_training/reports/grouped-minilm-v1/hybrid_regressions.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({key:value for key,value in report.items() if key!="results"},indent=2))
finally:
    embedding_worker.close()

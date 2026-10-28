"""Document-source retrieval evaluation; no answer-generation or training claims."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).parent


def normalized(text):
    return " ".join(re.findall(r"[a-z0-9]+",text.lower()))


def validate_cases(cases,sources):
    if len({row["id"] for row in cases})!=len(cases):
        raise ValueError("Duplicate question ids")
    if len({normalized(row["question"]) for row in cases})!=len(cases):
        raise ValueError("Duplicate normalized questions")
    for row in cases:
        if not row.get("scenario_group") or not row.get("stratum"):
            raise ValueError("Question grouping and provenance are required")
        expected=row["expected_sources"]
        if len(set(expected))!=len(expected):
            raise ValueError("Duplicate expected sources")
        if bool(expected)!=row["answerable"]:
            raise ValueError("Answerability and expected-source labels disagree")
        if any(source not in sources for source in expected):
            raise ValueError("Expected source is absent from the corpus")


def score_sources(actual,expected,k=3):
    if k<1:
        raise ValueError("k must be positive")
    # Match the source list shown by the application, without filling missing slots.
    actual=list(dict.fromkeys(actual[:k]))
    relevant=set(expected)
    if not relevant:
        return {"returned_sources":actual,"returned_context":bool(actual),
                "precision_at_k":None,"recall_at_k":None,"hit_at_k":None,
                "precision_over_returned":None,"top_1":None,"mrr":None,"ndcg_at_k":None}
    hits=len(set(actual)&relevant)
    ranks=[rank for rank,source in enumerate(actual,1) if source in relevant]
    ideal=sum(1/math.log2(rank+1) for rank in range(1,min(k,len(relevant))+1))
    dcg=sum(1/math.log2(rank+1) for rank in ranks)
    return {"returned_sources":actual,"returned_context":bool(actual),
            "precision_at_k":hits/k,"recall_at_k":hits/len(relevant),
            "hit_at_k":float(bool(hits)),"precision_over_returned":hits/len(actual) if actual else 0.,
            "top_1":float(bool(actual) and actual[0] in relevant),
            "mrr":1/ranks[0] if ranks else 0.,"ndcg_at_k":dcg/ideal}


METRICS=("precision_at_k","recall_at_k","hit_at_k","precision_over_returned","top_1","mrr","ndcg_at_k")


def summarize(rows,k):
    answerable=[row for row in rows if row["answerable"]]
    if not answerable:
        raise ValueError("No answerable evaluation cases")
    overall={metric:sum(row[metric] for row in answerable)/len(answerable) for metric in METRICS}
    groups=defaultdict(list)
    for row in answerable:
        groups[row["scenario_group"]].append(row)
    group_macro={metric:sum(sum(row[metric] for row in group)/len(group) for group in groups.values())/len(groups)
                 for metric in METRICS}
    strata={}
    for stratum in sorted({row["stratum"] for row in answerable}):
        values=[row for row in answerable if row["stratum"]==stratum]
        strata[stratum]={"questions":len(values),
                         **{metric:sum(row[metric] for row in values)/len(values) for metric in METRICS}}
    controls=[row for row in rows if not row["answerable"]]
    return {"questions":len(rows),"answerable_questions":len(answerable),
            "unsupported_controls":len(controls),"answerable_scenario_groups":len(groups),
            "overall":overall,"scenario_group_macro":group_macro,"by_stratum":strata,
            "expected_label_precision_ceiling":sum(min(k,len(row["expected_sources"]))/k for row in answerable)/len(answerable),
            "unsupported_returned_context":sum(row["returned_context"] for row in controls),
            "note":"Controls are excluded from precision/recall. Returned context alone is not an answerability decision."}


def evaluate(cases,retrieve,k):
    results=[]
    for index,row in enumerate(cases,1):
        started=time.perf_counter()
        chunks=retrieve(row["question"],top_k=k)
        sources=[chunk["source"] for chunk in chunks[:k]]
        results.append({**row,**score_sources(sources,row["expected_sources"],k),
                        "latency_ms":(time.perf_counter()-started)*1000})
        if index%20==0:
            print(f"Evaluated {index}/{len(cases)} questions",flush=True)
    return results


def run(dataset,output,k=3):
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0,str(ROOT/"support"))
    from src.retriever import get_embedding_model, retrieve_relevant_chunks
    dataset=Path(dataset)
    cases=json.loads(dataset.read_text())
    corpus=ROOT/"support/data"
    files=sorted(p for p in corpus.rglob("*") if p.suffix in (".md",".txt",".pdf") and p.is_file())
    sources={str(path.relative_to(corpus)) for path in files}
    validate_cases(cases,sources)
    started=time.perf_counter()
    results=evaluate(cases,retrieve_relevant_chunks,k)
    summary=summarize(results,k)
    encoder=get_embedding_model()
    report={"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),
            "dataset_sha256":hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "corpus_file_sha256":{str(path.relative_to(corpus)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
            "retriever_sha256":hashlib.sha256((ROOT/"support/src/retriever.py").read_bytes()).hexdigest(),
            "corpus_documents":len(files),"k":k,"summary":summary,"cases":results,
            "elapsed_seconds":time.perf_counter()-started,
            "encoder_device":str(encoder.device) if encoder is not None else "dense_unavailable",
            "evaluation_mode":"direct_fresh_retrieval_no_answer_cache_no_llm",
            "limitations":["Synthetic/project-authored labels, not independent human relevance judgments.",
                           "Expected-source lists are not exhaustive judgments of every possibly useful document.",
                           "100 answerable questions represent 80 scenario groups, not 100 independent incidents.",
                           "Existing development cases are reused; this is not an untouched external test.",
                           "No model or retrieval-rule tuning was performed during this evaluation.",
                           "This evaluates document-source retrieval, not answer faithfulness or ticket classification."]}
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps({"output":str(output),"summary":summary},indent=2))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path,default=ROOT/"support/tests/retrieval_eval_questions.json")
    parser.add_argument("--output",type=Path,default=ROOT/"retrieval_evaluation/results_100.json")
    parser.add_argument("--k",type=int,default=3)
    parser.add_argument("--retrieval-only",action="store_true",help="Compatibility flag; this runner is retrieval-only.")
    args=parser.parse_args(argv)
    if args.k<1:parser.error("k must be positive")
    run(args.dataset,args.output,args.k)


if __name__=="__main__":
    main()

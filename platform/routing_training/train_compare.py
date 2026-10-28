"""Three-model experiment: fit/train, select on validation, then open test once."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from domain_classifier import probabilities

LABELS=["cloud","data","it","other","production","security"]
BASE_MODEL="sentence-transformers/all-MiniLM-L6-v2"
BASE_REVISION="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def acceptance(probs,threshold,margin):
    ordered=np.sort(probs,axis=1)
    return (ordered[:,-1]>=threshold)&((ordered[:,-1]-ordered[:,-2])>=margin)


def select_threshold(probs,labels,classes):
    predicted=np.array(classes)[probs.argmax(axis=1)]
    labels=np.array(labels)
    candidates=[]
    for threshold in np.arange(.25,.96,.05):
        for margin in (0.,.05,.1,.15,.2,.3):
            accepted=acceptance(probs,threshold,margin)
            count=int(accepted.sum())
            accuracy=float(np.mean(predicted[accepted]==labels[accepted])) if count else 0
            if count>=6 and accuracy>=.90:
                candidates.append((count,accuracy,float(threshold),margin))
    if not candidates:
        return 1.,1.
    _,_,threshold,margin=max(candidates,key=lambda row:(row[0],row[1],row[2],row[3]))
    return threshold,margin


def metrics(rows,probs,classes,threshold,margin):
    labels=[row["domain"] for row in rows]
    predicted=np.array(classes)[probs.argmax(axis=1)]
    accepted=acceptance(probs,threshold,margin)
    report=classification_report(labels,predicted,labels=LABELS,output_dict=True,zero_division=0)
    return {
        "n":len(rows),"accuracy":float(accuracy_score(labels,predicted)),
        "macro_f1":float(f1_score(labels,predicted,labels=LABELS,average="macro",zero_division=0)),
        "per_domain_recall":{label:report[label]["recall"] for label in LABELS},
        "per_domain_f1":{label:report[label]["f1-score"] for label in LABELS},
        "confusion_matrix":confusion_matrix(labels,predicted,labels=LABELS).tolist(),
        "confusion_label_order":LABELS,"accepted_coverage":float(accepted.mean()),
        "accepted_accuracy":float(np.mean(predicted[accepted]==np.array(labels)[accepted])) if accepted.any() else None,
        "abstention_rate":float(1-accepted.mean()),
        "cases":[{"id":row["id"],"expected":row["domain"],"predicted":str(pred),
                  "accepted":bool(yes),"confidence":float(prob.max())}
                 for row,pred,yes,prob in zip(rows,predicted,accepted,probs)],
    }


def select_head(train_features,train_labels,val_features,val_rows):
    candidates=[]
    for strength in (.1,1.,10.):
        head=LogisticRegression(C=strength,class_weight="balanced",max_iter=2000,random_state=42)
        head.fit(train_features,train_labels)
        probs=head.predict_proba(val_features)
        threshold,margin=select_threshold(probs,[row["domain"] for row in val_rows],head.classes_)
        score=metrics(val_rows,probs,head.classes_,threshold,margin)
        candidates.append((score["macro_f1"],score["accepted_coverage"],-strength,head,threshold,margin,score))
    _,_,_,head,threshold,margin,score=max(candidates,key=lambda row:row[:3])
    return head,threshold,margin,score


def export_head(head,threshold,margin):
    return {"version":1,"classes":head.classes_.tolist(),"coefficients":head.coef_.tolist(),
            "intercept":head.intercept_.tolist(),"threshold":threshold,"margin_threshold":margin}


def latency(predict,texts):
    for text in texts[:3]:
        predict([text])
    values=[]
    for text in texts[:50]:
        start=time.perf_counter()
        predict([text])
        values.append((time.perf_counter()-start)*1000)
    ordered=sorted(values)
    import math
    return {"samples":len(values),"batch_size":1,"warm_p50_ms":ordered[math.ceil(len(values)*.5)-1],
            "warm_p95_ms":ordered[math.ceil(len(values)*.95)-1],"includes_process_ipc":False}


def weight_hash(model):
    digest=hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run(args):
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer
    from setfit import SetFitModel, Trainer, TrainingArguments
    torch.set_num_threads(4)
    np.random.seed(42)
    torch.manual_seed(42)
    started=time.perf_counter()
    out=ROOT/"routing_models"/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    report_dir=ROOT/"routing_training"/"reports"/args.run_id
    report_dir.mkdir(parents=True,exist_ok=False)
    data_dir=ROOT/"routing_training"/"data"
    dataset_manifest=json.loads((data_dir/"manifest.json").read_text())
    train=read_rows(data_dir/"train.jsonl")
    valid=read_rows(data_dir/"validation.jsonl")
    xt,yt=[row["text"] for row in train],[row["domain"] for row in train]
    xv=[row["text"] for row in valid]
    models={}
    print("Fitting TF-IDF baseline",flush=True)
    vectorizer=TfidfVectorizer(analyzer="char",ngram_range=(3,5),sublinear_tf=True)
    train_features=vectorizer.fit_transform(xt)
    val_features=vectorizer.transform(xv)
    head,threshold,margin,score=select_head(train_features,yt,val_features,valid)
    tfidf=export_head(head,threshold,margin)
    tfidf.update(vocabulary={key:int(value) for key,value in vectorizer.vocabulary_.items()},
                 idf=vectorizer.idf_.tolist())
    (out/"tfidf.json").write_text(json.dumps(tfidf))
    predict_tfidf=lambda texts:head.predict_proba(vectorizer.transform(texts))
    # Capture the fitted head now; later candidates use separate heads.
    tfidf_head=head
    predict_tfidf=lambda texts:tfidf_head.predict_proba(vectorizer.transform(texts))
    for text,expected in zip(xv[:10],predict_tfidf(xv[:10])):
        assert np.max(np.abs(np.array(probabilities(text,tfidf))-expected))<1e-9
    models["tfidf"]={"predict":predict_tfidf,"classes":head.classes_.tolist(),"threshold":threshold,
                     "margin":margin,"validation":score,"kind":"tfidf","head":"tfidf.json"}
    models["tfidf"]["latency"]=latency(predict_tfidf,xv)
    print("Encoding frozen SentenceTransformer features",flush=True)
    encoder=SentenceTransformer(BASE_MODEL,revision=BASE_REVISION,local_files_only=True,device="cpu")
    encoder.max_seq_length=64
    frozen_dir=out/"frozen_encoder"
    encoder.save_pretrained(str(frozen_dir))
    initial_hash=weight_hash(encoder)
    def encode(model,texts):
        return model.encode(texts,normalize_embeddings=True,convert_to_numpy=True,
                            batch_size=32,show_progress_bar=False)
    frozen_train=encode(encoder,xt)
    frozen_valid=encode(encoder,xv)
    frozen_head,threshold,margin,score=select_head(frozen_train,yt,frozen_valid,valid)
    (out/"frozen_head.json").write_text(json.dumps(export_head(frozen_head,threshold,margin)))
    predict_frozen=lambda texts:frozen_head.predict_proba(encode(encoder,texts))
    models["frozen"]={"predict":predict_frozen,"classes":frozen_head.classes_.tolist(),
                      "threshold":threshold,"margin":margin,"validation":score,"kind":"embedding",
                      "head":"frozen_head.json","encoder":"frozen_encoder"}
    models["frozen"]["latency"]=latency(predict_frozen,xv)
    print(f"Fine-tuning MiniLM with SetFit for {args.steps} optimizer steps",flush=True)
    model=SetFitModel.from_pretrained(str(frozen_dir),local_files_only=True,
                                     head_params={"class_weight":"balanced","max_iter":2000})
    model.model_body.max_seq_length=64
    label_ids={label:index for index,label in enumerate(LABELS)}
    train_dataset=Dataset.from_dict({"text":xt,"label":[label_ids[label] for label in yt]})
    training_args=TrainingArguments(output_dir=str(out/"trainer"),batch_size=16,
                                   num_epochs=1,max_steps=args.steps,num_iterations=2,
                                   body_learning_rate=2e-5,max_length=64,seed=42,
                                   report_to="none",save_strategy="no",eval_strategy="no",
                                   logging_steps=10,show_progress_bar=False)
    fine_start=time.perf_counter()
    trainer=Trainer(model=model,args=training_args,train_dataset=train_dataset,
                    metric=lambda predictions,labels:{"accuracy":accuracy_score(labels,predictions)})
    trainer.train()
    fine_seconds=time.perf_counter()-fine_start
    training_device=str(model.model_body.device)
    model.model_body.to("cpu")
    final_hash=weight_hash(model.model_body)
    if final_hash==initial_hash:
        raise RuntimeError("Encoder parameters did not change during fine-tuning")
    fine_dir=out/"setfit_encoder"
    model.model_body.save_pretrained(str(fine_dir))
    fine_train=encode(model.model_body,xt)
    fine_valid=encode(model.model_body,xv)
    fine_head,threshold,margin,score=select_head(fine_train,yt,fine_valid,valid)
    (out/"setfit_head.json").write_text(json.dumps(export_head(fine_head,threshold,margin)))
    predict_fine=lambda texts:fine_head.predict_proba(encode(model.model_body,texts))
    models["setfit"]={"predict":predict_fine,"classes":fine_head.classes_.tolist(),
                      "threshold":threshold,"margin":margin,"validation":score,"kind":"embedding",
                      "head":"setfit_head.json","encoder":"setfit_encoder"}
    models["setfit"]["latency"]=latency(predict_fine,xv)
    winner=max(models,key=lambda name:(models[name]["validation"]["macro_f1"],
                                       models[name]["validation"]["accepted_coverage"],
                                       -models[name]["latency"]["warm_p95_ms"]))
    selected=models[winner]
    selection={"winner":winner,"selection_basis":"validation macro-F1, then accepted coverage, then lower warm p95",
               "dataset_sha256":dataset_manifest["dataset_sha256"],"test_used_for_selection":False,
               "candidates":{name:{key:value for key,value in candidate.items() if key!="predict"}
                             for name,candidate in models.items()}}
    (report_dir/"selection.json").write_text(json.dumps(selection,indent=2))
    print("Validation winner: "+winner+"; selection frozen before opening test",flush=True)
    with (report_dir/"test_opened.lock").open("x") as guard:
        guard.write(hashlib.sha256((report_dir/"selection.json").read_bytes()).hexdigest())
    test=read_rows(data_dir/"test.locked.jsonl")
    xs=[row["text"] for row in test]
    test_results={}
    for name,candidate in models.items():
        probs=candidate["predict"](xs)
        test_results[name]=metrics(test,probs,candidate["classes"],candidate["threshold"],candidate["margin"])
    classifier=report_dir/"selected_head.json"
    classifier.write_text((out/selected["head"]).read_text())
    deployment={"schema_version":1,"name":winner,"kind":selected["kind"],
                "head":str(classifier.relative_to(ROOT)),
                "head_sha256":hashlib.sha256(classifier.read_bytes()).hexdigest(),
                "encoder":str((out/selected["encoder"]).relative_to(ROOT)) if winner=="setfit" else None,
                "encoder_model":BASE_MODEL if winner=="frozen" else None,"max_seq_length":64,
                "encoder_revision":BASE_REVISION if winner=="frozen" else None,
                "python":".venv-training/bin/python" if winner=="setfit" else ".venv-support/bin/python",
                "dataset_sha256":dataset_manifest["dataset_sha256"],"selected_on":"validation",
                "review_status":"synthetic_not_independently_reviewed",
                "report":str((report_dir/"comparison.json").relative_to(ROOT))}
    (report_dir/"deployment.json").write_text(json.dumps(deployment,indent=2))
    report={"created_at_utc":datetime.now(timezone.utc).isoformat(),"dataset":dataset_manifest,
            "selection":selection,"test":test_results,"fine_tuning":{"steps":args.steps,"device":training_device,"inference_device":"cpu",
            "seconds":fine_seconds,"base_weight_sha256":initial_hash,"fine_tuned_weight_sha256":final_hash},
            "total_seconds":time.perf_counter()-started,
            "base_model_revision":BASE_REVISION,
            "packages":{name:importlib.metadata.version(name) for name in
                        ("setfit","sentence-transformers","torch","scikit-learn","datasets","transformers")},
            "limitations":["Synthetic, unreviewed scenarios; not independent real support data.",
                           "Only 18 topic families and 90 base scenarios in the final test.",
                           "Training formatting variants do not add independent scenarios.",
                           "Measured model inference latency excludes queueing, IPC and end-to-end API processing.",
                           "Scores and acceptance thresholds are not production-calibrated."]}
    (report_dir/"comparison.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"report":str(report_dir/"comparison.json"),"winner":winner,
                      "test":{name:{key:value for key,value in score.items() if key not in ("cases","confusion_matrix")}
                              for name,score in test_results.items()}},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--steps",type=int,default=80)
    args=parser.parse_args()
    if not args.run_id.replace("-","").replace("_","").isalnum() or not 1<=args.steps<=10000:
        parser.error("Invalid run id or training steps")
    run(args)

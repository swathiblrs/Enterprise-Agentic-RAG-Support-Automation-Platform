"""Build grouped synthetic data; never label generated examples as human-reviewed."""
import hashlib
import json
from pathlib import Path
import re
from collections import Counter, defaultdict

ROOT = Path(__file__).parent
LABELS = ("cloud","data","it","other","production","security")


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+",text.lower()))


def build(seed_file=ROOT/"scenario_families.json", output=ROOT/"data", seed=41):
    source = json.loads(Path(seed_file).read_text())
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    splits = defaultdict(list)
    for domain in LABELS:
        families = source["families"][domain]
        if len(families) != 24:
            raise ValueError("Expected 24 topic families per domain")
        ranked = sorted(families,key=lambda row:hashlib.sha256(f"{seed}:{domain}:{row[0]}".encode()).hexdigest())
        for position,(family,context,details) in enumerate(ranked):
            split = "train" if position<18 else "validation" if position<21 else "test"
            observations = details.split(";")
            if len(observations)!=5:
                raise ValueError("Each family must contain five observations")
            for index,detail in enumerate(observations):
                base = context + ": " + detail.strip() + "."
                versions = [base]
                if split=="train":
                    versions += ["Please help investigate this request: "+base,
                                 base+" What should be checked next?"]
                for variant,text in enumerate(versions):
                    splits[split].append({
                        "id":f"{domain}-{family}-{index}-{variant}", "text":text, "domain":domain,
                        "group_id":f"{domain}/{family}", "scenario_id":f"{domain}/{family}/{index}",
                        "variant":variant,"split":split,"origin":"project_authored_synthetic",
                        "review_status":"not_independently_reviewed",
                    })
    rows = [row for split in ("train","validation","test") for row in splits[split]]
    audit = audit_rows(rows)
    for split,name in [("train","train.jsonl"),("validation","validation.jsonl"),("test","test.locked.jsonl")]:
        (output/name).write_text("".join(json.dumps(row)+"\n" for row in splits[split]))
    source_hash=hashlib.sha256(Path(seed_file).read_bytes()).hexdigest()
    manifest = {
        "schema_version":1,"seed":seed,"source_sha256":source_hash,"labels":list(LABELS),
        "provenance":source["provenance"],
        "split_method":"Topic-family grouping; all variants and scenarios in a family stay together.",
        "augmentation":"Two formatting variants per training base scenario; no validation/test augmentation.",
        "independent_human_review":False,"production_representativeness_established":False,
        "dataset_sha256":hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        "audit":audit,
    }
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    review=[row for row in rows if row["variant"]==0]
    (output/"review_queue.jsonl").write_text("".join(json.dumps({**row,"reviewer":"","reviewed_label":"","notes":""})+"\n" for row in review))
    return manifest


def audit_rows(rows):
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate example ids")
    if len({normalize(row["text"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate normalized example text")
    groups,scenarios=defaultdict(set),defaultdict(set)
    for row in rows:
        if row["domain"] not in LABELS or row["split"] not in ("train","validation","test"):
            raise ValueError("Invalid labels or splits")
        groups[row["group_id"]].add(row["split"])
        scenarios[row["scenario_id"]].add(row["split"])
    if any(len(values)!=1 for values in [*groups.values(),*scenarios.values()]):
        raise ValueError("Related examples cross split boundaries")
    # Compare unaugmented text only; augmentation is explicitly not independent evidence.
    training=[r for r in rows if r["split"]=="train" and r["variant"]==0]
    heldout=[r for r in rows if r["split"]!="train"]
    def shingles(text):
        text=normalize(text)
        return {text[i:i+5] for i in range(max(0,len(text)-4))}
    features=[shingles(row["text"]) for row in training]
    suspicious=[]
    max_similarity=0
    for row in heldout:
        values=shingles(row["text"])
        scores=[len(values&item)/max(1,len(values|item)) for item in features]
        best=max(scores,default=0)
        max_similarity=max(max_similarity,best)
        if best>=.85:
            suspicious.append({"heldout_id":row["id"],"train_id":training[scores.index(best)]["id"],"jaccard":best})
    if suspicious:
        raise ValueError("Near-duplicate held-out texts: "+json.dumps(suspicious))
    return {
        "examples":len(rows),"base_scenarios":len(scenarios),"topic_families":len(groups),
        "split_counts":dict(Counter(row["split"] for row in rows)),
        "class_counts_by_split":{split:dict(Counter(row["domain"] for row in rows if row["split"]==split))
                                 for split in ("train","validation","test")},
        "group_overlap":0,"scenario_overlap":0,"exact_text_duplicates":0,
        "near_duplicate_threshold":.85,"near_duplicates_at_threshold":0,
        "max_train_heldout_char5_jaccard":max_similarity,
        "note":"Lexical checks do not prove absence of semantic leakage or label ambiguity.",
    }


if __name__=="__main__":
    print(json.dumps(build(),indent=2))

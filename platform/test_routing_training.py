import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import routing_model
from routing_training.build_dataset import audit_rows


def load_rows():
    root=Path("routing_training/data")
    return [json.loads(line) for name in ("train.jsonl","validation.jsonl","test.locked.jsonl")
            for line in (root/name).read_text().splitlines()]


def test_grouped_dataset_counts_and_provenance():
    rows=load_rows()
    audit=audit_rows(rows)
    assert audit["examples"]==1800 and audit["base_scenarios"]==720
    assert audit["split_counts"]=={"train":1620,"validation":90,"test":90}
    assert audit["group_overlap"]==0 and audit["exact_text_duplicates"]==0
    assert all(row["review_status"]=="not_independently_reviewed" for row in rows)
    assert all(row["variant"]==0 for row in rows if row["split"]!="train")


def test_group_leakage_is_rejected():
    rows=load_rows()
    first=next(row for row in rows if row["split"]=="train")
    other=next(row for row in rows if row["split"]=="validation")
    other["group_id"]=first["group_id"]
    with pytest.raises(ValueError,match="Related examples"):
        audit_rows(rows)


def test_duplicate_text_is_rejected():
    rows=load_rows()
    rows[-1]["text"]=rows[0]["text"]
    with pytest.raises(ValueError,match="Duplicate normalized"):
        audit_rows(rows)


def test_missing_selected_model_uses_explicit_legacy_fallback():
    with patch("routing_model.load_manifest",return_value=None),patch(
        "domain_classifier.predict",return_value={"available":True,"domain":"it","accepted":True}):
        result=routing_model.predict("help")
    assert result["selection"]=="legacy_fallback_no_usable_manifest"


def test_selected_embedding_worker_failure_abstains():
    manifest={"kind":"embedding","_head":{"classes":["it","other"]}}
    with patch("routing_model.load_manifest",return_value=manifest),patch.object(
        routing_model.embedding_worker,"call",side_effect=RuntimeError("unavailable")):
        prediction=routing_model.predict("help")
    assert prediction["accepted"] is False and prediction["available"] is False


def test_selected_threshold_controls_acceptance():
    manifest={"kind":"embedding","name":"frozen","dataset_sha256":"test",
              "_head":{"classes":["it","other"],"threshold":.8,"margin_threshold":.2}}
    with patch("routing_model.load_manifest",return_value=manifest),patch.object(
        routing_model.embedding_worker,"call",return_value=[.6,.4]):
        result=routing_model.predict("uncertain request")
    assert result["domain"]=="it" and not result["accepted"]


def test_corrupt_artifact_is_not_loaded(tmp_path,monkeypatch):
    head=tmp_path/"head.json"
    head.write_text("{}")
    manifest=tmp_path/"active.json"
    manifest.write_text(json.dumps({"schema_version":1,"kind":"tfidf","head":str(head),
                                    "head_sha256":"invalid"}))
    monkeypatch.setattr(routing_model,"ROOT",tmp_path)
    monkeypatch.setattr(routing_model,"MANIFEST",manifest)
    routing_model.load_manifest.cache_clear()
    try:
        assert routing_model.load_manifest() is None
    finally:
        routing_model.load_manifest.cache_clear()


def test_report_freezes_selection_before_test():
    root=Path("routing_training/reports/grouped-minilm-v1")
    selection=json.loads((root/"selection.json").read_text())
    assert selection["test_used_for_selection"] is False
    assert (root/"test_opened.lock").read_text()==hashlib.sha256((root/"selection.json").read_bytes()).hexdigest()

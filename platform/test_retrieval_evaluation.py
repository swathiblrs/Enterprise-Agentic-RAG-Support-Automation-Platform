import json
from pathlib import Path

import pytest
from retrieval_benchmark import score_sources,summarize,validate_cases


def cases():
    return json.loads(Path("support/tests/retrieval_eval_questions.json").read_text())


def test_benchmark_has_100_answerable_questions_and_controls():
    rows=cases()
    sources={path.name for path in Path("support/data").glob("*.md")}
    validate_cases(rows,sources)
    assert len(rows)==120
    assert sum(row["answerable"] for row in rows)==100
    assert len({row["scenario_group"] for row in rows if row["answerable"]})==80


def test_fixed_precision_penalizes_missing_slots():
    score=score_sources(["a.md"],["a.md"])
    assert score["precision_at_k"]==pytest.approx(1/3)
    assert score["recall_at_k"]==1
    assert score["precision_over_returned"]==1


def test_duplicate_sources_do_not_inflate_hits():
    score=score_sources(["a.md","a.md","b.md"],["a.md","b.md"])
    assert score["precision_at_k"]==pytest.approx(2/3)
    assert score["recall_at_k"]==1
    assert score["returned_sources"]==["a.md","b.md"]


def test_rank_and_partial_recall_are_measured():
    score=score_sources(["unrelated.md","a.md"],["a.md","b.md"])
    assert score["recall_at_k"]==.5 and score["mrr"]==.5 and score["top_1"]==0
    assert 0<score["ndcg_at_k"]<1


def test_controls_do_not_receive_perfect_recall():
    score=score_sources(["a.md"],[])
    assert score["recall_at_k"] is None and score["precision_at_k"] is None


def test_missing_sources_and_duplicate_questions_are_rejected():
    rows=cases()
    with pytest.raises(ValueError,match="absent"):
        validate_cases(rows,set())
    rows[-1]["question"]=rows[0]["question"]
    with pytest.raises(ValueError,match="Duplicate normalized"):
        validate_cases(rows,{s for row in rows for s in row["expected_sources"]})


def test_summary_separates_controls_and_group_macro():
    rows=[
        {"answerable":True,"scenario_group":"one","stratum":"a","expected_sources":["a"],**score_sources(["a"],["a"])},
        {"answerable":False,"scenario_group":"two","stratum":"control","expected_sources":[],**score_sources(["a"],[])},
    ]
    report=summarize(rows,3)
    assert report["answerable_questions"]==1 and report["unsupported_controls"]==1
    assert report["overall"]["recall_at_k"]==1 and report["unsupported_returned_context"]==1

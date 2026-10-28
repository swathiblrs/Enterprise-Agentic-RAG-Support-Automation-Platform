import json
from pathlib import Path
from fastapi.testclient import TestClient
from api import Request, api, route
from public_faq_cache import PublicFAQCache


def test_university_questions_route_to_support():
    for case in json.loads(Path("university_eval_questions.json").read_text()):
        assert route(Request(question=case["question"]))["path"] == "support", case["id"]


def test_public_cache_is_bounded_and_expires(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"question":"first"},{"question":"second"}]))
    now = [0]
    cache = PublicFAQCache(path, ttl=10, max_entries=1, clock=lambda: now[0])
    calls = []
    def compute():
        calls.append(True)
        return {"answer":"public","nested":{"value":1}}
    first = cache.get("first", compute)
    first["nested"]["value"] = 9
    assert cache.get("first",compute)["nested"]["value"] == 1
    assert len(calls)==1
    now[0]=11
    assert not cache.get("first",compute)["delivery"]["public_faq_cache_hit"]
    cache.get("second",compute)
    assert len(cache.entries)==1
    cache.get("private user question",compute)
    cache.get("private user question",compute)
    assert "private user question" not in cache.entries


def test_auth_still_required():
    with TestClient(api) as client:
        assert client.post("/cases",json={"question":"How can I contact UTK OIT?"}).status_code==401

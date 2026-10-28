import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import api as app
from job_store import IdempotencyConflict, JobStore, QueueFull
from job_worker import process_one


@pytest.fixture
def store(tmp_path):
    now = [1000.]
    queue = JobStore(tmp_path/"jobs.sqlite3",clock=lambda:now[0],lease_seconds=10,retry_seconds=1)
    queue.initialize()
    queue.now = now
    return queue


def payload(text="VPN failure"):
    return {"question":text}


def result(_):
    return {"id":"temporary","question":"VPN failure","answer":"Review the VPN guidance.","ticket_draft":{}}


def test_enqueue_is_durable_and_idempotent(store):
    first,reused = store.enqueue("alice",payload(),"key-one")
    second,reused = store.enqueue("alice",payload(),"key-one")
    assert first["id"]==second["id"] and reused
    reopened = JobStore(store.path)
    reopened.initialize()
    assert reopened.get(first["id"],"alice")["state"]=="queued"
    with pytest.raises(IdempotencyConflict):
        store.enqueue("alice",payload("Different question"),"key-one")
    other,_ = store.enqueue("bob",payload(),"key-one")
    assert other["id"] != first["id"]


def test_capacity_and_payload_limits(store):
    store.max_pending=1
    store.enqueue("alice",payload())
    with pytest.raises(QueueFull):
        store.enqueue("alice",payload())
    with pytest.raises(ValueError):
        store.enqueue("alice",payload("x"*256001))


def test_atomic_completion_case_event_and_notification(store):
    queued,_ = store.enqueue("alice",payload())
    job = store.claim()
    assert store.complete(job,result({}))
    assert not store.complete(job,result({}))
    assert store.result(queued["id"],"alice")["id"]==queued["id"]
    assert store.result(queued["id"],"bob") is None
    assert [e["state"] for e in store.events(job["id"],"alice")] == ["queued","running","succeeded"]
    notes = store.notifications("alice")
    assert len(notes)==1 and notes[0]["state"]=="succeeded"
    assert not store.mark_read(notes[0]["id"],"bob")
    assert store.mark_read(notes[0]["id"],"alice")
    assert store.notifications("alice")[0]["read_at"] is not None


def test_expired_worker_is_recovered_and_fenced(store):
    store.enqueue("alice",payload())
    old = store.claim()
    store.now[0]+=11
    replacement = store.claim()
    assert replacement["id"]==old["id"] and replacement["attempts"]==2
    assert not store.heartbeat(old)
    assert not store.complete(old,result({}))
    assert store.complete(replacement,result({}))
    assert len(store.notifications("alice"))==1


def test_heartbeat_extends_active_lease(store):
    store.enqueue("alice",payload())
    job=store.claim()
    store.now[0]+=8
    assert store.heartbeat(job)
    store.now[0]+=5
    assert store.claim() is None
    assert store.complete(job,result({}))


def test_retries_use_backoff_then_fail_with_notification(store):
    queued,_=store.enqueue("alice",payload())
    for attempt in range(1,4):
        job=store.claim()
        assert job["attempts"]==attempt
        assert store.fail(job,transient=True)
        if attempt<3:
            assert store.claim() is None
            store.now[0]+=2**(attempt-1)
    assert store.get(queued["id"],"alice")["state"]=="failed"
    assert store.result(queued["id"],"alice") is None
    assert store.notifications("alice")[0]["state"]=="failed"


def test_expired_final_attempt_becomes_terminal(store):
    store.max_attempts=1
    queued,_=store.enqueue("alice",payload())
    store.claim()
    store.now[0]+=11
    assert store.claim() is None
    assert store.get(queued["id"],"alice")["state"]=="failed"
    assert len(store.notifications("alice"))==1


def test_competing_workers_claim_distinct_jobs(tmp_path):
    queue=JobStore(tmp_path/"parallel.sqlite3")
    queue.initialize()
    for _ in range(8):
        queue.enqueue("alice",payload())
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs=list(pool.map(lambda _:queue.claim(),range(8)))
    assert len({j["id"] for j in jobs})==8
    assert queue.claim() is None


def test_result_persistence_failure_rolls_back_completion(store):
    queued,_=store.enqueue("alice",payload())
    job=store.claim()
    with store.db() as db:
        db.execute("INSERT INTO cases VALUES (?,?,?)",(queued["id"],"alice","{}"))
    with pytest.raises(sqlite3.IntegrityError):
        store.complete(job,result({}))
    assert store.get(queued["id"],"alice")["state"]=="running"
    assert not store.notifications("alice")
    assert "succeeded" not in [e["state"] for e in store.events(queued["id"],"alice")]


def test_worker_processes_or_retries_and_sanitizes_errors(store):
    queued,_=store.enqueue("alice",payload())
    def unavailable(_):
        raise RuntimeError("private error details")
    assert process_one(store,unavailable,heartbeat=False)
    assert store.get(queued["id"],"alice")["state"]=="retrying"
    assert "private" not in store.get(queued["id"],"alice")["error"]
    store.now[0]+=1
    assert process_one(store,result,heartbeat=False)
    assert store.get(queued["id"],"alice")["state"]=="succeeded"
    assert process_one(store,result,heartbeat=False) is False


def test_nonretryable_worker_error_finishes_as_failed(store):
    queued,_=store.enqueue("alice",payload())
    def invalid(_):
        raise ValueError("invalid private input")
    process_one(store,invalid,heartbeat=False)
    assert store.get(queued["id"],"alice")["state"]=="failed"


@pytest.fixture
def web(tmp_path,monkeypatch):
    monkeypatch.setattr(app,"CASES_DB",tmp_path/"web.sqlite3")
    with TestClient(app.api) as client:
        token=client.post("/auth/login",json={"username":"demo","password":"local-demo-only"}).json()["access_token"]
        client.headers["Authorization"]="Bearer "+token
        yield client


def test_api_accepts_before_execution_and_result_follows_worker(web,monkeypatch):
    def forbidden(_):
        raise AssertionError("Request handler must not execute the workflow")
    monkeypatch.setattr(app,"execute",forbidden)
    response=web.post("/jobs",json=payload(),headers={"Idempotency-Key":"test-submit"})
    assert response.status_code==202 and response.json()["state"]=="queued"
    job_id=response.json()["id"]
    assert response.headers["Location"]=="/jobs/"+job_id
    assert web.get("/jobs/"+job_id+"/result").status_code==409
    assert process_one(app.get_job_queue(),result,heartbeat=False)
    assert web.get("/jobs/"+job_id).json()["state"]=="succeeded"
    assert web.get("/jobs/"+job_id+"/result").json()["id"]==job_id
    assert web.get("/cases").json()[0]["id"]==job_id
    assert len(web.get("/notifications").json())==1
    events=web.get("/jobs/"+job_id+"/events").json()
    assert [e["state"] for e in events]==["queued","running","succeeded"]
    assert web.get("/jobs/"+job_id+"/events",params={"after":events[0]["sequence"]}).json()[0]["state"]=="running"


def test_job_routes_are_authenticated_and_owner_scoped(web):
    job=web.post("/jobs",json=payload()).json()
    process_one(app.get_job_queue(),result,heartbeat=False)
    notification=web.get("/notifications").json()[0]
    web.headers["Authorization"]="Bearer "+app.create_access_token("other",app.UserRole.admin)
    for suffix in ("","/result","/events"):
        assert web.get("/jobs/"+job["id"]+suffix).status_code==404
    assert web.get("/jobs").json()==[]
    assert web.get("/notifications").json()==[]
    assert web.post("/notifications/"+notification["id"]+"/read").status_code==404
    web.headers.pop("Authorization")
    assert web.post("/jobs",json=payload()).status_code==401
    assert web.get("/notifications").status_code==401


def test_api_idempotency_and_input_validation(web):
    first=web.post("/jobs",json=payload(),headers={"Idempotency-Key":"same"})
    again=web.post("/jobs",json=payload(),headers={"Idempotency-Key":"same"})
    assert first.json()["id"]==again.json()["id"] and again.json()["reused"]
    assert web.post("/jobs",json=payload("Other question"),headers={"Idempotency-Key":"same"}).status_code==409
    assert web.post("/jobs",json={"question":"x"}).status_code==422
    queued=app.get_job_queue().claim()
    assert set(queued["payload"])=={"question"}


def test_api_queue_full_returns_backpressure(web):
    queue=app.get_job_queue()
    queue.max_pending=1
    web.post("/jobs",json=payload())
    response=web.post("/jobs",json=payload())
    assert response.status_code==503 and response.headers["Retry-After"]=="2"

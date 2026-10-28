"""Job submission, status, event history, results, and user notifications."""
import os
from functools import lru_cache

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response

from job_store import IdempotencyConflict, JobStore, QueueFull


def build_job_router(database_path, authenticate, request_model):
    router = APIRouter(tags=["Background jobs"])

    @lru_cache(maxsize=8)
    def queue_for_path(path):
        store = JobStore(path, max_pending=int(os.getenv("JOB_MAX_PENDING", "1000")),
                         max_attempts=int(os.getenv("JOB_MAX_ATTEMPTS", "3")),
                         lease_seconds=float(os.getenv("JOB_LEASE_SECONDS", "60")))
        store.initialize()
        return store

    def queue():
        return queue_for_path(str(database_path()))

    def owned(job_id, user, store):
        job = store.get(job_id,user.username)
        if job is None:
            raise HTTPException(404,"Job not found.")
        return job

    @router.post("/jobs", status_code=202)
    def submit(request: request_model, response: Response, user=Depends(authenticate),
               store=Depends(queue),
               idempotency_key: str | None = Header(default=None, min_length=1, max_length=128)):
        try:
            job, reused = store.enqueue(user.username,request.model_dump(mode="json",exclude_unset=True),
                                        idempotency_key)
        except IdempotencyConflict:
            raise HTTPException(409,"Idempotency key was already used for different input.") from None
        except QueueFull:
            raise HTTPException(503,"Job queue is full. Try again later.",headers={"Retry-After":"2"}) from None
        except ValueError as error:
            raise HTTPException(413,str(error)) from None
        response.headers["Location"] = job["status_url"]
        return {**job,"reused":reused}

    @router.get("/jobs")
    def recent(user=Depends(authenticate), store=Depends(queue)):
        return store.recent(user.username)

    @router.get("/jobs/{job_id}")
    def status(job_id: str, user=Depends(authenticate), store=Depends(queue)):
        return owned(job_id,user,store)

    @router.get("/jobs/{job_id}/result")
    def result(job_id: str, user=Depends(authenticate), store=Depends(queue)):
        job = owned(job_id,user,store)
        if job["state"] != "succeeded":
            raise HTTPException(409,"Job result is not ready. Check the job status.")
        return store.result(job_id,user.username)

    @router.get("/jobs/{job_id}/events")
    def events(job_id: str, after: int = Query(default=0,ge=0), user=Depends(authenticate),
               store=Depends(queue)):
        owned(job_id,user,store)
        return store.events(job_id,user.username,after)

    @router.get("/notifications")
    def notifications(user=Depends(authenticate), store=Depends(queue)):
        return store.notifications(user.username)

    @router.post("/notifications/{notification_id}/read")
    def read(notification_id: str, user=Depends(authenticate), store=Depends(queue)):
        if not store.mark_read(notification_id,user.username):
            raise HTTPException(404,"Notification not found.")
        return {"read":True}

    return router, queue

"""Independent consumer for the durable SQLite queue."""
import logging
import os
import signal
import sqlite3
import threading

from job_store import JobStore

logger = logging.getLogger("job_worker")


def process_one(store, handler, heartbeat=True):
    job = store.claim()
    if job is None:
        return False
    done = threading.Event()

    def renew():
        while not done.wait(max(.1,store.lease_seconds/3)):
            try:
                if not store.heartbeat(job):
                    return
            except sqlite3.Error:
                logger.warning("Lease renewal failed for job %s",job["id"])
                return

    thread = threading.Thread(target=renew,daemon=True)
    if heartbeat:
        thread.start()
    try:
        try:
            result = handler(job["payload"])
        except RuntimeError:
            store.fail(job,transient=True)
        except Exception as error:
            logger.error("Job %s failed with %s",job["id"],type(error).__name__)
            store.fail(job,transient=False)
        else:
            if not store.complete(job,result):
                logger.warning("Job %s lost its lease; result was not committed",job["id"])
    finally:
        done.set()
        if heartbeat:
            thread.join(timeout=2)
    return True


def main():
    from api import CASES_DB, Request, execute, worker, routing_worker
    logging.basicConfig(level=logging.INFO)
    store = JobStore(CASES_DB, max_pending=int(os.getenv("JOB_MAX_PENDING","1000")),
                     max_attempts=int(os.getenv("JOB_MAX_ATTEMPTS","3")),
                     lease_seconds=float(os.getenv("JOB_LEASE_SECONDS","60")))
    store.initialize()
    stop = threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stop.set())
    logger.info("Job worker started")
    try:
        while not stop.is_set():
            try:
                handled = process_one(store,lambda payload:execute(Request.model_validate(payload)))
            except sqlite3.Error:
                logger.warning("Queue storage temporarily unavailable")
                stop.wait(1)
                continue
            if not handled:
                stop.wait(.25)
    finally:
        worker.close()
        routing_worker.close()
        logger.info("Job worker stopped")


if __name__ == "__main__":
    main()

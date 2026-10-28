"""Durable single-host job queue with leases, fencing, retries, and notifications."""
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from uuid import uuid4


class QueueFull(Exception):
    pass


class IdempotencyConflict(Exception):
    pass


class JobStore:
    def __init__(self, path, max_pending=1000, max_attempts=3, lease_seconds=60,
                 retry_seconds=1, clock=time.time):
        if min(max_pending, max_attempts, lease_seconds) <= 0 or retry_seconds < 0:
            raise ValueError("Invalid job queue configuration")
        self.path = str(path)
        self.max_pending = max_pending
        self.max_attempts = max_attempts
        self.lease_seconds = lease_seconds
        self.retry_seconds = retry_seconds
        self.clock = clock

    def initialize(self):
        with self.db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, idempotency_key TEXT,
                    fingerprint TEXT NOT NULL, payload TEXT NOT NULL, state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL,
                    available_at REAL NOT NULL, lease_until REAL, lease_token TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL, error TEXT,
                    UNIQUE(owner,idempotency_key));
                CREATE INDEX IF NOT EXISTS jobs_ready ON jobs(state,available_at,created_at);
                CREATE TABLE IF NOT EXISTS job_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    state TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS job_events_job ON job_events(job_id,sequence);
                CREATE TABLE IF NOT EXISTS job_notifications (
                    id TEXT PRIMARY KEY, job_id TEXT UNIQUE NOT NULL, owner TEXT NOT NULL,
                    state TEXT NOT NULL, message TEXT NOT NULL, created_at REAL NOT NULL,
                    read_at REAL);
                CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, owner TEXT, result TEXT);
            """)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def event(self, db, job_id, state, now):
        db.execute("INSERT INTO job_events(job_id,state,created_at) VALUES (?,?,?)",
                   (job_id,state,now))

    def notify(self, db, row, state, now):
        message = "Your case is ready." if state == "succeeded" else "Case processing failed. Review the job status."
        db.execute("INSERT OR IGNORE INTO job_notifications VALUES (?,?,?,?,?,?,NULL)",
                   (str(uuid4()),row["id"],row["owner"],state,message,now))

    def enqueue(self, owner, payload, key=None):
        encoded = json.dumps(payload, sort_keys=True, separators=(",",":"))
        if len(encoded.encode()) > 256_000:
            raise ValueError("Job input exceeds 256 KB")
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        now = self.clock()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if key:
                prior = db.execute("SELECT * FROM jobs WHERE owner=? AND idempotency_key=?",
                                   (owner,key)).fetchone()
                if prior:
                    if prior["fingerprint"] != fingerprint:
                        raise IdempotencyConflict()
                    return self.public(prior), True
            pending = db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running','retrying')").fetchone()[0]
            if pending >= self.max_pending:
                raise QueueFull()
            job_id = str(uuid4())
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,'queued',0,?,?,NULL,NULL,?,?,NULL)",
                       (job_id,owner,key,fingerprint,encoded,self.max_attempts,now,now,now))
            self.event(db,job_id,"queued",now)
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return self.public(row), False

    @staticmethod
    def public(row):
        result = {key:row[key] for key in ("id","state","attempts","max_attempts","created_at","updated_at","error")}
        result["question_preview"] = str(json.loads(row["payload"]).get("question",""))[:100]
        result["status_url"] = "/jobs/" + row["id"]
        result["result_url"] = "/jobs/" + row["id"] + "/result" if row["state"] == "succeeded" else None
        result["case_id"] = row["id"] if row["state"] == "succeeded" else None
        return result

    def get(self, job_id, owner):
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=? AND owner=?", (job_id,owner)).fetchone()
            return self.public(row) if row else None

    def recent(self, owner):
        with self.db() as db:
            return [self.public(row) for row in db.execute(
                "SELECT * FROM jobs WHERE owner=? ORDER BY created_at DESC LIMIT 20", (owner,))]

    def claim(self):
        now = self.clock()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            expired = db.execute("SELECT * FROM jobs WHERE state='running' AND lease_until<=?", (now,)).fetchall()
            for row in expired:
                state = "failed" if row["attempts"] >= row["max_attempts"] else "retrying"
                db.execute("UPDATE jobs SET state=?,available_at=?,lease_token=NULL,lease_until=NULL,"
                           "updated_at=?,error=? WHERE id=?",
                           (state,now,now,"Worker lease expired.",row["id"]))
                self.event(db,row["id"],state,now)
                if state == "failed":
                    self.notify(db,row,state,now)
            row = db.execute("SELECT * FROM jobs WHERE state IN ('queued','retrying') AND available_at<=? "
                             "ORDER BY available_at,created_at LIMIT 1", (now,)).fetchone()
            if row is None:
                return None
            token = str(uuid4())
            db.execute("UPDATE jobs SET state='running',attempts=attempts+1,lease_token=?,lease_until=?,"
                       "updated_at=?,error=NULL WHERE id=?", (token,now+self.lease_seconds,now,row["id"]))
            self.event(db,row["id"],"running",now)
            claimed = dict(db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())
            claimed["payload"] = json.loads(claimed["payload"])
            return claimed

    def heartbeat(self, job):
        now = self.clock()
        with self.db() as db:
            return bool(db.execute("UPDATE jobs SET lease_until=?,updated_at=? "
                                   "WHERE id=? AND state='running' AND lease_token=? AND lease_until>?",
                                   (now+self.lease_seconds,now,job["id"],job["lease_token"],now)).rowcount)

    def complete(self, job, result):
        now = self.clock()
        result = dict(result, id=job["id"], job_id=job["id"])
        encoded = json.dumps(result)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            updated = db.execute("UPDATE jobs SET state='succeeded',lease_token=NULL,lease_until=NULL,"
                                 "updated_at=?,error=NULL WHERE id=? AND state='running' "
                                 "AND lease_token=? AND lease_until>?",
                                 (now,job["id"],job["lease_token"],now)).rowcount
            if not updated:
                return False
            db.execute("INSERT INTO cases VALUES (?,?,?)",(job["id"],job["owner"],encoded))
            self.event(db,job["id"],"succeeded",now)
            self.notify(db,job,"succeeded",now)
            return True

    def fail(self, job, transient):
        now = self.clock()
        retry = transient and job["attempts"] < job["max_attempts"]
        state = "retrying" if retry else "failed"
        message = "Workflow unavailable; waiting to retry." if retry else "Workflow failed. Contact support with this job id."
        available = now + min(30, self.retry_seconds * 2 ** (job["attempts"]-1))
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            updated = db.execute("UPDATE jobs SET state=?,available_at=?,error=?,lease_token=NULL,"
                                 "lease_until=NULL,updated_at=? WHERE id=? AND state='running' "
                                 "AND lease_token=? AND lease_until>?",
                                 (state,available,message,now,job["id"],job["lease_token"],now)).rowcount
            if not updated:
                return False
            self.event(db,job["id"],state,now)
            if not retry:
                self.notify(db,job,state,now)
            return True

    def result(self, job_id, owner):
        with self.db() as db:
            row = db.execute("SELECT result FROM cases WHERE id=? AND owner=?", (job_id,owner)).fetchone()
            return json.loads(row[0]) if row else None

    def events(self, job_id, owner, after=0):
        with self.db() as db:
            return [dict(row) for row in db.execute(
                "SELECT e.sequence,e.state,e.created_at FROM job_events e JOIN jobs j ON j.id=e.job_id "
                "WHERE j.id=? AND j.owner=? AND e.sequence>? ORDER BY e.sequence LIMIT 100",
                (job_id,owner,after))]

    def notifications(self, owner):
        with self.db() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM job_notifications WHERE owner=? ORDER BY created_at DESC LIMIT 50", (owner,))]

    def mark_read(self, notification_id, owner):
        with self.db() as db:
            return bool(db.execute("UPDATE job_notifications SET read_at=COALESCE(read_at,?) WHERE id=? AND owner=?",
                                   (self.clock(),notification_id,owner)).rowcount)

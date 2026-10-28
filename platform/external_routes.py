"""Authenticated, explicit integration actions against stored, user-owned cases."""
import json
import secrets
import sqlite3
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from external_clients import ExternalAPIs, IntegrationError


@lru_cache(maxsize=1)
def get_clients():
    return ExternalAPIs()


class ConfirmAction(BaseModel):
    confirmed: Literal[True]


class AnswerAction(ConfirmAction):
    drive_file_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,160}$")


def build_router(database_path, authenticate):
    router = APIRouter(tags=["External integrations"])

    def operator(user=Depends(authenticate), clients=Depends(get_clients),
                 x_integration_key: str = Header(default="")):
        if getattr(user.role, "value", user.role) != "admin":
            raise HTTPException(403, "Integration actions require an administrator.")
        configured = clients.config.get("INTEGRATION_ADMIN_KEY", "")
        if len(configured) < 24:
            raise HTTPException(503, "A separate integration administration key must be configured.")
        if not secrets.compare_digest(configured, x_integration_key):
            raise HTTPException(403, "Invalid integration administration key.")
        return user

    def case_for_owner(case_id, owner):
        with sqlite3.connect(database_path(), timeout=30) as db:
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cases'").fetchone()
            row = db.execute("SELECT result FROM cases WHERE id=? AND owner=?", (case_id, owner)).fetchone() if exists else None
        if row is None:
            raise HTTPException(404, "Case not found.")
        return json.loads(row[0])

    def invoke(call):
        try:
            return call()
        except IntegrationError as error:
            raise HTTPException(error.status, str(error)) from None

    def once(case_id, owner, operation, call):
        # Persist the reservation before any outbound POST to prevent duplicate actions.
        with sqlite3.connect(database_path(), timeout=30) as db:
            db.execute("CREATE TABLE IF NOT EXISTS external_actions "
                       "(case_id TEXT, owner TEXT, operation TEXT, status TEXT, result TEXT, "
                       "PRIMARY KEY(case_id,owner,operation))")
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT status,result FROM external_actions WHERE case_id=? AND owner=? AND operation=?",
                               (case_id,owner,operation)).fetchone()
            if prior:
                if prior[0] == "succeeded":
                    return {"reused": True, **json.loads(prior[1])}
                if prior[0] != "failed":
                    raise HTTPException(409, "Action pending or outcome unknown; reconcile with the provider before retrying.")
                db.execute("DELETE FROM external_actions WHERE case_id=? AND owner=? AND operation=?",
                           (case_id,owner,operation))
            db.execute("INSERT INTO external_actions VALUES (?,?,?,?,?)", (case_id,owner,operation,"pending","{}"))
        try:
            result = call()
        except IntegrationError as error:
            with sqlite3.connect(database_path(), timeout=30) as db:
                db.execute("UPDATE external_actions SET status=? WHERE case_id=? AND owner=? AND operation=?",
                           ("unknown" if error.outcome_unknown else "failed",case_id,owner,operation))
            raise HTTPException(error.status, str(error)) from None
        # If this commit fails, the pending record prevents accidental resubmission.
        with sqlite3.connect(database_path(), timeout=30) as db:
            db.execute("UPDATE external_actions SET status='succeeded',result=? "
                       "WHERE case_id=? AND owner=? AND operation=?",
                       (json.dumps(result),case_id,owner,operation))
        return {"reused": False, **result}

    @router.get("/integrations")
    def catalog(user=Depends(authenticate), clients=Depends(get_clients)):
        return {"providers": clients.catalog(), "verification": "Configuration only; no live verification performed."}

    @router.get("/integrations/microsoft-graph/me")
    def directory(user=Depends(operator), clients=Depends(get_clients)):
        return invoke(clients.graph_me)

    @router.get("/integrations/google-drive/files/{file_id}")
    def knowledge(file_id: str, user=Depends(operator), clients=Depends(get_clients)):
        return invoke(lambda: clients.drive_document(file_id))

    @router.post("/cases/{case_id}/jira-ticket")
    def ticket(case_id: str, action: ConfirmAction, user=Depends(operator), clients=Depends(get_clients)):
        case = case_for_owner(case_id, user.username)
        invoke(lambda: clients.require("jira"))
        return once(case_id,user.username,"jira_ticket",lambda: clients.create_jira_ticket(case))

    @router.post("/cases/{case_id}/slack-notification")
    def notification(case_id: str, action: ConfirmAction, user=Depends(operator), clients=Depends(get_clients)):
        case = case_for_owner(case_id,user.username)
        invoke(lambda: clients.require("slack"))
        return once(case_id,user.username,"slack_notification",lambda: clients.notify_slack(case))

    @router.post("/cases/{case_id}/external-answer")
    def answer(case_id: str, action: AnswerAction, user=Depends(operator), clients=Depends(get_clients)):
        case = case_for_owner(case_id,user.username)
        invoke(lambda: clients.require("openai"))
        sources = list((case.get("support") or {}).get("retrieved_chunks", []))
        if action.drive_file_id:
            document = invoke(lambda: clients.drive_document(action.drive_file_id))
            sources.insert(0, document)
        if not sources:
            raise HTTPException(409, "This case has no retrieved support context; supply an authorized Drive document.")
        operation = "openai_answer:" + (action.drive_file_id or "local_sources")
        return once(case_id,user.username,operation,
                    lambda: clients.grounded_answer(case["question"],sources))

    return router

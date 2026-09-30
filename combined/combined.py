"""Local integration prototype: support, investigation, and reviewed ticket drafts."""
import os
import secrets
import sqlite3
import sys
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'incident'))
os.environ.update(LLM_PROVIDER='offline', OPENAI_API_KEY='', ANTHROPIC_API_KEY='', USE_POSTGRES='false',
                  LANGFUSE_ENABLED='false', A2A_PROVIDER='local', AUTH_REQUIRED='true', JWT_SECRET_KEY=secrets.token_hex(32))

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from app.core.security import CurrentUser, create_access_token, get_current_user
from app.models.schemas import IncidentInput, SecurityAlert, UserRole
from app.services.generic_incident_graph import GenericIncidentGraph
from app.services.investigation_graph import InvestigationGraph
from routing import route
from evidence_provenance import annotate
from support_client import worker

@asynccontextmanager
async def lifespan(app):
    yield
    worker.close()

api = FastAPI(title='Agentic RAG Incident Investigation & Support Automation - Local Prototype', lifespan=lifespan)
TEAMS = {'security':'SOC', 'production':'SRE', 'cloud':'Cloud Infrastructure', 'data':'Data Engineering', 'it':'IT Operations'}

class Request(BaseModel):
    question: str = Field(min_length=3, max_length=12000)
    domain: Literal['auto','security','production','cloud','data','it'] = 'auto'
    severity: Literal['low','medium','high','critical'] = 'medium'
    logs: list[str] = Field(default_factory=list, max_length=200)
    events: list[dict] = Field(default_factory=list, max_length=200)
    raw_events: list[dict] = Field(default_factory=list, max_length=200)
    metrics: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list, max_length=100)
    id: str | None = None
    title: str | None = None
    description: str | None = None
    source: str = 'manual'
    service: str | None = None
    environment: str | None = None
    owner_team: str | None = None
    user: str | None = None
    host: str | None = None
    ip_address: str | None = None
    geo: str | None = None
    tactic: str | None = None
    technique: str | None = None
    detected_at: datetime | None = None

class Login(BaseModel):
    username: str
    password: str

def support(question):
    return worker.call(question)

def investigate(request, domain):
    common = {'title': request.title or request.question[:150],
              'description': request.description if request.description is not None else request.question,
              'severity': request.severity, 'source': request.source, 'tags': request.tags}
    if request.id:
        common['id'] = request.id
    if request.detected_at:
        common['detected_at'] = request.detected_at
    if domain == 'security':
        payload = SecurityAlert(**common, user=request.user, host=request.host, ip_address=request.ip_address,
                                geo=request.geo, tactic=request.tactic, technique=request.technique,
                                raw_events=[*request.raw_events, *request.events,
                                            *[{'event':'submitted_log','message':line} for line in request.logs]])
        return annotate(InvestigationGraph().investigate(payload).model_dump(mode='json'))
    payload = IncidentInput(**common, domain=domain, service=request.service, environment=request.environment,
                            owner_team=request.owner_team, logs=request.logs, metrics=request.metrics,
                            events=[*request.events, *request.raw_events])
    return annotate(GenericIncidentGraph().investigate(payload).model_dump(mode='json'))

def execute(request):
    start = perf_counter()
    routing = route(request)
    result = {'id':str(uuid4()), 'question':request.question, 'routing':routing, 'mode':'local_offline',
              'support':None, 'investigation':None, 'ticket_draft':None, 'execution_trace':[],
              'limitations':['Domain routing combines policy with a small synthetic-data ML fallback; scores are uncalibrated.',
                             'Offline generation; tool evidence may be simulated. No external actions are executed.',
                             'A human must review findings and ticket submission.']}
    if routing['path'] == 'clarify':
        result['answer'] = 'Please identify the affected service or domain and describe the observed failure, impact, and available evidence.'
    else:
        domain = routing['domain']
        if domain == 'it' or routing['support_context']:
            result['support'] = support(request.question)
            result['ticket_draft'] = dict(result['support']['ticket_draft'])
            result['answer'] = result['support']['answer']
            result['execution_trace'].append('LlamaIndex support workflow / NLP classifier / support RAG')
        if routing['path'] == 'investigate':
            report = investigate(request, domain)
            result['investigation'] = report
            result['answer'] = report['executive_summary']
            result['execution_trace'].append(f'LangGraph {domain} investigation / local tools / runbook retrieval')
            draft = dict(result['ticket_draft'] or {})
            support_team = draft.get('assigned_team')
            investigation_sources = list(dict.fromkeys(r['source'] for r in report['references']))
            # Support citations and ownership remain distinct from investigation evidence.
            draft.update(title=request.title or request.question[:150], investigation_summary=report['executive_summary'],
                         investigation_id=report['investigation_id'], evidence=report['evidence'],
                         recommended_actions=report['recommended_actions'],
                         investigation_source_references=investigation_sources,
                         next_action='human_review_before_submission')
            draft.setdefault('description',report['executive_summary'])
            draft.setdefault('source_references',investigation_sources)
            draft.setdefault('category',domain)
            draft['support_team'] = support_team
            draft['assigned_team'] = support_team if domain == 'it' and support_team else TEAMS[domain]
            draft['assignment_reason'] = 'Preserved specialist IT team.' if domain == 'it' and support_team else f'{domain} investigation owner; prior support team retained separately.'
            priorities = ['low','medium','high','critical']
            original = priorities.index(str(draft.get('priority','low')).lower())
            explicit = priorities.index(request.severity) if 'severity' in request.model_fields_set else 0
            draft['priority'] = priorities[max(original, explicit, 2 if report['risk_score'] >= 75 else 0)]
            result['ticket_draft'] = draft
    result['latency_ms'] = round((perf_counter()-start)*1000,2)
    return result

@api.get('/', include_in_schema=False)
def home():
    return FileResponse(ROOT/'index.html')

@api.get('/architecture.svg', include_in_schema=False)
def architecture():
    return FileResponse(ROOT/'architecture.svg', media_type='image/svg+xml')

@api.post('/auth/login')
def login(credentials: Login):
    if not (secrets.compare_digest(credentials.username,'demo') and secrets.compare_digest(credentials.password,'local-demo-only')):
        raise HTTPException(401,'Invalid credentials')
    return {'access_token':create_access_token('demo',UserRole.admin),'token_type':'bearer'}

@api.post('/cases')
def create_case(request: Request, user: CurrentUser = Depends(get_current_user)):
    try:
        result = execute(request)
    except RuntimeError as error:
        raise HTTPException(503,'Workflow unavailable; no ticket was submitted.') from error
    with sqlite3.connect(ROOT/'cases.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, owner TEXT, result TEXT)')
        db.execute('INSERT INTO cases VALUES (?, ?, ?)',(result['id'],user.username,json.dumps(result)))
    return result

@api.get('/cases')
def history(user: CurrentUser = Depends(get_current_user)):
    with sqlite3.connect(ROOT/'cases.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, owner TEXT, result TEXT)')
        return [json.loads(row[0]) for row in db.execute('SELECT result FROM cases WHERE owner = ? ORDER BY rowid DESC LIMIT 20',(user.username,))]


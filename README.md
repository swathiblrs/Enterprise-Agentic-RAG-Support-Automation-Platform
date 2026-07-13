# Enterprise Agentic RAG Support Automation Platform

A production-style Retrieval-Augmented Generation platform for enterprise IT support automation.

The system retrieves relevant internal knowledge-base content, generates grounded support answers, classifies IT issues, predicts ticket priority, recommends the right support team, creates ticket drafts, tracks confidence, collects feedback, and exposes metrics for observability.

## Project Overview

This project simulates an enterprise IT service desk assistant powered by RAG and agentic workflow orchestration. Instead of answering only from general model knowledge, the assistant searches a local enterprise knowledge base, retrieves relevant document chunks, and uses those chunks to produce source-backed support responses.

Beyond basic RAG, the platform adds support automation:

- issue classification
- priority prediction
- team routing
- confidence scoring
- escalation decisions
- ticket draft generation
- feedback logging
- metrics and evaluation
- Docker and CI/CD readiness
- Kubernetes deployment readiness

## Key Features

- Markdown, text, and PDF document ingestion
- Text chunking with metadata
- Embedding generation using SentenceTransformers
- ChromaDB vector storage
- BM25 keyword retrieval
- Hybrid retrieval with reranking
- Domain-aware retrieval for configurable knowledge domains
- Optional LLM-based grounded answer generation
- Deterministic fallback answer generation for local/offline use
- Source-backed responses
- Ticket classification
- Priority prediction
- Support team routing
- Agent-style workflow orchestration
- Confidence scoring and escalation decisions
- Structured ticket draft generation
- User feedback capture
- Metrics endpoint for latency, fallback rate, confidence, categories, and agent decisions
- Configurable API key and JWT role-based authentication
- Streamlit UI with Ask, Analytics, and Upload tabs
- FastAPI backend with Swagger docs
- JSONL query and feedback logging
- Evaluation pipeline with retrieval, routing, confidence, and faithfulness metrics
- Docker, Docker Compose, Kubernetes manifests, Makefile, and GitHub Actions CI

## System Architecture
<img width="1581" height="995" alt="image" src="https://github.com/user-attachments/assets/045adcad-2f54-4cb0-8a32-27e7c77d6845" />

## Current Knowledge Base

The project uses synthetic enterprise IT support documents:

```text
data/
  password_reset_kb.md
  vpn_troubleshooting_kb.md
  mfa_duo_kb.md
  ticket_routing_rules.md
  priority_matrix.md
```

These documents cover:

- password reset
- account lockout
- VPN troubleshooting
- Duo/MFA issues
- ticket routing rules
- ticket priority rules

No private company, university, or personal user data is used.

## Tech Stack

- Python
- FastAPI
- Streamlit
- ChromaDB
- SentenceTransformers
- BM25
- LangChain / LangChain OpenAI
- Pydantic
- pypdf
- Docker
- GitHub Actions
- JSONL logging

## Project Structure

```text
Enterprise RAG Support Automation Platform/
  app/
    streamlit_app.py
  data/
    password_reset_kb.md
    vpn_troubleshooting_kb.md
    mfa_duo_kb.md
    ticket_routing_rules.md
    priority_matrix.md
  src/
    api.py
    config.py
    document_store.py
    evaluator.py
    generator.py
    ingest.py
    logger.py
    orchestrator.py
    retriever.py
    security.py
    ticket_classifier.py
  tests/
    eval_questions.json
    test_api.py
  .github/
    workflows/
      ci.yml
  k8s/
    api-deployment.yaml
    api-service.yaml
    configmap.yaml
    ingress.yaml
    secret.example.yaml
    namespace.yaml
    persistent-volume-claims.yaml
    streamlit-deployment.yaml
    streamlit-service.yaml
  Dockerfile
  docker-compose.yml
  Makefile
  requirements.txt
  .env.example
  README.md
```

`logs/`, `vectorstore/`, and `.venv/` are excluded from GitHub.

## How It Works

### 1. Document Ingestion

The ingestion pipeline reads Markdown, text, and PDF files from the `data/` folder.

```text
Documents -> text extraction -> chunks -> embeddings -> ChromaDB
```

Run:

```bash
python src/ingest.py
```

### 2. Hybrid Retrieval

The retriever narrows the knowledge base using:

- vector similarity search through ChromaDB
- BM25 keyword search
- domain metadata filtering
- result merging
- lightweight reranking

Example:

```text
Question:
My VPN is not working after I reset my password

Likely retrieved sources:
- vpn_troubleshooting_kb.md
- password_reset_kb.md
```

### 3. Grounded Answer Generation

The generator uses retrieved chunks to produce an answer.

By default, the project uses deterministic rule-based generation so it can run locally without external API access.

To enable LLM-based grounded generation:

```env
USE_LLM_GENERATION=true
OPENAI_API_KEY=your_api_key
LLM_MODEL_NAME=gpt-4o-mini
```

The LLM prompt instructs the model to answer only from retrieved context and cite source files.

### 4. Agent Workflow Orchestration

The API calls:

```python
run_support_workflow(question, domain="it_support")
```

The orchestrator coordinates:

- retrieval
- answer generation
- ticket classification
- confidence scoring
- confusion detection
- next-action decisions
- escalation recommendations
- ticket draft generation

Example agent decisions:

- `answer_and_create_ticket_draft`
- `ask_clarifying_question`
- `create_urgent_ticket_draft`
- `escalate_to_human`

### 5. Ticket Intelligence

The classifier predicts:

- ticket summary
- category
- priority
- assigned support team

Example:

```json
{
  "summary": "My VPN is not working after I reset my password",
  "category": "VPN Connectivity",
  "priority": "Medium",
  "assigned_team": "Network Support"
}
```

## API Endpoints

Start the backend:

```bash
uvicorn src.api:app --reload
```

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

```text
GET  /
GET  /health
POST /auth/login
POST /ask
GET  /logs
POST /feedback
GET  /feedback
GET  /metrics
POST /documents/upload
```

Example request:

```json
{
  "question": "My VPN is not working after I reset my password",
  "domain": "it_support"
}
```

Example response:

```json
{
  "request_id": "6ed8718a-597a-4926-9f84-d93a7fe1507b",
  "question": "My VPN is not working after I reset my password",
  "domain": "it_support",
  "answer": "Based on the knowledge base...",
  "sources": [
    "vpn_troubleshooting_kb.md",
    "password_reset_kb.md"
  ],
  "ticket": {
    "summary": "My VPN is not working after I reset my password",
    "category": "VPN Connectivity",
    "priority": "Medium",
    "assigned_team": "Network Support"
  },
  "answer_generation_mode": "rule_based",
  "fallback_triggered": false,
  "confidence": {
    "retrieval_confidence": 0.87,
    "classification_confidence": 0.75,
    "overall_confidence": 0.82
  },
  "agent_decision": {
    "next_action": "answer_and_create_ticket_draft",
    "reason": "Sufficient confidence to answer and prepare a support ticket draft.",
    "assigned_team": "Network Support"
  },
  "ticket_draft": {
    "title": "My VPN is not working after I reset my password",
    "category": "VPN Connectivity",
    "priority": "Medium",
    "assigned_team": "Network Support"
  },
  "latency_ms": 1749.13
}
```

## Security

API key authentication is optional and controlled through environment configuration.

```env
SUPPORT_API_KEY=your_api_key
```

When `SUPPORT_API_KEY` is empty, local development endpoints remain open. When it is set, protected endpoints require:

```text
x-api-key: your_api_key
```

Protected endpoints include:

- `POST /ask`
- `GET /logs`
- `POST /feedback`
- `GET /feedback`
- `GET /metrics`
- `POST /documents/upload`

JWT authentication is also supported when `JWT_SECRET_KEY` is configured:

```env
JWT_SECRET_KEY=replace-with-secret
JWT_EXPIRATION_MINUTES=60
AUTH_DEMO_ADMIN_USERNAME=admin
AUTH_DEMO_ADMIN_PASSWORD=replace-admin-password
AUTH_DEMO_AGENT_USERNAME=agent
AUTH_DEMO_AGENT_PASSWORD=replace-agent-password
```

Login endpoint:

```text
POST /auth/login
```

Roles:

- `admin`: can upload documents, view logs, view feedback, and view metrics
- `support_agent`: can ask questions and submit feedback
- `viewer`: reserved for future read-only workflows

JWT requests use:

```text
Authorization: Bearer <access_token>
```

## Streamlit UI

Start the backend first:

```bash
uvicorn src.api:app --reload
```

Then start the frontend:

```bash
streamlit run app/streamlit_app.py
```

The UI includes three tabs:

- `Ask`: ask support questions and view answer, sources, ticket recommendation, confidence, and ticket draft
- `Analytics`: view latency, fallback rate, helpful feedback rate, category counts, and agent decision counts
- `Upload Documents`: upload Markdown, text, or PDF documents into a selected knowledge domain

## Document Upload

Supported upload types:

- `.md`
- `.txt`
- `.pdf`

Uploaded documents are saved under:

```text
data/<domain>/uploads/
```

The upload endpoint can optionally reindex the vector store after saving a document.

## Logging and Metrics

Query logs:

```text
logs/query_logs.jsonl
```

Feedback logs:

```text
logs/feedback_logs.jsonl
```

Metrics include:

- total queries
- total feedback submissions
- fallback rate
- average latency
- average confidence
- ticket category counts
- agent decision counts
- helpful feedback rate

## Evaluation

Run:

```bash
python -m src.evaluator
```

The evaluator measures:

- retrieval accuracy
- category accuracy
- team routing accuracy
- priority accuracy
- Precision@K
- Recall@K
- grounded answer rate
- faithfulness heuristic rate
- safe agent decision rate
- average confidence
- average latency

Current local evaluation result:

```text
Retrieval Accuracy: 100.00%
Category Accuracy: 100.00%
Team Routing Accuracy: 100.00%
Priority Accuracy: 100.00%
Average Recall@K: 100.00%
Grounded Answer Rate: 100.00%
Faithfulness Heuristic Rate: 100.00%
Safe Agent Decision Rate: 100.00%
```

## Setup

Clone the repository:

```bash
git clone https://github.com/swathiblrs/Enterprise-Rag-Support-Platform.git
cd Enterprise-Rag-Support-Platform
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ingest documents:

```bash
python src/ingest.py
```

Run API:

```bash
uvicorn src.api:app --reload
```

Run UI:

```bash
streamlit run app/streamlit_app.py
```

## Makefile Commands

```bash
make install
make test
make evaluate
make run-api
make run-ui
make docker-build
make docker-up
make docker-down
```

## Docker

Run the full stack:

```bash
docker compose up --build
```

FastAPI:

```text
http://127.0.0.1:8000
```

Streamlit:

```text
http://127.0.0.1:8501
```

## Kubernetes

Kubernetes manifests are provided under:

```text
k8s/
```

They include:

- namespace
- config map
- secret template
- API deployment and service
- Streamlit deployment and service
- persistent volume claims for logs and vectorstore
- ingress template

See [k8s/README.md](k8s/README.md) for deployment steps.

## CI/CD

GitHub Actions runs on push and pull request.

The CI workflow validates:

- dependency installation
- API tests
- workflow evaluation

## Sample Questions

```text
My VPN is not working after I reset my password.
I cannot approve Duo push notifications.
My account is locked.
Company-wide authentication failure.
Multiple users cannot access VPN.
Production VPN outage for many users.
Duo MFA app stopped sending push requests.
```

## Current Status

Completed:

- synthetic enterprise knowledge base
- Markdown, text, and PDF ingestion
- ChromaDB vector store
- hybrid retrieval with BM25
- lightweight reranking
- optional LLM grounded generation
- source-backed answers
- ticket classification
- priority prediction
- team routing
- agent workflow orchestration
- confidence scoring
- escalation decisions
- ticket draft generation
- API key authentication
- JWT role-based authentication
- domain-aware retrieval
- feedback logging
- metrics endpoint
- analytics UI
- document upload UI
- evaluation pipeline
- API tests
- Docker and Docker Compose
- Kubernetes manifests
- Makefile
- GitHub Actions CI

Remaining future improvements:

- external ITSM integration such as ServiceNow or Jira Service Management
- stronger LLM answer-quality evaluation
- larger enterprise knowledge base
- production monitoring stack such as Prometheus and Grafana

## Project Positioning

This project started as a RAG support assistant and has evolved into an agentic enterprise support automation platform. It demonstrates not only retrieval and answer generation, but also workflow orchestration, ticket intelligence, observability, feedback loops, evaluation, security configuration, document upload, and deployment readiness.

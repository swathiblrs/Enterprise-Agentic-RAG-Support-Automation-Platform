# ***Enterprise RAG Support Platform***

A production-style Retrieval-Augmented Generation platform for enterprise IT support automation. The system retrieves relevant knowledge-base content, generates source-backed support answers, classifies user issues, predicts ticket priority, recommends the correct support team, and logs query activity for observability.

## Project Overview

This project simulates an enterprise IT service desk assistant powered by Retrieval-Augmented Generation. Instead of answering only from general model knowledge, the assistant searches a local knowledge base, retrieves relevant document chunks, and generates grounded support responses with source references.

The platform is designed to demonstrate backend engineering, semantic search, vector databases, API design, ticket automation, routing logic, evaluation, and observability.

## Key Features

* Synthetic enterprise IT knowledge-base documents
* Document ingestion from Markdown, text, and PDF files
* Text chunking with metadata
* Embedding generation using SentenceTransformers
* Vector storage using ChromaDB
* Semantic retrieval over knowledge-base chunks
* RAG-style answer generation with source references
* Ticket classification
* Priority prediction
* Support team routing
* Agent-style support workflow orchestration
* Confidence scoring and escalation decisions
* Structured ticket draft generation
* User feedback capture for answer, source, routing, and priority quality
* Metrics endpoint for latency, fallback rate, confidence, and agent decisions
* Optional LLM-based grounded answer generation
* Configurable API key authentication
* Document upload workflow
* FastAPI backend
* Streamlit web interface
* Query logging using JSONL logs
* Swagger API documentation
* Evaluation pipeline for retrieval and routing quality

## System Architecture

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/b2c2557f-4724-4253-9962-df8f552a4d25" />

## Current Knowledge Base

The project currently uses synthetic enterprise IT support documents:

data/
  password_reset_kb.md
  vpn_troubleshooting_kb.md
  mfa_duo_kb.md
  ticket_routing_rules.md
  priority_matrix.md

These documents cover:

* Password reset
* Account lockout
* VPN troubleshooting
* Duo MFA issues
* Ticket routing rules
* Ticket priority rules

No private company, university, or personal user data is used.

## Tech Stack

* Python
* FastAPI
* Streamlit
* ChromaDB
* SentenceTransformers
* LangChain
* Pydantic
* Requests
* JSONL logging
* Docker
* GitHub Actions CI

## Project Structure

Enterprise RAG Support Automation Platform/
  app/
    streamlit_app.py
  data/
    password_reset_kb.md
    vpn_troubleshooting_kb.md
    mfa_duo_kb.md
    ticket_routing_rules.md
    priority_matrix.md
  logs/
    query_logs.jsonl
  src/
    __init__.py
    api.py
    config.py
    evaluator.py
    generator.py
    ingest.py
    logger.py
    orchestrator.py
    retriever.py
    ticket_classifier.py
  tests/
    eval_questions.json
  .github/
    workflows/
      ci.yml
  Dockerfile
  docker-compose.yml
  Makefile
  vectorstore/
    chroma/
  README.md
  requirements.txt
  .env.example
  .gitignore

Note: logs/, vectorstore/, and .venv/ are excluded from GitHub using .gitignore.

How It Works

1. Document Ingestion

The ingestion script reads Markdown, text, and PDF files from the data/ folder.

Markdown, text, and PDF files → text extraction → chunks → metadata

Run:

python src/ingest.py

This creates embeddings and stores them in ChromaDB.

2. Semantic Retrieval

The retriever converts a user question into an embedding and searches ChromaDB for the most relevant document chunks.

Example:

Question:
My VPN is not working after I reset my password
Retrieved sources:
- vpn_troubleshooting_kb.md
- password_reset_kb.md

Run:

python src/retriever.py

3. Answer Generation

The generator uses retrieved context and returns a support answer with source references.

By default, the project uses deterministic rule-based generation so it can run without external API access. To enable LLM-based grounded generation, set:

USE_LLM_GENERATION=true
OPENAI_API_KEY=your_api_key
LLM_MODEL_NAME=gpt-4o-mini

Run:

python src/generator.py

4. Agent Workflow Orchestration

The orchestrator coordinates answer generation, confidence scoring, confusion detection, next-action decisions, escalation recommendations, and ticket draft creation.

The API calls:

run_support_workflow(question)

This creates a structured workflow state for every support request.

5. Ticket Classification and Routing

The classifier predicts:

* Ticket summary
* Category
* Priority
* Assigned support team

Example output:

{
  "summary": "My VPN is not working after I reset my password",
  "category": "VPN Connectivity",
  "priority": "Medium",
  "assigned_team": "Network Support"
}

Run:

python src/ticket_classifier.py

6. FastAPI Backend

Start the API server:

uvicorn src.api:app --reload

Open Swagger docs:

http://127.0.0.1:8000/docs

Available endpoints:

GET  /
GET  /health
POST /ask
GET  /logs
POST /feedback
GET  /feedback
GET  /metrics
POST /documents/upload

Example /ask request:

{
  "question": "My VPN is not working after I reset my password"
}

Example response:

{
  "request_id": "6ed8718a-597a-4926-9f84-d93a7fe1507b",
  "question": "My VPN is not working after I reset my password",
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
  "fallback_triggered": false,
  "answer_generation_mode": "llm",
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

7. Streamlit UI

Start FastAPI first:

uvicorn src.api:app --reload

Then open a second terminal and run:

streamlit run app/streamlit_app.py

The UI allows users to enter support questions and view:

* Generated answer
* Source documents
* Ticket category
* Priority
* Assigned team
* Agent next action
* Confidence scores
* Structured ticket draft
* Latency
* Fallback status
* Feedback form
* Analytics dashboard
* Document upload workflow

8. Query Logging, Feedback, and Metrics

Every API query is logged to:

logs/query_logs.jsonl

User feedback is logged to:

logs/feedback_logs.jsonl

Each log stores:

* Request ID
* Timestamp
* Question
* Answer
* Sources
* Ticket recommendation
* Fallback status
* Confidence scores
* Agent decision
* Latency

The metrics endpoint summarizes:

* Total queries
* Total feedback submissions
* Fallback rate
* Average latency
* Average confidence
* Agent decision counts
* Ticket category counts
* Helpful feedback rate

9. Security

API key authentication is configurable through:

SUPPORT_API_KEY=your_api_key

When this value is empty, local development endpoints remain open. When it is set, protected endpoints require:

x-api-key: your_api_key

10. Document Upload

The platform supports uploading Markdown, text, and PDF documents. Uploaded documents are saved under data/uploads/. The upload endpoint can optionally reindex the vector store after a document is saved.

Example log entry:

{
  "timestamp": "2026-06-15T17:51:07.894699Z",
  "request_id": "6ed8718a-597a-4926-9f84-d93a7fe1507b",
  "question": "My VPN is not working after I reset my password",
  "sources": ["vpn_troubleshooting_kb.md", "password_reset_kb.md"],
  "ticket": {
    "category": "VPN Connectivity",
    "priority": "Medium",
    "assigned_team": "Network Support"
  },
  "fallback_triggered": false,
  "agent_decision": {
    "next_action": "answer_and_create_ticket_draft"
  },
  "latency_ms": 1749.13
}

Setup Instructions

1. Clone the repository

git clone https://github.com/swathiblrs/Enterprise-Rag-Support-Platform.git
cd Enterprise-Rag-Support-Platform

2. Create a virtual environment

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Ingest documents

python src/ingest.py

5. Start FastAPI backend

uvicorn src.api:app --reload

6. Start Streamlit frontend

Open a second terminal:

source .venv/bin/activate
streamlit run app/streamlit_app.py

7. Run tests and evaluation

make test
make evaluate

8. Run with Docker Compose

docker compose up --build

FastAPI will be available at:

http://127.0.0.1:8000

Streamlit will be available at:

http://127.0.0.1:8501

CI/CD

GitHub Actions runs on push and pull request.

The CI workflow validates:

* Dependency installation
* API tests
* Workflow evaluation

Sample Questions

Try these questions:

My VPN is not working after I reset my password.
I cannot approve Duo push notifications.
My account is locked.
Company-wide authentication failure.
Multiple users cannot access VPN.

Current Status

Completed:

- Project setup
- Synthetic knowledge-base documents
- Document ingestion
- Chunking
- Embedding generation
- ChromaDB vector store
- Semantic retriever
- RAG-style answer generation
- Source references
- Ticket classification
- Priority prediction
- Team routing
- Agent workflow orchestration
- Confidence scoring
- Ticket draft generation
- Feedback logging
- Metrics endpoint
- Optional LLM grounded generation
- API key authentication
- PDF/document upload support
- Expanded RAG evaluation metrics
- Dockerfile
- Docker Compose
- Makefile
- GitHub Actions CI
- FastAPI backend
- Streamlit UI
- Query logging
- GitHub repository setup

In progress / planned:

- More knowledge-base documents
- Improved unsupported-question fallback
- Metadata-aware retrieval
- Stronger LLM answer-quality evaluation
- Authentication roles and user management
- External ITSM integration
- README screenshots and architecture diagram

Production-Ready Project Roadmap

This project is currently an MVP version of an enterprise RAG support automation platform. The current implementation demonstrates the core workflow: document ingestion, vector search, answer generation, source reference generation, ticket classification, routing, API access, Streamlit UI, and query logging.

The long-term goal is to evolve this into a production-style enterprise project that demonstrates scalable retrieval architecture, evaluation, observability, automation, testing, and deployment readiness.

Current Completion Estimate

Beginner/Intermediate RAG project: 90% complete
Production-style enterprise project: 65–70% complete

The current version proves the end-to-end RAG workflow. The remaining work focuses on making the system more measurable, scalable, reliable, and closer to real enterprise engineering standards.

Roadmap to Production-Ready Version

Phase 1: Evaluation and Quality Metrics

Add an evaluation pipeline to measure the quality of retrieval, routing, and system behavior.

Planned metrics:

- Retrieval accuracy
- Source reference correctness
- Ticket category accuracy
- Team routing accuracy
- Priority prediction accuracy
- Fallback detection accuracy
- Average response latency

Expected outcome:

The project should not only generate answers, but also measure whether the answers, retrieved sources, and routing decisions are correct.

Phase 2: Expanded Knowledge Base

Increase the size and diversity of the knowledge base.

Planned additions:

- Wi-Fi troubleshooting
- Email access issues
- Software installation requests
- Laptop hardware support
- Printer troubleshooting
- Production outage handling
- Security incident escalation
- Access request workflows

Expected outcome:

The assistant should handle a wider range of realistic enterprise IT support issues.

Phase 3: PDF and Document Upload Support

Add support for more document formats beyond Markdown.

Planned additions:

- PDF ingestion using pypdf
- Text extraction from uploaded files
- Metadata extraction
- Document source tracking
- Admin upload workflow

Expected outcome:

The platform should support realistic enterprise knowledge sources such as PDFs, manuals, internal docs, and FAQ files.

Phase 4: Hybrid Retrieval

Improve retrieval quality by combining semantic search and keyword search.

Planned additions:

- Vector search using ChromaDB
- Keyword search using BM25
- Hybrid retrieval scoring
- Top-k result merging
- Metadata-based filtering

Expected outcome:

The retriever should handle both meaning-based questions and exact keyword-based queries.

Phase 5: Reranking

Add a reranking layer after retrieval.

Planned additions:

- Retrieve more candidate chunks
- Rerank top results based on relevance
- Improve final context quality
- Reduce irrelevant source references

Expected outcome:

The system should return more accurate context before generating answers.

Phase 6: LLM-Based Answer Generation

Replace the current rule-based answer generator with an LLM-based generation layer.

Planned additions:

- Prompt template for grounded answers
- Context-aware answer generation
- Source-backed responses
- Guardrails against unsupported answers
- Fallback response when context is insufficient

Expected outcome:

The assistant should generate more natural and flexible answers while staying grounded in retrieved documents.

Phase 7: Improved Ticket Intelligence

Make ticket automation more advanced.

Planned additions:

- Better intent classification
- Improved priority prediction
- Confidence scores
- Recommended next actions
- Escalation detection
- Multi-user impact detection

Expected outcome:

The system should behave more like an intelligent ITSM assistant, not just a basic classifier.

Phase 8: Monitoring and Analytics Dashboard

Expand logging into a monitoring layer.

Planned additions:

- Query history dashboard
- Latency trends
- Most common ticket categories
- Fallback rate
- Source usage frequency
- User feedback tracking

Expected outcome:

The project should demonstrate observability and operational awareness.

Phase 9: Testing

Add automated tests for the backend and core logic.

Planned additions:

- Unit tests for classifier
- Unit tests for retriever
- API tests for FastAPI endpoints
- Evaluation test set
- Regression tests for routing behavior

Expected outcome:

The project should demonstrate software engineering quality and maintainability.

Phase 10: Infrastructure and DevOps

Add deployment-ready tooling.

Planned additions:

- Dockerfile
- Docker Compose
- GitHub Actions CI
- Environment-based configuration
- Secrets handling
- Makefile or run scripts

Expected outcome:

The project should be easy to run, test, and deploy in a clean environment.

Phase 11: Multi-Domain RAG

Add a second knowledge domain to show scalability.

Possible addition:

- Enterprise IT support domain
- Public university student-services domain

Example public-information domain:

- CPT policy
- On-campus employment rules
- Graduate assistantship information
- Enrollment requirements

Expected outcome:

The project should demonstrate domain routing and metadata-aware retrieval across multiple knowledge areas.

Final Target Architecture

The final version aims to include the following layers:

1. Data Ingestion Layer
2. Document Processing Pipeline
3. Indexing and Storage Layer
4. Hybrid Retrieval Layer
5. RAG and Generation Layer
6. Automation and Ticket Intelligence Layer
7. Evaluation and Monitoring Layer
8. Observability and Logging Layer
9. Infrastructure and DevOps Layer
10. Security, Privacy, and Access Control Considerations

Definition of Production-Ready Completion

This project will be considered production-ready when it demonstrates:

- End-to-end RAG workflow
- Multi-format document ingestion
- Hybrid retrieval
- Source-grounded LLM answers
- Ticket automation and routing
- Measurable evaluation metrics
- API and UI layers
- Logging and monitoring
- Automated tests
- Dockerized deployment
- Clean GitHub documentation
- Clear system design explanation

Estimated Remaining Work

Current MVP completion: 40%
Remaining for strong resume-ready version: 30–40%
Remaining for production-ready version: 60–65%

The next recommended implementation step is the evaluation pipeline because it proves the system can be measured and improved objectively.

Planned Evaluation Metrics

The evaluation module will measure:

* Retrieval accuracy
* Routing accuracy
* Priority prediction accuracy
* Fallback accuracy
* Average response latency
* Source reference correctness

Future Improvements

* Add public student-services documents as a second knowledge domain
* Add support for PDF ingestion using pypdf
* Add hybrid retrieval using semantic search and BM25
* Add LLM integration for more natural answer generation
* Add admin upload page for new documents
* Add Docker and Docker Compose
* Add GitHub Actions CI
* Add richer dashboard for query analytics
* Add user feedback collection
* Add role-based access control design notes



Disclaimer

This project uses synthetic support documents for demonstration purposes. It is not connected to any real enterprise ITSM system and does not use private company, university, or personal data.

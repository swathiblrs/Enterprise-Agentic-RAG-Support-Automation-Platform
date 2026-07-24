import base64
import os

import requests
import streamlit as st


BASE_API_URL = os.getenv("SUPPORT_API_BASE_URL", "http://127.0.0.1:8000")
SUPPORT_API_KEY = os.getenv("SUPPORT_API_KEY", "")
ASK_API_URL = f"{BASE_API_URL}/ask"
LOGIN_API_URL = f"{BASE_API_URL}/auth/login"
FEEDBACK_API_URL = f"{BASE_API_URL}/feedback"
METRICS_API_URL = f"{BASE_API_URL}/metrics"
UPLOAD_API_URL = f"{BASE_API_URL}/documents/upload"


def auth_headers() -> dict:
    if "access_token" in st.session_state:
        return {"Authorization": f"Bearer {st.session_state['access_token']}"}

    if not SUPPORT_API_KEY:
        return {}

    return {"x-api-key": SUPPORT_API_KEY}


st.set_page_config(
    page_title="Enterprise RAG Support Assistant",
    page_icon="🤖",
    layout="wide",
)


st.title("Enterprise RAG Support Assistant")

with st.sidebar:
    st.header("Access")

    if "access_token" in st.session_state:
        st.write(f"Signed in as `{st.session_state.get('auth_role', 'unknown')}`")
        if st.button("Sign Out"):
            del st.session_state["access_token"]
            st.session_state.pop("auth_role", None)
    else:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Sign In")

        if login_submitted:
            try:
                login_response = requests.post(
                    LOGIN_API_URL,
                    json={"username": username, "password": password},
                    timeout=30,
                )

                if login_response.status_code == 200:
                    login_data = login_response.json()
                    st.session_state["access_token"] = login_data["access_token"]
                    st.session_state["auth_role"] = login_data["role"]
                    st.success("Signed in.")
                else:
                    st.error(f"Login failed: {login_response.status_code}")
            except Exception as error:
                st.error(f"Could not sign in: {error}")

ask_tab, analytics_tab, upload_tab = st.tabs(["Ask", "Analytics", "Upload Documents"])


with ask_tab:
    st.write(
        "Ask an IT support question. The system will retrieve knowledge-base context, "
        "generate an answer, show sources, recommend ticket routing, and expose agent workflow decisions."
    )

    question = st.text_area(
        "Enter your support question:",
        placeholder="Example: My VPN is not working after I reset my password",
        height=120,
    )
    domain = st.text_input("Knowledge domain", value="it_support")

    if st.button("Ask Assistant"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving knowledge and generating answer..."):
                try:
                    response = requests.post(
                        ASK_API_URL,
                        json={"question": question, "domain": domain},
                        headers=auth_headers(),
                        timeout=60,
                    )

                    if response.status_code != 200:
                        st.error(f"API error: {response.status_code}")
                        st.text(response.text)
                    else:
                        st.session_state["last_response"] = response.json()

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Could not connect to the FastAPI backend. "
                        "Please start it using: uvicorn src.api:app --reload"
                    )

                except Exception as error:
                    st.error(f"Unexpected error: {error}")

    if "last_response" in st.session_state:
        data = st.session_state["last_response"]

        st.subheader("Answer")
        st.write(data["answer"])

        st.subheader("Sources")
        if data["sources"]:
            for source in data["sources"]:
                st.write(f"- {source}")
        else:
            st.write("No sources found.")

        st.subheader("Ticket Recommendation")
        ticket = data["ticket"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Category", ticket["category"])

        with col2:
            st.metric("Priority", ticket["priority"])

        with col3:
            st.metric("Assigned Team", ticket["assigned_team"])

        st.write("**Summary:**", ticket["summary"])

        st.subheader("Agent Workflow")
        decision = data["agent_decision"]
        confidence = data["confidence"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Next Action", decision["next_action"])

        with col2:
            st.metric("Overall Confidence", confidence["overall_confidence"])

        with col3:
            st.metric("Generation", data.get("answer_generation_mode", "unknown"))

        with col4:
            st.metric("Fallback", str(data["fallback_triggered"]))

        st.write("**Reason:**", decision["reason"])

        st.subheader("Ticket Draft")
        ticket_draft = data["ticket_draft"]
        st.write("**Title:**", ticket_draft["title"])
        st.write("**Description:**", ticket_draft["description"])
        st.write("**Suggested Steps:**")
        for step in ticket_draft["suggested_steps"]:
            st.write(f"- {step}")

        if data.get("integrations"):
            st.subheader("Integration Status")
            itsm = data["integrations"].get("itsm", {})
            chat = data["integrations"].get("chat", {})

            col1, col2 = st.columns(2)
            with col1:
                st.metric("ITSM", itsm.get("status", "unknown"))
                st.caption(itsm.get("message", ""))
            with col2:
                st.metric("Chat", chat.get("status", "unknown"))
                st.caption(chat.get("message", ""))

        st.subheader("System Metadata")
        st.write(f"Request ID: `{data['request_id']}`")
        st.write(f"Domain: `{data.get('domain', 'it_support')}`")
        st.write(f"Latency: `{data['latency_ms']} ms`")
        st.write(f"Workflow Engine: `{data.get('workflow_engine', 'unknown')}`")

        engineering_metrics = data.get("engineering_metrics", {})
        if engineering_metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Workflow",
                    f"{engineering_metrics.get('total_workflow_latency_ms', 0.0)} ms",
                )
            with col2:
                st.metric(
                    "Answer Stage",
                    f"{engineering_metrics.get('answer_stage_latency_ms', 0.0)} ms",
                )
            with col3:
                st.metric("Chunks", engineering_metrics.get("retrieved_chunk_count", 0))
            with col4:
                st.metric("Sources", engineering_metrics.get("source_count", 0))

        st.subheader("Feedback")
        with st.form("feedback_form"):
            answer_helpful = st.checkbox("Answer was helpful", value=True)
            correct_sources = st.checkbox("Sources were correct", value=True)
            correct_ticket_routing = st.checkbox("Ticket routing was correct", value=True)
            correct_priority = st.checkbox("Priority was correct", value=True)
            comments = st.text_area("Comments", height=80)

            submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            feedback_payload = {
                "request_id": data["request_id"],
                "question": data["question"],
                "answer_helpful": answer_helpful,
                "correct_sources": correct_sources,
                "correct_ticket_routing": correct_ticket_routing,
                "correct_priority": correct_priority,
                "comments": comments,
            }

            try:
                feedback_response = requests.post(
                    FEEDBACK_API_URL,
                    json=feedback_payload,
                    headers=auth_headers(),
                    timeout=30,
                )

                if feedback_response.status_code == 200:
                    st.success("Feedback submitted.")
                else:
                    st.error(f"Feedback API error: {feedback_response.status_code}")
            except Exception as error:
                st.error(f"Could not submit feedback: {error}")


with analytics_tab:
    st.write("Operational metrics from recent query and feedback logs.")

    if st.button("Refresh Metrics"):
        try:
            metrics_response = requests.get(
                METRICS_API_URL,
                headers=auth_headers(),
                timeout=30,
            )

            if metrics_response.status_code == 200:
                st.session_state["metrics"] = metrics_response.json()
            else:
                st.error(f"Metrics API error: {metrics_response.status_code}")
        except Exception as error:
            st.error(f"Could not load metrics: {error}")

    if "metrics" in st.session_state:
        metrics = st.session_state["metrics"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Queries", metrics["total_queries"])
        with col2:
            st.metric("Fallback Rate", metrics["fallback_rate"])
        with col3:
            st.metric("Avg Latency", f"{metrics['average_latency_ms']} ms")
        with col4:
            st.metric("Helpful Rate", metrics["feedback_helpful_rate"])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Workflow Latency", f"{metrics.get('average_workflow_latency_ms', 0.0)} ms")
        with col2:
            st.metric("Answer Stage", f"{metrics.get('average_answer_stage_latency_ms', 0.0)} ms")
        with col3:
            st.metric("Avg Chunks", metrics.get("average_retrieved_chunk_count", 0.0))
        with col4:
            st.metric("Avg Sources", metrics.get("average_source_count", 0.0))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ticket Categories")
            st.bar_chart(metrics["ticket_category_counts"])
        with col2:
            st.subheader("Agent Decisions")
            st.bar_chart(metrics["agent_decision_counts"])


with upload_tab:
    st.write("Upload Markdown, text, or PDF documents into the knowledge base.")

    uploaded_file = st.file_uploader("Choose a document", type=["md", "txt", "pdf"])
    upload_domain = st.text_input("Document domain", value="it_support")
    source_type = st.text_input("Source type", value="upload")
    reindex = st.checkbox("Reindex vector store after upload", value=False)

    if st.button("Upload Document"):
        if uploaded_file is None:
            st.warning("Please choose a document first.")
        else:
            payload = {
                "filename": uploaded_file.name,
                "content_base64": base64.b64encode(uploaded_file.getvalue()).decode("utf-8"),
                "domain": upload_domain,
                "source_type": source_type,
                "reindex": reindex,
            }

            try:
                upload_response = requests.post(
                    UPLOAD_API_URL,
                    json=payload,
                    headers=auth_headers(),
                    timeout=120,
                )

                if upload_response.status_code == 200:
                    st.success("Document uploaded.")
                    st.json(upload_response.json())
                else:
                    st.error(f"Upload API error: {upload_response.status_code}")
                    st.text(upload_response.text)
            except Exception as error:
                st.error(f"Could not upload document: {error}")

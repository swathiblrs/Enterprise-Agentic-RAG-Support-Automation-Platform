"""Provider HTTP contract tests; no requests leave the process."""
import json
import sqlite3
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import api as app
from external_clients import ExternalAPIs, IntegrationError
from external_routes import get_clients


ENV = {
    "EXTERNAL_APIS_ENABLED": "true", "INTEGRATION_ADMIN_KEY": "test-only-integration-key-123456",
    "JIRA_BASE_URL": "https://example.atlassian.net", "JIRA_EMAIL": "tester@example.invalid",
    "JIRA_API_TOKEN": "test-jira-token", "JIRA_PROJECT_KEY": "HELP", "JIRA_ISSUE_TYPE_ID": "10001",
    "MICROSOFT_GRAPH_ACCESS_TOKEN": "test-graph-token", "GOOGLE_DRIVE_ACCESS_TOKEN": "test-drive-token",
    "EXTERNAL_OPENAI_API_KEY": "test-model-key", "EXTERNAL_OPENAI_MODEL": "test-model",
    "SLACK_BOT_TOKEN": "test-slack-token", "SLACK_CHANNEL_ID": "C_TEST",
}
CASE = {"id": "case-one", "question": "VPN cannot connect", "answer": "Check approved VPN guidance.",
        "routing": {"domain": "it"}, "ticket_draft": {"title": "VPN failure", "description": "Cannot connect",
        "assigned_team": "Network Support", "priority": "medium"},
        "support": {"retrieved_chunks": [{"source": "vpn.md", "text": "Use the approved VPN client."}]}}


def provider_response(request):
    path = request.url.path
    if request.url.host == "example.atlassian.net":
        assert request.method == "POST" and path == "/rest/api/3/issue"
        assert request.headers["authorization"].startswith("Basic ")
        fields = json.loads(request.content)["fields"]
        assert fields["project"]["key"] == "HELP"
        assert fields["description"]["type"] == "doc"
        assert "Network Support" in fields["description"]["content"][0]["content"][0]["text"]
        return httpx.Response(201, json={"key":"HELP-42", "id":"42"})
    if request.url.host == "graph.microsoft.com":
        assert request.method == "GET" and path == "/v1.0/me"
        assert request.headers["authorization"] == "Bearer test-graph-token"
        assert "$select" in request.url.params
        return httpx.Response(200, json={"id":"user-1","displayName":"Test User","department":"Support"})
    if request.url.host == "www.googleapis.com":
        assert request.headers["authorization"] == "Bearer test-drive-token"
        if path.endswith("/export"):
            assert request.url.params["mimeType"] == "text/plain"
            return httpx.Response(200, text="Use the approved VPN client.")
        if request.url.params.get("alt") == "media":
            return httpx.Response(200, text="Plain document content.")
        mime = "text/plain" if path.endswith("/plain-file") else "application/vnd.google-apps.document"
        return httpx.Response(200, json={"id":path.rsplit("/",1)[1],"name":"VPN guide","mimeType":mime})
    if request.url.host == "api.openai.com":
        assert path == "/v1/responses" and request.method == "POST"
        body = json.loads(request.content)
        assert body["store"] is False and body["max_output_tokens"] == 600
        assert "approved VPN" in body["input"]
        return httpx.Response(200, json={"id":"resp-test", "status":"completed", "output":[
            {"type":"message","content":[{"type":"output_text","text":"Use the approved VPN client [S1]."}]}]})
    if request.url.host == "slack.com":
        assert path == "/api/chat.postMessage" and request.method == "POST"
        body = json.loads(request.content)
        assert body["channel"] == "C_TEST" and "case-one" in body["text"]
        assert body["unfurl_links"] is False
        return httpx.Response(200, json={"ok":True,"channel":"C_TEST","ts":"123.456"})
    raise AssertionError("Unexpected provider host")


@pytest.fixture
def clients():
    return ExternalAPIs(ENV, httpx.MockTransport(provider_response), sleep=lambda _: None)


@pytest.fixture
def web(tmp_path, monkeypatch, clients):
    database = tmp_path / "cases.sqlite3"
    monkeypatch.setattr(app, "CASES_DB", database)
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE cases (id TEXT PRIMARY KEY, owner TEXT, result TEXT)")
        db.execute("INSERT INTO cases VALUES (?,?,?)", ("case-one","demo",json.dumps(CASE)))
    app.api.dependency_overrides[get_clients] = lambda: clients
    with TestClient(app.api) as client:
        token = client.post("/auth/login",json={"username":"demo","password":"local-demo-only"}).json()["access_token"]
        client.headers["Authorization"] = "Bearer " + token
        client.headers["X-Integration-Key"] = ENV["INTEGRATION_ADMIN_KEY"]
        yield client
    app.api.dependency_overrides.clear()


def test_jira_contract(clients):
    assert clients.create_jira_ticket(CASE)["issue_key"] == "HELP-42"


def test_graph_contract(clients):
    assert clients.graph_me()["user"]["department"] == "Support"


@pytest.mark.parametrize("file_id", ["doc-file", "plain-file"])
def test_drive_contract(clients, file_id):
    result = clients.drive_document(file_id)
    assert result["text"] and result["source"] == "drive:" + file_id


def test_openai_contract(clients):
    result = clients.grounded_answer(CASE["question"],CASE["support"]["retrieved_chunks"])
    assert "[S1]" in result["answer"] and result["requires_human_review"]


def test_slack_contract(clients):
    assert clients.notify_slack(CASE)["message_ts"] == "123.456"


def test_disabled_and_missing_config_never_call_network():
    def unexpected(request):
        raise AssertionError("Network must not be called")
    for config in ({}, {"EXTERNAL_APIS_ENABLED":"true"}):
        client = ExternalAPIs(config,httpx.MockTransport(unexpected))
        for call in (client.graph_me, lambda:client.drive_document("file"),lambda:client.create_jira_ticket(CASE),
                     lambda:client.grounded_answer("question",[{"text":"context"}]),lambda:client.notify_slack(CASE)):
            with pytest.raises(IntegrationError) as error:
                call()
            assert error.value.status == 503


def test_read_rate_limit_retry_is_bounded():
    calls = []
    def rate_limited(request):
        calls.append(True)
        return httpx.Response(429,headers={"Retry-After":"999"})
    sleeps = []
    client = ExternalAPIs(ENV,httpx.MockTransport(rate_limited),sleep=sleeps.append)
    with pytest.raises(IntegrationError) as error:
        client.graph_me()
    assert error.value.status == 429 and len(calls)==3 and sleeps == [2,2]


def test_post_timeout_is_unknown_and_never_retried():
    calls = []
    def timeout(request):
        calls.append(True)
        raise httpx.ReadTimeout("secret that must not leak",request=request)
    client = ExternalAPIs(ENV,httpx.MockTransport(timeout))
    with pytest.raises(IntegrationError) as error:
        client.create_jira_ticket(CASE)
    assert error.value.outcome_unknown and len(calls)==1
    assert "secret" not in str(error.value)


def test_provider_error_body_is_not_exposed():
    client = ExternalAPIs(ENV,httpx.MockTransport(lambda r:httpx.Response(401,text="private token")))
    with pytest.raises(IntegrationError) as error:
        client.graph_me()
    assert "private token" not in str(error.value)


def test_redirect_does_not_forward_credentials():
    calls = []
    def redirect(request):
        calls.append(str(request.url))
        return httpx.Response(302,headers={"Location":"https://untrusted.invalid"})
    client = ExternalAPIs(ENV,httpx.MockTransport(redirect))
    with pytest.raises(IntegrationError):
        client.graph_me()
    assert len(calls)==1


def test_jira_rejects_arbitrary_hosts():
    client = ExternalAPIs({**ENV,"JIRA_BASE_URL":"https://localhost"},httpx.MockTransport(provider_response))
    with pytest.raises(IntegrationError):
        client.create_jira_ticket(CASE)


def test_drive_rejects_unsupported_files_and_oversized_content():
    client = ExternalAPIs(ENV,httpx.MockTransport(lambda r:httpx.Response(200,json={
        "id":"file","mimeType":"application/pdf"})))
    with pytest.raises(IntegrationError) as error:
        client.drive_document("file")
    assert error.value.status == 422
    client = ExternalAPIs(ENV,httpx.MockTransport(lambda r:httpx.Response(200,content=b"x"*2_000_001)))
    with pytest.raises(IntegrationError):
        client.graph_me()


def test_slack_application_error_is_not_success():
    client = ExternalAPIs(ENV,httpx.MockTransport(lambda r:httpx.Response(200,json={"ok":False,"error":"not_in_channel"})))
    with pytest.raises(IntegrationError):
        client.notify_slack(CASE)


def test_empty_model_output_is_not_success():
    client = ExternalAPIs(ENV,httpx.MockTransport(lambda r:httpx.Response(200,json={"status":"incomplete","output":[]})))
    with pytest.raises(IntegrationError):
        client.grounded_answer("Question",[{"text":"context"}])


def test_integration_endpoints_require_jwt_and_separate_key(web):
    web.headers.pop("Authorization")
    assert web.get("/integrations").status_code == 401
    token = web.post("/auth/login",json={"username":"demo","password":"local-demo-only"}).json()["access_token"]
    web.headers["Authorization"] = "Bearer " + token
    web.headers.pop("X-Integration-Key")
    assert web.get("/integrations/microsoft-graph/me").status_code == 403


def test_catalog_is_five_providers_without_secrets(web):
    response = web.get("/integrations")
    assert len(response.json()["providers"]) == 5
    assert "test-jira-token" not in response.text
    assert all(not p["live_verified"] for p in response.json()["providers"])


def test_read_endpoints_and_case_history_use_configured_database(web):
    assert web.get("/integrations/microsoft-graph/me").json()["user"]["id"] == "user-1"
    assert web.get("/integrations/google-drive/files/doc-file").json()["text"]
    assert web.get("/cases").json()[0]["id"] == "case-one"


def test_confirmed_jira_and_slack_actions_are_idempotent(web):
    for path in ("jira-ticket","slack-notification"):
        assert web.post("/cases/case-one/"+path,json={"confirmed":False}).status_code == 422
        first = web.post("/cases/case-one/"+path,json={"confirmed":True})
        second = web.post("/cases/case-one/"+path,json={"confirmed":True})
        assert first.status_code==200 and second.status_code==200
        assert first.json()["reused"] is False and second.json()["reused"] is True


def test_drive_context_flows_into_case_answer(web):
    response = web.post("/cases/case-one/external-answer",json={"confirmed":True,"drive_file_id":"doc-file"})
    assert response.status_code == 200
    assert response.json()["sources"][0]["source"] == "drive:doc-file"
    assert "[S1]" in response.json()["answer"]


def test_other_users_cannot_submit_case_actions(web):
    web.headers["Authorization"] = "Bearer " + app.create_access_token("another-user",app.UserRole.admin)
    assert web.post("/cases/case-one/jira-ticket",json={"confirmed":True}).status_code == 404


def test_unknown_post_outcome_blocks_repeated_submission(web,clients):
    calls = []
    def timeout(request):
        calls.append(True)
        raise httpx.ReadTimeout("timed out",request=request)
    clients.transport = httpx.MockTransport(timeout)
    first = web.post("/cases/case-one/jira-ticket",json={"confirmed":True})
    second = web.post("/cases/case-one/jira-ticket",json={"confirmed":True})
    assert first.status_code==502 and second.status_code==409 and len(calls)==1

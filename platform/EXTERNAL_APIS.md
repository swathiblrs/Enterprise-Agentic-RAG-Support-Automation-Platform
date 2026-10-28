# Five External API Adapters

The platform implements five provider-specific REST adapters and authenticated endpoints. Their HTTP contracts are tested with simulated provider responses. Live provider verification has not been performed. No service credentials are bundled, and normal case processing stays offline.

| Provider | Implemented operation | Platform endpoint |
| --- | --- | --- |
| Jira Cloud REST API v3 | Create an issue from a saved ticket draft | POST /cases/{case_id}/jira-ticket |
| Microsoft Graph v1.0 | Read the profile associated with a delegated access token | GET /integrations/microsoft-graph/me |
| Google Drive API v3 | Retrieve a Google Doc, plain text, or Markdown document | GET /integrations/google-drive/files/{file_id} |
| OpenAI Responses API | Generate an answer using saved RAG context and optionally a Drive document | POST /cases/{case_id}/external-answer |
| Slack Web API | Post a case notification to the configured channel | POST /cases/{case_id}/slack-notification |

GET /integrations lists all five adapters and missing configuration names. Configured means values are present; it does not prove valid credentials, permissions, or connectivity.

## How They Fit the Workflow

Create and review a case through POST /cases. Retrieve an authorized Drive document if extra context is needed, then explicitly request an external model answer. Review the resulting guidance. Submit the saved ticket draft to Jira or send a Slack notification as separate confirmed actions.

The Graph endpoint reads the configured token holder's profile. It is not SSO, automatic identity matching, or arbitrary organization-wide directory search. Drive content can be included directly in an external answer; it is not automatically added to ChromaDB or refreshed in the local index. Jira uses the configured project and issue type. Recommended team/priority appear in its description; automatic assignee mapping and Jira priority changes are not implemented.

## Configure Authorized Test Accounts

Use platform/external-apis.env.example as a reference. Create a local .env.integrations file in this directory; it is ignored by Git. Set EXTERNAL_APIS_ENABLED=true and provide only the providers you intend to use. Generate a separate random INTEGRATION_ADMIN_KEY of at least 24 characters.

- Jira requires an authorized Atlassian Cloud site, account email/API token, project key, and valid issue-type ID. The implementation uses Jira Cloud REST v3 issue creation, not the separate Jira Service Management customer-request API. Project-required custom fields may need additional implementation.
- Microsoft Graph requires a delegated token for the intended account with User.Read. The adapter does not obtain or refresh OAuth tokens.
- Google Drive requires an authorized access token with a suitable Drive read scope. File permissions still apply. Google Docs exports as text; PDFs, Sheets, Slides, and binary files are not supported by this adapter.
- OpenAI requires EXTERNAL_OPENAI_API_KEY and an accessible model name in EXTERNAL_OPENAI_MODEL. These distinct variable names avoid the offline workflow resetting them. Calls may incur provider charges.
- Slack requires a bot token with chat:write and access to SLACK_CHANNEL_ID.

Use personal test workspaces or accounts you are authorized to connect. Keep the current demo server bound to localhost.

```sh
export SUPPORT_PYTHON="$PWD/.venv-support/bin/python"
.venv-api/bin/python -m uvicorn api:api --host 127.0.0.1 --port 8765 --env-file .env.integrations
```

Open /docs and authenticate using POST /auth/login. Include the returned JWT as the Bearer token. For integration operations, also supply the X-Integration-Key header matching INTEGRATION_ADMIN_KEY. The separate key is required because the current demo account uses a shared local password.

Mutation/answer endpoints require:

```json
{"confirmed": true}
```

To add authorized Drive context to the model answer:

```json
{"confirmed": true, "drive_file_id": "YOUR_FILE_ID"}
```

The OpenAI request uses the Responses API with store=false, bounded output tokens, supplied-source instructions, and citation identifiers. This does not independently verify factual accuracy or guarantee a provider's data-retention policy.

## Reliability and Access

- Existing JWT authentication remains required. Integration operations require an admin role plus the separate integration key.
- Case actions check ownership before retrieving or submitting case content.
- Each case/action is reserved in SQLite before an outbound POST. Successful actions return the stored receipt when repeated.
- POST requests are not automatically retried. Timeouts, ambiguous server errors, and invalid success receipts retain an unknown outcome that blocks replay. Reconcile with the provider before manually resetting an unknown/pending action record.
- GET retries for 429/502/503/504 are bounded to three attempts and bounded delays.
- HTTP calls have timeouts, no automatic redirects, and a 2 MB response-size limit. Provider error bodies and credentials are not included in error responses.
- The demo's offline latency/load benchmark excludes these live provider calls. Those earlier metrics must not be attributed to this integration path.

## Tests and Current Verification

From this directory:

```sh
.venv-api/bin/python -m pytest test_external_apis.py -q
```

Tests cover all five provider request/response contracts, disabled/missing configuration, rate limits, timeouts, invalid data, protected endpoints, case ownership, explicit confirmation, duplicate prevention, and Drive-to-model context flow. HTTP responses are simulated using httpx.MockTransport.

Supported resume wording at this stage:

> Implemented five REST API adapters for Jira, Microsoft Graph, Google Drive, OpenAI, and Slack, with authenticated endpoints, contract tests, bounded retries, and duplicate-submission protection.

Live integration or production adoption should be claimed only after credential-backed testing records real provider receipts.

## Provider References

- [Jira Cloud issue API](https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-issues/)
- [Microsoft Graph user retrieval](https://learn.microsoft.com/en-us/graph/api/user-get?view=graph-rest-1.0)
- [Google Drive file retrieval](https://developers.google.com/workspace/drive/api/reference/rest/v3/files/get) and [export](https://developers.google.com/workspace/drive/api/reference/rest/v3/files/export)
- [OpenAI text generation](https://developers.openai.com/api/docs/guides/text)
- [Slack chat.postMessage](https://docs.slack.dev/reference/methods/chat.postMessage/)

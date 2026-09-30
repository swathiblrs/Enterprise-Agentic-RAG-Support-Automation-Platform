import json
from pathlib import Path
from fastapi.testclient import TestClient
from combined import Request, api, execute, route

def test_auth_and_clarification():
    client = TestClient(api)
    assert client.post('/cases', json={'question': 'Reset my password'}).status_code == 401
    assert client.post('/auth/login', json={'username': 'demo', 'password': 'wrong'}).status_code == 401
    token = client.post('/auth/login', json={'username': 'demo', 'password': 'local-demo-only'}).json()['access_token']
    headers = {'Authorization': 'Bearer ' + token}
    response = client.post('/cases', headers=headers, json={'question': 'Something is broken'})
    assert response.status_code == 200
    assert response.json()['routing']['path'] == 'clarify'
    assert response.json()['ticket_draft'] is None
    assert client.get('/cases', headers=headers).status_code == 200

def test_routing():
    assert route(Request(question='How do I reset my password?'))['path'] == 'support'
    assert route(Request(question='VPN is down for all users'))['path'] == 'investigate'
    assert route(Request(question='VPN login failed', logs=['certificate expired']))['path'] == 'investigate'
    assert route(Request(question='Help with design'))['path'] == 'clarify'
    assert route(Request(question='VPN issue', domain='security'))['domain'] == 'security'

def test_real_workflows():
    cases = [
        ('routine_support', Request(question='How do I reset my password?'), 'support', 'it'),
        ('it_escalation', Request(question='VPN outage for all users after a change', severity='critical', logs=['vpn gateway connection failed']), 'investigate', 'it'),
        ('security', Request(question='Suspicious login and impossible-travel account takeover', events=[{'event': 'authentication.failed'}]), 'investigate', 'security'),
        ('production', Request(question='Checkout returns 503 after deployment', logs=['checkout HTTP 503']), 'investigate', 'production'),
        ('cloud', Request(question='AWS autoscaling quota exhausted', logs=['capacity provisioning failed']), 'investigate', 'cloud'),
        ('data', Request(question='ETL pipeline failed schema drift validation', logs=['schema validation failed']), 'investigate', 'data'),
    ]
    results = []
    for name, request, path, domain in cases:
        result = execute(request)
        assert result['routing']['path'] == path
        assert result['routing']['domain'] == domain
        assert result['ticket_draft']
        assert result['answer']
        if path == 'investigate':
            assert result['investigation']['references']
            assert result['investigation']['evidence']
            assert result['ticket_draft']['investigation_id']
        else:
            assert result['support']['sources']
        if name == 'it_escalation':
            assert result['support'] and result['investigation']
            assert result['ticket_draft']['priority'] == 'critical'
        results.append({'scenario': name, **result})
    Path('results.json').write_text(json.dumps(results, indent=2))

import json
from pathlib import Path
from unittest.mock import patch

import combined as c
from domain_classifier import predict
from evidence_provenance import annotate

def test_ml_handles_unmatched_policy_language():
    for question,domain in [('My virtual private network stopped connecting','it'),
                            ('Unexpected OAuth permissions on user identity','security'),
                            ('Ingress reports HTTP 502 errors','production')]:
        decision=c.route(c.Request(question=question))
        assert decision['domain']==domain
        assert decision['routing_mode']=='ml_fallback'
        assert decision['model_prediction']['accepted']

def test_missing_model_and_unknown_requests_clarify():
    with patch('domain_classifier.load_model',return_value=None):
        assert c.route(c.Request(question='A virtual private network fault'))['path']=='clarify'
    assert c.route(c.Request(question='CPU 98 percent please help'))['path']=='clarify'
    assert c.route(c.Request(question='Write a birthday poem'))['path']=='clarify'

def test_explicit_and_ambiguous_routes_do_not_depend_on_ml():
    with patch('routing.predict',side_effect=AssertionError('ML must not override explicit/tied domains')):
        assert c.route(c.Request(question='VPN failure',domain='cloud'))['domain']=='cloud'
        assert c.route(c.Request(question='ETL failure and Kubernetes failure'))['path']=='clarify'

def test_priority_policy_is_retrieved_with_vpn_topic():
    result=c.support('Multiple users cannot access VPN')
    assert 'vpn_troubleshooting_kb.md' in result['sources']
    assert 'priority_matrix.md' in result['sources']

def test_team_policy_is_retrieved_with_vpn_topic():
    result=c.support('Which support team handles VPN connectivity issues?')
    assert 'vpn_troubleshooting_kb.md' in result['sources']
    assert 'ticket_routing_rules.md' in result['sources']
    assert len(result['sources'])==len(set(result['sources']))

def test_evidence_origin_does_not_imply_verification():
    report=annotate({'evidence':[{'kind':'log','value':'user log'},{'kind':'mcp_tool','value':'simulated signal'},
                                {'kind':'access_impact','value':'simulated agent observation'},
                                {'kind':'triage_signal','value':'rule decision'}], 'findings':[{'summary':'assessment'}]})
    assert report['evidence_summary']=={'user_supplied':1,'local_simulation':2,'rule_derived':1,'independently_verified':0}
    assert all(not e['independently_verified'] for e in report['evidence'])
    assert report['semantic_groundedness']=='not_evaluated'
    assert report['findings'][0]['verification_status']=='unverified_offline_assessment'

def test_dataset_has_disjoint_text_splits():
    rows=json.loads(Path('domain_dataset.json').read_text())['examples']
    assert len(rows)==len({r['text'].strip().lower() for r in rows})
    assert {r['split'] for r in rows}=={'train','validation','test'}

from unittest.mock import patch
from fastapi.testclient import TestClient
import combined as c

def test_preserve_payload():
    captured = {}
    class Report:
        def model_dump(self, **kwargs):
            return {'ok':True}
    def security(self,payload):
        captured['security']=payload
        return Report()
    def generic(self,payload):
        captured['generic']=payload
        return Report()
    request=c.Request(question='Suspicious OAuth grant',id='original-123',title='Original title',description='Original description',
        source='entra-id',user='analyst',host='host-1',ip_address='192.0.2.1',geo='US',technique='T1098',tactic='Persistence',
        tags=['oauth'],events=[{'event':'oauth.grant'}],service='identity',owner_team='IAM',environment='test',metrics={'failures':4})
    with patch.object(c.InvestigationGraph,'investigate',security), patch.object(c.GenericIncidentGraph,'investigate',generic):
        c.investigate(request,'security');c.investigate(request,'it')
    alert=captured['security']
    assert (alert.id,alert.title,alert.description,alert.user,alert.ip_address,alert.technique,alert.tags)==('original-123','Original title','Original description','analyst','192.0.2.1','T1098',['oauth'])
    incident=captured['generic']
    assert (incident.service,incident.owner_team,incident.environment,incident.metrics)==('identity','IAM','test',{'failures':4})

def test_support_assignment_and_citations_survive():
    support={'answer':'Support guidance','ticket_draft':{'priority':'High','assigned_team':'Network Support','category':'VPN Connectivity','source_references':['vpn.md'],'description':'Support guidance'}}
    report={'executive_summary':'Investigation summary','investigation_id':'id','evidence':[], 'recommended_actions':[], 'references':[{'source':'incident.md'}],'risk_score':55}
    with patch.object(c,'support',return_value=support),patch.object(c,'investigate',return_value=report):
        result=c.execute(c.Request(question='Multiple users cannot access VPN'))
    ticket=result['ticket_draft']
    assert ticket['assigned_team']=='Network Support'
    assert ticket['source_references']==['vpn.md']
    assert ticket['investigation_source_references']==['incident.md']
    assert ticket['priority']=='high'
    assert ticket['description']=='Support guidance'

def test_security_assignment_is_explicit():
    support={'answer':'Guidance','ticket_draft':{'priority':'Critical','assigned_team':'Identity and Access Management','source_references':['mfa.md']}}
    report={'executive_summary':'Security finding','investigation_id':'id','evidence':[], 'recommended_actions':[], 'references':[],'risk_score':80}
    with patch.object(c,'support',return_value=support),patch.object(c,'investigate',return_value=report):
        result=c.execute(c.Request(question='Security incident affecting company authentication'))
    assert result['ticket_draft']['assigned_team']=='SOC'
    assert result['ticket_draft']['support_team']=='Identity and Access Management'
    assert result['ticket_draft']['priority']=='critical'

def test_worker_reuse_and_restart():
    first=c.support('How do I reset my password?')
    pid=c.worker.process.pid
    second=c.support('How do I reset my password?')
    assert c.worker.process.pid==pid
    assert first['sources']==second['sources']
    c.worker.process.terminate();c.worker.process.wait()
    third=c.support('How do I reset my password?')
    assert c.worker.process.pid!=pid
    assert third['sources']==first['sources']

def test_auth_before_worker():
    with patch.object(c,'execute') as execute:
        client=TestClient(c.api)
        assert client.post('/cases',json={'question':'VPN failure'}).status_code==401
        execute.assert_not_called()

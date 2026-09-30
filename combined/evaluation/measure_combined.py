import concurrent.futures
import json
import math
import os
import sys
import time
from pathlib import Path
ROOT=Path('/Users/swathibs/Desktop/projects/Combined Agentic RAG Prototype')
sys.path.insert(0,str(ROOT))
import combined as c
OUT=Path('/private/tmp/combined-evaluation')
OUT.mkdir(exist_ok=True)

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,default=str))

def sources(result):
    # Evaluate the sources attached to the final delivered ticket, not discarded intermediate retrieval.
    return list(dict.fromkeys((result.get('ticket_draft') or {}).get('source_references',[])))

def score_support(case,result,combined=False):
    actual=sources(result) if combined else result['sources']
    expected=set(case['expected_sources'])
    ticket=(result.get('ticket_draft') or {}) if combined else result['ticket']
    relevant=sum(x in expected for x in actual)
    return {'hit':int(relevant>0),'precision_returned':relevant/len(actual) if actual else 0,
            'precision_at_3':sum(x in expected for x in actual[:3])/3,
            'recall':len(set(actual)&expected)/len(expected),
            'top1':int(bool(actual) and actual[0] in expected),
            'category':int(ticket.get('category')==case['expected_category']),
            'team':int(ticket.get('assigned_team')==case['expected_team']),
            'priority':int(str(ticket.get('priority','')).lower()==case['expected_priority'].lower()),
            'legacy_groundedness_proxy':int(bool(actual) and not ((result.get('support') or {}).get('fallback_triggered',False) if combined else result.get('fallback_triggered',False))),
            'legacy_faithfulness_proxy':int(not any(p in result.get('answer','').lower() for p in ['guaranteed','definitely resolved','without review','ignore policy']))}

def means(rows):
    return {k:sum(x[k] for x in rows)/len(rows) for k in rows[0]} if rows else {}

def support_comparison():
    baseline=json.loads(Path('/private/tmp/support_baseline_results.json').read_text())
    rows=[]
    def one(row):
        result=c.execute(c.Request(question=row['case']['question']))
        return {'case':row['case'],'baseline_score':score_support(row['case'],row['result']),
                'combined_score':score_support(row['case'],result,True),'combined':result,
                'baseline_ms':row['wall_ms']}
    # Serial runs keep measured latency free from competing support workers.
    for row in baseline:
        rows.append(one(row));save('support_detail.json',rows)
        print('support comparison',len(rows),'/',len(baseline),flush=True)
    return {'n':len(rows),'baseline':means([x['baseline_score'] for x in rows]),'combined':means([x['combined_score'] for x in rows])}

def check_incident(case,report):
    if not report:return {'pass':False,'status':False,'risk':False,'actions':False,'evidence':False}
    actions=' '.join(x['action'] for x in report['recommended_actions']).lower()
    evidence=' '.join(x['value'] for x in report['evidence']).lower()
    checks={'status':report.get('verdict',report.get('status')) in case.get('allowed_verdicts',case.get('allowed_statuses',[])),
            'risk':report['risk_score']>=case['min_risk_score'],
            'actions':all(x.lower() in actions for x in case['required_actions']),
            'evidence':all(x.lower() in evidence for x in case.get('required_evidence',[]))}
    return {'pass':all(checks.values()),**checks}

def incident_comparison():
    rows=[]
    for file in sorted((ROOT/'incident/evals/cases').glob('*.json')):
        case=json.loads(file.read_text())
        security='alert' in case or 'alert_file' in case
        key='alert' if security else 'incident'
        payload=case.get(key) or json.loads((ROOT/'incident'/case[key+'_file']).read_text())
        domain='security' if security else payload['domain']
        baseline=(c.InvestigationGraph().investigate(c.SecurityAlert(**payload)) if security else c.GenericIncidentGraph().investigate(c.IncidentInput(**payload))).model_dump(mode='json')
        request=c.Request(question=payload['title']+' '+payload.get('description',''),severity=payload['severity'],
                          logs=payload.get('logs',[]), events=payload.get('raw_events',payload.get('events',[])),metrics=payload.get('metrics',{}))
        result=c.execute(request)
        rows.append({'name':case['name'],'expected_domain':domain,'baseline':baseline,'combined':result,
                     'baseline_checks':check_incident(case,baseline),'combined_checks':check_incident(case,result.get('investigation')),
                     'domain_correct':result['routing']['domain']==domain,
                     'discarded_input_fields':[k for k in payload if k not in ['title','description','severity','logs','events','metrics','raw_events']]})
        print('incident',len(rows),flush=True)
    save('incident_detail.json',rows)
    return {'n':len(rows),'baseline':means([r['baseline_checks'] for r in rows]),'combined':means([r['combined_checks'] for r in rows]),'domain_accuracy':sum(r['domain_correct'] for r in rows)/len(rows)}

def routing_diagnostic():
    # Hand-authored diagnostic cases, not independent human-reviewed ground truth.
    cases=[
      ('How do I reset my password?','it','support'),('VPN disconnects on my laptop','it','support'),
      ('I changed phones and need Duo setup','it','support'),('MFA push is not arriving','it','support'),
      ('My account is locked','it','support'),('Company-wide authentication failure','it','investigate'),
      ('Several users in finance cannot log in','it','investigate'),('Multiple users cannot access VPN','it','investigate'),
      ('VPN outage for all users','it','investigate'),('Our department cannot access VPN','it','investigate'),
      ('Suspicious login from a new country','security','investigate'),('Account takeover and mailbox rule changes','security','investigate'),
      ('Malware detected on workstation','security','investigate'),('Phishing credentials stolen','security','investigate'),
      ('Security incident affecting company authentication','security','investigate'),('Ransomware encrypted finance files','security','investigate'),
      ('Checkout returns 503 after deployment','production','investigate'),('Production service outage','production','investigate'),
      ('API requests timing out with elevated latency','production','investigate'),('Payment service error rate exceeds SLO','production','investigate'),
      ('AWS autoscaling quota exhausted','cloud','investigate'),('Kubernetes node capacity exhausted','cloud','investigate'),
      ('Cloudwatch reports unavailable instances','cloud','investigate'),('Load balancer targets are unhealthy','cloud','investigate'),
      ('ETL pipeline failed schema drift validation','data','investigate'),('Data quality checks failed','data','investigate'),
      ('Backfill job stopped','data','investigate'),('Warehouse ingestion stopped updating tables','data','investigate'),
      ('Something is broken',None,'clarify'),('Can you help me?',None,'clarify'),
      ('Write me a poem',None,'clarify'),('Recommend a restaurant',None,'clarify'),
      ('No malware was detected; how do I reset my password?','it','support'),
      ('What does phishing mean?',None,'clarify'),
      ('VPN password reset instructions mention AWS but this is just my account','it','support'),
      ('This is not a production outage; I forgot my password','it','support'),
    ]
    rows=[{'question':q,'expected_domain':d,'expected_path':p,'actual':c.route(c.Request(question=q))} for q,d,p in cases]
    save('routing_cases.json',rows)
    tp=sum(r['expected_path']=='investigate' and r['actual']['path']=='investigate' for r in rows)
    fp=sum(r['expected_path']!='investigate' and r['actual']['path']=='investigate' for r in rows)
    fn=sum(r['expected_path']=='investigate' and r['actual']['path']!='investigate' for r in rows)
    labels=['security','production','cloud','data','it',None]
    f1=[]
    for label in labels:
        a=sum(r['expected_domain']==label and r['actual']['domain']==label for r in rows)
        b=sum(r['expected_domain']!=label and r['actual']['domain']==label for r in rows)
        z=sum(r['expected_domain']==label and r['actual']['domain']!=label for r in rows)
        f1.append(2*a/(2*a+b+z) if 2*a+b+z else 0)
    return {'n':len(rows),'domain_accuracy':sum(r['expected_domain']==r['actual']['domain'] for r in rows)/len(rows),
            'domain_macro_f1':sum(f1)/len(f1),'path_accuracy':sum(r['expected_path']==r['actual']['path'] for r in rows)/len(rows),
            'escalation_precision':tp/(tp+fp),'escalation_recall':tp/(tp+fn),
            'unnecessary_escalation_rate':fp/sum(r['expected_path']!='investigate' for r in rows),
            'missed_escalation_rate':fn/(tp+fn),'tp':tp,'fp':fp,'fn':fn}

if __name__=='__main__':
    summary={'mode':'offline','routing_diagnostic':routing_diagnostic()}
    save('summary.json',summary)
    summary['incident_regression']=incident_comparison();save('summary.json',summary)
    summary['support_regression']=support_comparison();save('summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

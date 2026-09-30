import json
import sys
from pathlib import Path
sys.path.insert(0,'/private/tmp')
from measure_combined import ROOT, OUT, c, check_incident, save, means
rows=[]
for file in sorted((ROOT/'incident/evals/cases').glob('*.json')):
    case=json.loads(file.read_text());security='alert' in case or 'alert_file' in case
    key='alert' if security else 'incident'
    payload=case.get(key) or json.loads((ROOT/'incident'/case[key+'_file']).read_text())
    domain='security' if security else payload['domain']
    request=c.Request(question=payload['title']+' '+payload.get('description',''),domain=domain,severity=payload['severity'],logs=payload.get('logs',[]),events=payload.get('raw_events',payload.get('events',[])),metrics=payload.get('metrics',{}))
    result=c.execute(request)
    rows.append({'name':case['name'],'checks':check_incident(case,result.get('investigation')),'result':result})
save('explicit_domain_control.json',{'n':len(rows),'metrics':means([r['checks'] for r in rows]),'cases':rows})
print(json.dumps(means([r['checks'] for r in rows])))

import concurrent.futures
import json
import math
import time
from pathlib import Path
import httpx

OUT=Path('/private/tmp/combined-evaluation')
client=httpx.Client(base_url='http://127.0.0.1:8765',timeout=180)
response=client.post('/auth/login',json={'username':'demo','password':'local-demo-only'})
response.raise_for_status()
client.headers['Authorization']='Bearer '+response.json()['access_token']

def request(payload):
    start=time.perf_counter()
    try:
        response=client.post('/cases',json=payload)
        return {'ms':(time.perf_counter()-start)*1000,'status':response.status_code,'question':payload['question']}
    except Exception as e:
        return {'ms':(time.perf_counter()-start)*1000,'status':0,'error':str(e),'question':payload['question']}

def run(name,payloads,concurrency):
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows=list(pool.map(request,payloads))
    seconds=time.perf_counter()-start
    latencies=sorted(r['ms'] for r in rows)
    output={'workload':name,'concurrency':concurrency,'n':len(rows),'success':sum(r['status']==200 for r in rows),
            'p50_ms':latencies[math.ceil(.5*len(rows))-1],'p95_ms':latencies[math.ceil(.95*len(rows))-1],
            'throughput_requests_per_second':len(rows)/seconds,'wall_seconds':seconds,'requests':rows}
    print(json.dumps({k:v for k,v in output.items() if k!='requests'}),flush=True)
    return output

incident={'question':'Checkout returns 503 after deployment','logs':['checkout HTTP 503']}
support={'question':'How do I reset my password?'}
results=[]
request(incident)
for concurrency in (1,2):
    results.append(run('incident_offline', [incident]*20,concurrency))
    results.append(run('mixed_50_percent_support', [support,incident]*4,concurrency))
    (OUT/'http_performance.json').write_text(json.dumps(results,indent=2))
client.close()

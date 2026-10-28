"""Expose origin without implying that supplied or simulated evidence is verified."""

SIMULATED = {'mcp_tool', 'handoff_context', 'identity', 'identity_signal', 'service_health',
             'cloud_event', 'data_quality', 'user_impact', 'access_impact', 'access_path', 'data_signal', 'security_signal'}
SUPPLIED = {'log','metric','event','raw_event'}

def annotate(report):
    records=[]
    for index,item in enumerate(report.get('evidence',[]),start=1):
        kind=item.get('kind','unknown')
        origin='user_supplied' if kind in SUPPLIED else ('local_simulation' if kind in SIMULATED else 'rule_derived')
        records.append({**item,'evidence_id':f'evidence-{index}','origin':origin,'independently_verified':False})
    report['evidence']=records
    report['evidence_summary']={name:sum(item['origin']==name for item in records) for name in ['user_supplied','local_simulation','rule_derived']}
    report['evidence_summary']['independently_verified']=0
    report['assessment_mode']='offline_rules_and_simulation'
    report['semantic_groundedness']='not_evaluated'
    for finding in report.get('findings',[]):
        finding['verification_status']='unverified_offline_assessment'
    return report


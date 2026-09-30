"""Auditable domain policy. Scores are evidence weights, not probabilities."""
import json
import re
from domain_classifier import predict

PATTERNS = {
    'security': [(r'account takeover|impossible[- ]travel|password[- ]spray|brute[- ]force', 6),
                 (r'malware|ransomware|phishing|breach|security (?:incident|alert)', 5),
                 (r'(?:suspicious|unusual|unauthorized).{0,30}(?:login|oauth|grant|access)|mailbox rule|credentials stolen', 5)],
    'it': [(r'vpn|duo|mfa|sso|authentication|log[ -]?in|sign[ -]?in|password|locked account', 4),
           (r'email|internet|account|printer|laptop|service[- ]desk', 2)],
    'data': [(r'pipeline|etl|backfill|schema|data quality|feature drift|model drift|data freshness|warehouse|ingestion', 5)],
    'cloud': [(r'load balancer|autoscal\w*|kubernetes|cloudwatch|quota|pending pods|node capacity', 5),
              (r'aws|azure|gcp|cloud infrastructure|instance', 2)],
    'production': [(r'checkout|payment service|503|500|queue backlog|worker queue|slo|error rate|latency|timing out|service outage', 4),
                   (r'production|deploy\w*|datadog|grafana', 2)],
}

def active_text(text):
    """Remove bounded negated/background clauses while retaining contrast clauses."""
    parts = re.split(r'[;.!?]|\bbut\b|\bhowever\b', text.lower())
    active = []
    for part in parts:
        # 'Cannot login' is an active failure, not a negated incident.
        if re.search(r'\b(?:no|not|without)\b.{0,25}\b(?:malware|security|production outage|breach|phishing|ransomware)', part):
            continue
        part = re.sub(r'\b(?:instructions|documentation|guide)\s+(?:mentions?|discusses?)\b.*', '', part)
        active.append(part)
    return ' '.join(active)

def route(request):
    question = active_text(request.question)
    evidence = bool(request.logs or request.events or request.raw_events or request.metrics)
    context = active_text(' '.join([request.source or '', request.service or '', ' '.join(request.tags),
                                    request.title or '', request.description or '', ' '.join(request.logs), json.dumps([*request.events, *request.raw_events])]))
    text = question + ' ' + context
    scores, matches = {}, {}
    for domain, patterns in PATTERNS.items():
        matches[domain] = [m.group() for pattern, _ in patterns for m in re.finditer(r'\b(?:' + pattern + r')\b', text)]
        scores[domain] = sum(weight for pattern, weight in patterns if re.search(r'\b(?:' + pattern + r')\b', text))
    definition = bool(re.match(r'\s*(?:what (?:does|is)|define|explain the meaning)', request.question.lower()))
    candidates = sorted(scores, key=scores.get, reverse=True)
    model_prediction = {'available': False, 'accepted': False}
    if request.domain != 'auto':
        domain, mode = request.domain, 'explicit'
    elif definition and not evidence:
        domain, mode = None, 'clarification'
    elif scores['security']:
        domain, mode = 'security', 'security_policy'
    elif scores[candidates[0]] == 0:
        model_prediction = predict(text.strip())
        if model_prediction.get('accepted') and model_prediction.get('domain') != 'other':
            domain, mode = model_prediction['domain'], 'ml_fallback'
        else:
            domain, mode = None, 'clarification'
    elif scores[candidates[0]] == scores[candidates[1]]:
        domain, mode = None, 'ambiguous'
    else:
        domain, mode = candidates[0], 'weighted_policy'
    impact = bool(re.search(r'\b(?:all users|hundreds|company[- ]wide|outage|many (?:users|staff)|multiple (?:users|staff)|several (?:users|staff)|department)\b', question))
    path = 'clarify' if domain is None else ('investigate' if domain != 'it' or evidence or impact or request.severity in ('high','critical') else 'support')
    return {'domain': domain, 'path': path, 'routing_mode': mode, 'scores': scores,
            'matched_signals': matches.get(domain, []),
            'support_context': bool(scores['it']) or domain == 'it',
            'model_prediction': model_prediction,
            'reason': 'Clarify missing or ambiguous domain.' if domain is None else 'Explicit domain or active evidence policy; preserve specialist support context.'}



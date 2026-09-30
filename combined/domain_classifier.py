"""Portable inference for exported sklearn character-TF-IDF logistic regression."""
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

MODEL = Path(__file__).with_name('domain_model.json')

@lru_cache(maxsize=1)
def load_model():
    try:
        return json.loads(MODEL.read_text())
    except (OSError, ValueError):
        return None

def probabilities(text, model):
    text = re.sub(r'\s\s+', ' ', text.lower())
    counts = Counter(text[i:i+n] for n in range(3,6) for i in range(max(0,len(text)-n+1)))
    vocabulary = model['vocabulary']
    features = {vocabulary[t]:(1+math.log(count))*model['idf'][vocabulary[t]] for t,count in counts.items() if t in vocabulary}
    norm = math.sqrt(sum(x*x for x in features.values()))
    if not norm:
        return None
    scores = [bias+sum(weights[i]*value/norm for i,value in features.items()) for bias,weights in zip(model['intercept'],model['coefficients'])]
    exps = [math.exp(s-max(scores)) for s in scores]
    return [x/sum(exps) for x in exps]

def predict(text):
    model = load_model()
    if model is None:
        return {'available':False,'accepted':False}
    probs = probabilities(text,model)
    if probs is None:
        return {'available':True,'accepted':False,'reason':'No known text features'}
    ranked = sorted(range(len(probs)),key=lambda i:probs[i],reverse=True)
    first,second = ranked[:2]
    return {'available':True,'model':'char_tfidf_logistic_regression','domain':model['classes'][first],
            'score':round(probs[first],5),'margin':round(probs[first]-probs[second],5),
            'accepted':probs[first]>=model['threshold'] and probs[first]-probs[second]>=model['margin_threshold'],
            'threshold':model['threshold'],'score_kind':'uncalibrated_model_probability'}

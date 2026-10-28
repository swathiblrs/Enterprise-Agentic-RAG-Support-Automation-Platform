"""Fit on train; select C and acceptance on validation; report test once."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from domain_classifier import probabilities

ROOT=Path(__file__).parent
data=json.loads((ROOT/'domain_dataset.json').read_text())
rows=data['examples']
assert len({r['text'].strip().lower() for r in rows})==len(rows), 'Duplicate examples across splits'
splits={s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
train,valid,test=(splits[s] for s in ['train','validation','test'])
def xy(rows):return [r['text'] for r in rows],[r['domain'] for r in rows]
xt,yt=xy(train);xv,yv=xy(valid);xs,ys=xy(test)
candidates=[]
for strength in [1,5,10]:
    pipe=Pipeline([('tfidf',TfidfVectorizer(analyzer='char',ngram_range=(3,5),sublinear_tf=True)),('lr',LogisticRegression(C=strength,class_weight='balanced',max_iter=2000,random_state=42))])
    pipe.fit(xt,yt)
    score=f1_score(yv,pipe.predict(xv),average='macro')
    candidates.append((score,strength,pipe))
# Select a useful abstaining model on validation only, before testing.
settings=[]
for score,strength,candidate in candidates:
    probs=candidate.predict_proba(xv); pred=candidate.predict(xv)
    for threshold in [.2,.25,.3,.4,.5,.6,.7]:
        for margin in [.05,.1,.15,.2]:
            accepted=(probs.max(axis=1)>=threshold)&((np.sort(probs,axis=1)[:,-1]-np.sort(probs,axis=1)[:,-2])>=margin)
            precision=float(np.mean(pred[accepted]==np.array(yv)[accepted])) if accepted.any() else 0
            if precision>=.9 and accepted.sum()>=6:
                settings.append((int(accepted.sum()),score,precision,strength,threshold,margin,candidate))
if settings:
    _,_,_,strength,threshold,margin,pipe=max(settings,key=lambda x:(x[0],x[1],x[2],-x[3]))
else:
    _,strength,pipe=max(candidates,key=lambda x:(x[0],-x[1]))
    threshold,margin=.7,.2
vectorizer=pipe.named_steps['tfidf'];lr=pipe.named_steps['lr']
model={'version':1,'classes':lr.classes_.tolist(),'vocabulary':vectorizer.vocabulary_, 'idf':vectorizer.idf_.tolist(),
       'coefficients':lr.coef_.tolist(),'intercept':lr.intercept_.tolist(),'threshold':threshold,'margin_threshold':margin,
       'dataset_sha256':hashlib.sha256((ROOT/'domain_dataset.json').read_bytes()).hexdigest()}
model['vocabulary']={k:int(v) for k,v in model['vocabulary'].items()}
# Validate the dependency-free runtime against the training implementation.
for text,expected in zip(xv+xs,pipe.predict_proba(xv+xs)):
    assert np.max(np.abs(np.array(probabilities(text,model))-expected))<1e-9
(ROOT/'domain_model.json').write_text(json.dumps(model,separators=(',',':')))
report={'provenance':data['provenance'],'counts':{k:len(v) for k,v in splits.items()},'selected_C':strength,'threshold':threshold,'margin':margin,'runtime_equivalence_max_tolerance':1e-9,'selection':'validation only','results':{}}
for split,examples in [('validation',valid),('test',test)]:
    texts,labels=xy(examples);pred=pipe.predict(texts);probs=pipe.predict_proba(texts)
    accepted=(probs.max(axis=1)>=threshold)&((np.sort(probs,axis=1)[:,-1]-np.sort(probs,axis=1)[:,-2])>=margin)
    report['results'][split]={'accuracy':accuracy_score(labels,pred),'macro_f1':f1_score(labels,pred,average='macro'),
         'coverage':float(accepted.mean()),'accepted_accuracy':float(np.mean(pred[accepted]==np.array(labels)[accepted])) if accepted.any() else None,
         'cases':[{'text':r['text'],'expected':r['domain'],'predicted':str(p),'accepted':bool(a)} for r,p,a in zip(examples,pred,accepted)]}
(ROOT/'domain_model_evaluation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
print(json.dumps({k:{m:v for m,v in r.items() if m!='cases'} for k,r in report['results'].items()},indent=2))


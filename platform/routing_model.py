"""Use the validation-selected routing model, with explicit legacy fallback."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
from functools import lru_cache

import domain_classifier as legacy

ROOT=Path(__file__).parent
MANIFEST=ROOT/"routing_training"/"active_model.json"


@lru_cache(maxsize=1)
def load_manifest():
    try:
        data=json.loads(MANIFEST.read_text())
        if data.get("schema_version")!=1 or data.get("kind") not in ("tfidf","embedding"):
            return None
        head=(ROOT/data["head"]).resolve()
        head.relative_to(ROOT.resolve())
        if hashlib.sha256(head.read_bytes()).hexdigest()!=data["head_sha256"]:
            return None
        data["_head"]=json.loads(head.read_text())
        return data
    except (OSError,ValueError,KeyError,TypeError):
        return None


class EmbeddingWorker:
    def __init__(self):
        self.process=None
        self.responses=None
        self.lock=threading.Lock()

    def close(self):
        with self.lock:
            self._stop()

    def _stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
        self.process=None

    def call(self,text,manifest):
        if not self.lock.acquire(timeout=60):
            raise RuntimeError("Routing worker is busy")
        try:
            if self.process is None or self.process.poll() is not None:
                self._stop()
                python=ROOT/manifest["python"]
                python.parent.resolve().relative_to(ROOT.resolve())
                self.responses=queue.Queue()
                self.process=subprocess.Popen(
                    [str(python),"-u",str(ROOT/"routing_training"/"serve_model.py"),str(MANIFEST)],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                    text=True,bufsize=1,cwd=ROOT,
                    env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                             TOKENIZERS_PARALLELISM="false"))
                process,responses=self.process,self.responses
                def read():
                    try:
                        for line in process.stdout:responses.put(line)
                    finally:responses.put(None)
                threading.Thread(target=read,daemon=True).start()
            self.process.stdin.write(json.dumps({"text":text})+"\n")
            self.process.stdin.flush()
            line=self.responses.get(timeout=60)
            if line is None:raise RuntimeError("Routing worker exited")
            response=json.loads(line)
            if not response.get("ok"):raise RuntimeError("Routing inference failed")
            return response["probabilities"]
        except (OSError,ValueError,KeyError,queue.Empty,RuntimeError):
            self._stop()
            raise RuntimeError("Selected routing model is unavailable") from None
        finally:
            self.lock.release()


embedding_worker=EmbeddingWorker()
atexit.register(embedding_worker.close)


def predict(text):
    manifest=load_manifest()
    if manifest is None:
        return dict(legacy.predict(text),selection="legacy_fallback_no_usable_manifest")
    head=manifest["_head"]
    try:
        probs=(legacy.probabilities(text,head) if manifest["kind"]=="tfidf"
               else embedding_worker.call(text,manifest))
    except RuntimeError:
        return {"available":False,"accepted":False,"reason":"Selected model worker unavailable"}
    if not probs:
        return {"available":True,"accepted":False,"reason":"No known text features"}
    ranked=sorted(range(len(probs)),key=lambda index:probs[index],reverse=True)
    first,second=ranked[:2]
    return {"available":True,"model":manifest["name"],"domain":head["classes"][first],
            "score":round(probs[first],5),"margin":round(probs[first]-probs[second],5),
            "accepted":probs[first]>=head["threshold"] and probs[first]-probs[second]>=head["margin_threshold"],
            "threshold":head["threshold"],"score_kind":"uncalibrated_model_probability",
            "selection":"validation_selected","dataset_sha256":manifest["dataset_sha256"]}

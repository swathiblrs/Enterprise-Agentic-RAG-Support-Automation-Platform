"""One bounded, restartable worker preserves model state across requests."""
import atexit
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = os.getenv('SUPPORT_PYTHON', sys.executable)

class SupportWorker:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.responses = None

    def _start(self):
        env = dict(os.environ, USE_LLM_GENERATION='false', OPENAI_API_KEY='', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', ITSM_INTEGRATION_MODE='disabled', CHAT_INTEGRATION_MODE='disabled')
        self.responses = queue.Queue()
        self.process = subprocess.Popen([PYTHON, '-u', str(ROOT / 'support_worker.py')], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, cwd=ROOT, env=env)
        process, responses = self.process, self.responses
        def read():
            try:
                for line in process.stdout:
                    responses.put(line)
            finally:
                responses.put(None)
        threading.Thread(target=read, daemon=True).start()

    def _stop(self):
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
        self.process = None

    def call(self, question):
        if not self.lock.acquire(timeout=120):
            raise RuntimeError('Support worker busy')
        try:
            if not self.process or self.process.poll() is not None:
                self._stop()
                self._start()
            self.process.stdin.write(json.dumps({'question': question}) + '\n')
            self.process.stdin.flush()
            line = self.responses.get(timeout=120)
            if line is None:
                raise RuntimeError('Support worker exited')
            response = json.loads(line)
            if not response['ok']:
                raise RuntimeError(response['error'])
            return response['result']
        except (OSError, ValueError, queue.Empty, RuntimeError) as error:
            self._stop()
            raise RuntimeError('Support workflow unavailable') from error
        finally:
            self.lock.release()

    def close(self):
        with self.lock:
            self._stop()

worker = SupportWorker()
atexit.register(worker.close)


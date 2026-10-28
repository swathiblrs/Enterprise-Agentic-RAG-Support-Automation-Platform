"""Launch the API and one independent job worker; stop both process groups on exit."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


def main():
    stop = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:stop.set())
    processes = []
    root = Path(__file__).parent
    env = dict(os.environ)
    env.setdefault("SUPPORT_PYTHON",str(root/".venv-support"/"bin"/"python"))
    commands = [[sys.executable,"-m","uvicorn","api:api","--host","127.0.0.1",
                 "--port",os.getenv("PORT","8765")],
                [sys.executable,str(root/"job_worker.py")]]
    try:
        for command in commands:
            processes.append(subprocess.Popen(command,cwd=root,env=env,start_new_session=True))
        while not stop.wait(.25):
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("API or job worker exited. Restart the local demo.")
    finally:
        # Groups also include each worker's support subprocess.
        for process in processes:
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


if __name__ == "__main__":
    main()

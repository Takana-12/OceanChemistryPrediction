import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from common import ALT_SPLIT_SEED

workers, names = int(sys.argv[1]), sys.argv[2:]
os.makedirs("logs_v3", exist_ok=True)


def run(job):
    name, alt = job
    run_name = name + (f"_alt{ALT_SPLIT_SEED}" if alt else "")
    if os.path.exists(f"runs/{run_name}/meta.json"):
        return
    cmd = [sys.executable, "experiments.py", name] + (["--alt-split"] if alt else [])
    with open(f"logs_v3/{run_name}.log", "w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


with ThreadPoolExecutor(workers) as ex:
    list(ex.map(run, [(n, a) for n in names for a in (False, True)]))
print("done", flush=True)

"""Bounded native evaluation subprocesses with candidate-local logs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def execute(run_dir, candidate_dir, mode, settings):
    run_dir, candidate_dir = Path(run_dir).resolve(), Path(candidate_dir).resolve()
    request = dict(run_dir=str(run_dir), candidate_dir=str(candidate_dir), mode=mode,
                   rtol=settings.rtol, atol=settings.atol, warmup=settings.warmup,
                   trials=settings.trials, result_path=str(candidate_dir / (mode + ".json")))
    request_path = candidate_dir / (mode + "_request.json")
    request_path.write_text(json.dumps(request, indent=2))
    result_path = Path(request["result_path"])
    result_path.unlink(missing_ok=True)
    project = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    metadata = json.loads((run_dir / "reference.json").read_text())
    if metadata["target"] != "cuda":
        from .cpu_execution import apply_cpu_execution
        configuration = metadata.get("cpu_execution")
        if configuration:
            apply_cpu_execution(configuration, environment=env)
        else:
            env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(metadata.get("threads", 1))
    process = subprocess.Popen([sys.executable, "-B", "-m", "astra.eval_worker", str(request_path)],
                               cwd=project, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=settings.execution_timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        (candidate_dir / (mode + ".log")).write_text(output)
        raise RuntimeError(mode + " timed out; worker process group terminated")
    (candidate_dir / (mode + ".log")).write_text(output)
    if process.returncode or not result_path.exists():
        raise RuntimeError(f"{mode} worker exited with code {process.returncode}:\n{output[-12000:]}")
    return json.loads(result_path.read_text())

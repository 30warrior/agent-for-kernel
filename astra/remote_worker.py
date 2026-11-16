"""Offline worker. This module never imports or configures an LLM client."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from types import SimpleNamespace


def inside(base, relative):
    path = (base / relative).resolve()
    if base not in path.parents:
        raise ValueError("Path escapes workspace")
    return path


def pin(request):
    if request["target"] != "cuda":
        allowed = os.sched_getaffinity(0)
        core = request["cpu_core"] if request["cpu_core"] is not None else min(allowed)
        if core not in allowed:
            raise ValueError("cpu-core is outside the remote process's allowed affinity")
        os.sched_setaffinity(0, {core})


def perform(base, request):
    pin(request)
    from .backends import Backend
    backend = Backend(request["target"], request["compile_timeout"])
    reference_dir = base / "reference"
    if request["action"] == "freeze_suite":
        from .reference import restore_reference
        from .test_cases import freeze_suite
        reference = restore_reference(reference_dir)
        return freeze_suite(reference, request["spec"], reference_dir, SimpleNamespace(**request))
    if request["action"] == "prepare":
        from .datasets import load_task
        from .reference import prepare_reference, save_reference
        from .timing import measure
        hardware = backend.capabilities()
        task = load_task(request["dataset"], base / "dataset", request["task_id"])
        if task.source_hash != request["source_sha256"]:
            raise RuntimeError("Uploaded task checksum mismatch")
        reference = prepare_reference(task, request["target"], request["seed"], request["threads"])
        reference.metadata["hardware"] = hardware
        save_reference(reference, reference_dir)
        baseline = [measure(reference.model, case, request["target"], request["warmup"], request["trials"])
                    for case in reference.cases]
        (reference_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))
        return dict(reference=reference.metadata, baseline=baseline)
    candidate = inside(base / "candidates", request["candidate"])
    if request["action"] == "compile":
        return backend.compile(candidate / ("kernel.cu" if request["target"] == "cuda" else "kernel.cpp"), candidate)
    if request["action"] not in ("verify", "benchmark"):
        raise ValueError("Unknown remote action")
    from .execution import execute
    return execute(reference_dir, candidate, request["action"], SimpleNamespace(**request))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    base = Path.cwd().resolve()
    request_path = inside(base, args.request)
    request = json.loads(request_path.read_text())
    result_path = inside(base, request["result_name"])
    if args.child:
        result_path.write_text(json.dumps(dict(ok=True, result=perform(base, request)), indent=2))
        return
    # Clear inherited credentials; even a server with API settings is an offline worker.
    env = {key: value for key, value in os.environ.items()
           if not any(token in key.upper() for token in ("API_KEY", "API_TOKEN"))}
    process = subprocess.Popen([sys.executable, "-B", "-m", "astra.remote_worker", args.request, "--child"],
                               cwd=base, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, start_new_session=True)
    deadline = request["compile_timeout"] if request["action"] == "compile" else request["execution_timeout"]
    response, output = None, ""
    try:
        output, _ = process.communicate(timeout=deadline + 5)
        if process.returncode or not result_path.exists():
            raise RuntimeError(f"Worker exited with code {process.returncode}:\n{output[-24000:]}")
        response = json.loads(result_path.read_text())
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                output, _ = process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
        response = dict(ok=False, error=traceback.format_exc()[-24000:])
    finally:
        log_path = base / (request_path.stem + ".log")
        log_path.write_text(output)
        logs = {request["action"] + "_worker.log": output[-64000:]}
        if "candidate" in request:
            candidate = inside(base / "candidates", request["candidate"])
            for name in ("build.log", "verify.log", "benchmark.log"):
                path = candidate / name
                if path.is_file():
                    logs[name] = path.read_text(errors="replace")[-64000:]
        response["logs"] = logs
        result_path.write_text(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()

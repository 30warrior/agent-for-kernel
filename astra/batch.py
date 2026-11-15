"""Run an explicit manifest serially, preserving failures and declared skips."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from .reports import atomic_write, write_csv


def run_batch(args):
    manifest = json.loads(Path(args.manifest).read_text())
    entries = manifest.get("tasks") if isinstance(manifest, dict) else manifest
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain a nonempty tasks list")
    project = Path(__file__).resolve().parent.parent
    folder = project / "optimization_runs" / ("batch_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True)
    atomic_write(folder / "manifest.json", json.dumps(manifest, indent=2))
    outcomes = []
    for index, entry in enumerate(entries):
        entry = dict(task=entry) if isinstance(entry, str) else entry
        if not isinstance(entry, dict) or not isinstance(entry.get("task"), str):
            raise ValueError("Each manifest entry needs a task path")
        result_path = folder / f"task_{index:04d}.json"
        if entry.get("skip_reason"):
            outcome = dict(task_id=entry["task"], status="skipped", reason=entry["skip_reason"], success=False)
        else:
            command = [sys.executable, "-B", str(project / "kernel_optimizer.py")]
            excluded = {"api_key", "manifest", "task", "result_file", "list_tasks"}
            for key, value in vars(args).items():
                if key in excluded or value is None or value is False:
                    continue
                flag = "--" + key.replace("_", "-")
                command.append(flag)
                if value is not True:
                    command.append(str(value))
            command += ["--task", entry["task"], "--result-file", str(result_path)]
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            if args.api_key:
                env["OPENAI_API_KEY"] = args.api_key
            log_path = folder / f"task_{index:04d}.log"
            print(f"Task {index+1}/{len(entries)}: {entry['task']}", flush=True)
            with log_path.open("w") as stream:
                # Serial workers avoid overlapping benchmark workloads and isolate state.
                result = subprocess.run(command, env=env, cwd=project, stdout=stream, stderr=subprocess.STDOUT)
            if result_path.exists():
                outcome = json.loads(result_path.read_text())
            else:
                outcome = dict(success=False, status="error", error="Task worker produced no result")
            outcome.update(task_id=entry["task"], exit_code=result.returncode, log_path=str(log_path))
        outcomes.append(outcome)
        attempted = [item for item in outcomes if item.get("status") != "skipped"]
        successful = [item for item in attempted if item.get("success")]
        speedups = [item["reference_speedup"] for item in successful]
        n = len(attempted)
        summary = dict(tasks=outcomes, attempted_tasks=n,
                       declared_skips=len(outcomes)-n, successful_tasks=len(successful),
                       compile_rate=sum(bool(item.get("compiled")) for item in attempted)/n if n else None,
                       correctness_rate=sum(bool(item.get("correctness_passed")) for item in attempted)/n if n else None,
                       fast_1=sum(item.get("success",False) and item.get("reference_speedup",0)>1 for item in attempted)/n if n else None,
                       geometric_mean_speedup_successful=math.exp(sum(math.log(value) for value in speedups)/len(speedups)) if speedups else None)
        atomic_write(folder / "batch_results.json", json.dumps(summary, indent=2))
        write_csv(folder / "batch_summary.csv", outcomes,
                  ["task_id","status","success","compiled","correctness_passed","best_version","reference_speedup","refinement_speedup","run_directory","reason","error","log_path"])
    print("Batch reports:", folder, flush=True)
    return 0 if all(item.get("success") or item.get("status")=="skipped" for item in outcomes) else 1

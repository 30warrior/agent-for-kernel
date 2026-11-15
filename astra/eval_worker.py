"""Candidate execution subprocess; crashes cannot take down the agent host."""
import json
from pathlib import Path
import sys


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    from .datasets import load_task
    from .reference import prepare_reference
    from .backends import Backend
    from .evaluation import verify, benchmark
    import torch
    meta = json.loads((Path(request["run_dir"]) / "reference.json").read_text())
    task = load_task(meta["dataset"], meta["dataset_root"], meta["task_id"])
    if task.source_hash != meta["source_sha256"]:
        raise RuntimeError("Task source changed after reference preparation")
    reference = prepare_reference(task, meta["target"], meta["seed"], meta["threads"])
    snapshot = torch.load(Path(request["run_dir"]) / "reference.pt", weights_only=False,
                          map_location="cuda" if meta["target"] == "cuda" else "cpu")
    reference.model.load_state_dict(snapshot["model_state"])
    reference.cases = snapshot["cases"]
    reference.expected = snapshot["expected"]
    reference.mutated_inputs = snapshot["mutated_inputs"]
    folder = Path(request["candidate_dir"])
    artifact = json.loads((folder / "build.json").read_text())
    candidate = Backend(meta["target"]).load_candidate(artifact, folder / "model.py", reference.model)
    result = verify(reference, candidate, meta["target"], request["rtol"], request["atol"])
    if request["mode"] == "benchmark":
        result = benchmark(reference, candidate, meta["target"], result, request["warmup"], request["trials"])
    Path(request["result_path"]).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

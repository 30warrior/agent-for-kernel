"""Candidate execution subprocess; crashes cannot take down the agent host."""
import json
from pathlib import Path
import sys


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    from .reference import restore_reference
    from .backends import Backend
    from .evaluation import verify, benchmark
    reference = restore_reference(request["run_dir"])
    meta = reference.metadata
    folder = Path(request["candidate_dir"])
    artifact = json.loads((folder / "build.json").read_text())
    candidate = Backend(meta["target"]).load_candidate(artifact, folder / "model.py", reference.model)
    result = verify(reference, candidate, meta["target"], request["rtol"], request["atol"])
    if request["mode"] == "benchmark":
        result = benchmark(reference, candidate, meta["target"], result, request["warmup"], request["trials"])
    Path(request["result_path"]).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

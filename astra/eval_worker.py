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
    if request["mode"] == "final_benchmark":
        import copy
        labels = reference.metadata.get("case_labels", [])
        indices = [i for i in range(len(reference.cases))
                   if not labels or labels[i]["origin"] == "dataset"]
        if not indices:
            raise RuntimeError("No original dataset cases available for final evaluation")
        for field in ("cases", "expected", "mutated_inputs"):
            setattr(reference, field, [getattr(reference, field)[i] for i in indices])
        reference.metadata = copy.deepcopy(reference.metadata)
        reference.metadata["case_labels"] = [labels[i] if labels else dict(name=f"dataset_{i}", origin="dataset") for i in indices]
        reference.metadata["dataset_case_count"] = len(indices)
    meta = reference.metadata
    folder = Path(request["candidate_dir"])
    artifact = json.loads((folder / "build.json").read_text())
    candidate = Backend(meta["target"]).load_candidate(artifact, folder / "model.py", reference.model)
    result = verify(reference, candidate, meta["target"], request["rtol"], request["atol"])
    if request["mode"] == "benchmark":
        result = benchmark(reference, candidate, meta["target"], result, request["warmup"], request["trials"])
    if request["mode"] == "final_benchmark":
        from .timing import measure_pair
        result.update(status="failed", case_scope="dataset", baseline=[],
                      protocol="alternating_reference_candidate")
        if result["correctness"] == "passed":
            for record, case in zip(result["cases"], reference.cases):
                ref_timing, timing = measure_pair(copy.deepcopy(reference.model), copy.deepcopy(candidate),
                    case, meta["target"], request["warmup"], request["trials"])
                result["baseline"].append(ref_timing)
                record["benchmark"] = timing
                record["reference_speedup"] = ref_timing["median_ms"] / timing["median_ms"]
            result.update(complete=True, status="passed")
    Path(request["result_path"]).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Candidate execution subprocess; crashes cannot take down the agent host."""
import json
from pathlib import Path
import sys


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    saved = json.loads((Path(request["run_dir"]) / "reference.json").read_text())
    if saved.get("cpu_execution"):
        from .cpu_execution import apply_cpu_execution
        apply_cpu_execution(saved["cpu_execution"])
    from .reference import restore_reference
    from .backends import Backend
    from .evaluation import verify, benchmark
    reference = restore_reference(request["run_dir"])
    if request['mode'] == 'perf_baseline':
        from .cpu_profiling import profile_suite
        if reference.metadata['target'] == 'cuda':
            raise ValueError('CPU perf does not support CUDA targets')
        result = profile_suite(reference, reference.model, Path(request['run_dir']) / 'perf_baseline',
                               request['warmup'], request['perf_repeats'], request['perf_batch_mib'])
        Path(request['result_path']).write_text(json.dumps(result, indent=2))
        return
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
    candidate = Backend(meta["target"], threads=meta.get("threads", 1)).load_candidate(artifact, folder / "model.py", reference.model)
    result = verify(reference, candidate, meta["target"], request["rtol"], request["atol"])
    if request['mode'] == 'perf' and result['correctness'] == 'passed':
        from .cpu_profiling import profile_suite
        if meta['target'] == 'cuda':
            raise ValueError('CPU perf does not support CUDA targets')
        profile = profile_suite(reference, candidate, folder / 'perf', request['warmup'],
                                request['perf_repeats'], request['perf_batch_mib'])
        profile['correctness'] = result['correctness']
        result = profile
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
    if meta.get("cpu_execution"):
        import torch
        from .cpu_execution import runtime_snapshot
        result["cpu_execution"] = runtime_snapshot(torch, meta["cpu_execution"])
    Path(request["result_path"]).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

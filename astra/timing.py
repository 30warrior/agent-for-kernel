"""Measure the same Python model boundary on the selected device."""
import copy
import math
import statistics
import time
import torch
from .reference import clone_inputs


def measure(model, case, target, warmup=5, trials=30, execution=None, case_threads=None):
    if warmup < 0 or trials < 1:
        raise ValueError("warmup >= 0 and trials >= 1 are required")
    state = copy.deepcopy(model.state_dict())
    times = []
    with torch.inference_mode():
        for index in range(warmup + trials):
            if target != "cuda" and execution is not None and case_threads is not None:
                from .cpu_execution import apply_case_execution
                apply_case_execution(execution, case_threads, torch)
            inputs = clone_inputs(case)
            model.load_state_dict(state)
            if target == "cuda":
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                output = model(*inputs)
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end)
            else:
                start = time.perf_counter_ns()
                output = model(*inputs)
                elapsed = (time.perf_counter_ns() - start) / 1e6
            # Retain the output through the timed region, then release it.
            del output
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("Timer returned a non-positive or non-finite duration")
            if index >= warmup:
                times.append(elapsed)
    model.load_state_dict(state)
    ordered = sorted(times)
    return dict(mean_ms=statistics.mean(times), median_ms=statistics.median(times),
                min_ms=min(times), p95_ms=ordered[min(len(ordered)-1, math.ceil(0.95*len(ordered))-1)],
                std_ms=statistics.pstdev(times), samples_ms=times, threads=case_threads,
                boundary="Model.forward including output allocation; excludes input cloning/state reset",
                timing="cuda_event" if target == "cuda" else "perf_counter_ns")


def measure_pair(reference, candidate, case, target, warmup=5, trials=30, execution=None, case_threads=None):
    """Alternate reference/candidate order; keep preparation outside each timed forward."""
    if warmup < 0 or trials < 1:
        raise ValueError("warmup >= 0 and trials >= 1 are required")
    samples = [[], []]
    models = [reference, candidate]
    for index in range(warmup + trials):
        for side in ((0, 1) if index % 2 == 0 else (1, 0)):
            value = measure(models[side], case, target, warmup=0, trials=1,
                            execution=execution, case_threads=case_threads)["samples_ms"][0]
            if index >= warmup:
                samples[side].append(value)
    results = []
    for values in samples:
        ordered = sorted(values)
        results.append(dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
            min_ms=min(values), p95_ms=ordered[min(len(values)-1, math.ceil(0.95*len(values))-1)],
            std_ms=statistics.pstdev(values), samples_ms=values, threads=case_threads,
            boundary="Model.forward including output allocation; excludes input cloning/state reset",
            timing="cuda_event" if target == "cuda" else "perf_counter_ns",
            protocol="alternating_reference_candidate"))
    return results

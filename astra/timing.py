"""Measure the same Python model boundary on the selected device."""
import copy
import math
import statistics
import time
import torch
from .reference import clone_inputs


def measure(model, case, target, warmup=5, trials=30):
    if warmup < 0 or trials < 1:
        raise ValueError("warmup >= 0 and trials >= 1 are required")
    state = copy.deepcopy(model.state_dict())
    times = []
    with torch.inference_mode():
        for index in range(warmup + trials):
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
                std_ms=statistics.pstdev(times), samples_ms=times,
                boundary="Model.forward including output allocation; excludes input cloning/state reset",
                timing="cuda_event" if target == "cuda" else "perf_counter_ns")

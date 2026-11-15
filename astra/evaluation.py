"""Compare task outputs and input mutations before measuring a candidate."""
import copy
import math
import traceback
import torch
from .reference import clone_inputs
from .timing import measure


def compare(expected, actual, path="output", rtol=None, atol=None):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(path + ": expected Tensor")
        if expected.shape != actual.shape or expected.dtype != actual.dtype or expected.device != actual.device:
            raise AssertionError(path + ": shape/dtype/device mismatch")
        if expected.is_floating_point() or expected.is_complex():
            default = (2e-2, 1e-2) if expected.dtype == torch.bfloat16 else (1e-2, 5e-3) if expected.dtype == torch.float16 else (1e-4, 1e-5)
            relative = default[0] if rtol is None else rtol
            absolute = default[1] if atol is None else atol
            if not torch.allclose(expected, actual, rtol=relative, atol=absolute, equal_nan=True):
                difference = (expected - actual).abs()
                finite = difference[torch.isfinite(difference)]
                maximum = finite.max().item() if finite.numel() else None
                raise AssertionError(f"{path}: numeric mismatch, max_finite_abs_diff={maximum}, rtol={relative}, atol={absolute}")
        elif not torch.equal(expected, actual):
            raise AssertionError(path + ": exact comparison failed")
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(expected) != len(actual):
            raise AssertionError(path + ": sequence type/length mismatch")
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{path}[{index}]", rtol, atol)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError(path + ": dictionary keys mismatch")
        for key in expected:
            compare(expected[key], actual[key], f"{path}[{key!r}]", rtol, atol)
    elif type(expected) is not type(actual):
        raise AssertionError(path + ": scalar type mismatch")
    elif isinstance(expected, float):
        if not ((math.isnan(expected) and math.isnan(actual)) or math.isclose(expected, actual, rel_tol=1e-4 if rtol is None else rtol, abs_tol=1e-5 if atol is None else atol)):
            raise AssertionError(path + ": scalar value mismatch")
    elif expected != actual:
        raise AssertionError(path + ": value mismatch")


def verify(reference, candidate, target, rtol=None, atol=None):
    records = []
    for index, case in enumerate(reference.cases):
        record = dict(case_id=index, correctness="failed", benchmark=None)
        try:
            inputs = clone_inputs(case)
            model = copy.deepcopy(candidate)
            with torch.inference_mode():
                output = model(*inputs)
            if target == "cuda":
                torch.cuda.synchronize()
            compare(reference.expected[index], output, rtol=rtol, atol=atol)
            compare(reference.mutated_inputs[index], inputs, path="inputs_after_forward", rtol=rtol, atol=atol)
            record["correctness"] = "passed"
        except Exception:
            record["error"] = traceback.format_exc()[-12000:]
        records.append(record)
    passed = bool(records) and all(record["correctness"] == "passed" for record in records)
    return dict(correctness="passed" if passed else "failed", complete=False, cases=records)


def benchmark(reference, candidate, target, verification, warmup=5, trials=30):
    result = copy.deepcopy(verification)
    if result["correctness"] != "passed":
        return result
    for record, case in zip(result["cases"], reference.cases):
        try:
            record["benchmark"] = measure(copy.deepcopy(candidate), case, target, warmup, trials)
        except Exception:
            record["error"] = traceback.format_exc()[-12000:]
    result["complete"] = bool(result["cases"]) and all(record["benchmark"] is not None for record in result["cases"])
    return result


def evaluate(reference, candidate, target, warmup=5, trials=30, rtol=None, atol=None):
    return benchmark(reference, candidate, target,
                     verify(reference, candidate, target, rtol, atol), warmup, trials)

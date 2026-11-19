"""Prepare a reproducible, device-specific PyTorch reference for a dataset task."""
import copy
import hashlib
import importlib.util
import random
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import torch


def map_tree(value, tensor_fn):
    if isinstance(value, torch.Tensor):
        return tensor_fn(value)
    if isinstance(value, tuple):
        return tuple(map_tree(item, tensor_fn) for item in value)
    if isinstance(value, list):
        return [map_tree(item, tensor_fn) for item in value]
    if isinstance(value, dict):
        return {key: map_tree(item, tensor_fn) for key, item in value.items()}
    return copy.deepcopy(value)


def clone_inputs(inputs):
    return map_tree(inputs, lambda tensor: tensor.detach().clone(memory_format=torch.preserve_format))


def describe(value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype), device=str(value.device))
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    return dict(type=type(value).__name__, value=repr(value))


@contextmanager
def task_import_directory(path):
    previous = list(sys.path)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = previous


@dataclass
class Reference:
    model: object
    cases: list
    expected: list
    mutated_inputs: list
    metadata: dict


def prepare_reference(task, target, seed=42, threads=1, cpu_execution=None, thread_plan=None):
    if threads < 1:
        raise ValueError("threads must be positive")
    if target == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA target requires an available CUDA device")
    device = torch.device("cuda" if target == "cuda" else "cpu")
    from .cpu_execution import configure_torch_threads, runtime_snapshot, apply_cpu_execution
    if target == "cuda":
        torch.set_num_threads(threads)
    else:
        configure_torch_threads(torch, threads)
    random.seed(seed)
    torch.manual_seed(seed)
    # Prevent implicit allocations from selecting an NPU/GPU on CPU runs.
    with task_import_directory(task.path.parent), torch.device(device):
        name = "astra_task_" + hashlib.sha256(str(task.path).encode()).hexdigest()[:16]
        spec = importlib.util.spec_from_file_location(name, task.path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        init = module.get_init_inputs()
        if not isinstance(init, (list, tuple)):
            raise ValueError("get_init_inputs must return a list or tuple")
        model = module.Model(*init).to(device).eval()
        raw = module.get_inputs_dyn_list() if task.dynamic else [module.get_inputs()]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("Task must supply at least one input case")
        cases = []
        for case in raw:
            if not isinstance(case, (list, tuple)):
                raise ValueError("Each case must be a list/tuple of positional arguments")
            cases.append(map_tree(case, lambda tensor: tensor.detach().to(device)))
    from .cpu_execution import apply_case_execution, build_thread_plan
    if target != "cuda" and cpu_execution is not None:
        thread_plan = thread_plan or build_thread_plan(cases, cpu_execution["threads"])
    expected, mutated = [], []
    for index, case in enumerate(cases):
        if target != "cuda" and cpu_execution is not None:
            apply_case_execution(cpu_execution, (thread_plan or [])[index]["threads"], torch)
        inputs = clone_inputs(case)
        with torch.inference_mode():
            output = copy.deepcopy(model)(*inputs)
        expected.append(map_tree(output, lambda tensor: tensor.detach().clone()))
        mutated.append(clone_inputs(inputs))
    if target != "cuda" and cpu_execution is not None:
        apply_cpu_execution(cpu_execution)
        configure_torch_threads(torch, cpu_execution["threads"])
    metadata = dict(task.metadata(), target=target, seed=seed, threads=threads,
                    mode="eval/inference", torch_version=torch.__version__,
                    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    init_inputs=describe(init), cases=[describe(case) for case in cases],
                    outputs=[describe(output) for output in expected])
    if thread_plan is not None:
        metadata["thread_plan"] = thread_plan
    if cpu_execution is not None:
        metadata["affinity"] = list(cpu_execution["affinity"])
        metadata["cpu_execution"] = runtime_snapshot(torch, cpu_execution)
    return Reference(model, cases, expected, mutated, metadata)


def save_reference(reference, directory):
    import json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "reference.json").write_text(json.dumps(reference.metadata, indent=2))
    torch.save(dict(model_state=reference.model.state_dict(), cases=reference.cases,
                    expected=reference.expected, mutated_inputs=reference.mutated_inputs),
               directory / "reference.pt")


def restore_reference(directory):
    """Reconstruct the reference model and load the fixed input/output snapshot."""
    import json
    from .datasets import load_task
    directory = Path(directory)
    meta = json.loads((directory / "reference.json").read_text(encoding="utf-8"))
    task = load_task(meta["dataset"], meta["dataset_root"], meta["task_id"])
    if task.source_hash != meta["source_sha256"]:
        raise RuntimeError("Task source changed after reference preparation")
    reference = prepare_reference(task, meta["target"], meta["seed"], meta["threads"],
                                 cpu_execution=meta.get("cpu_execution"),
                                 thread_plan=meta.get("thread_plan"))
    snapshot = torch.load(directory / "reference.pt", weights_only=False,
                          map_location="cuda" if meta["target"] == "cuda" else "cpu")
    reference.model.load_state_dict(snapshot["model_state"])
    reference.cases, reference.expected = snapshot["cases"], snapshot["expected"]
    reference.mutated_inputs = snapshot["mutated_inputs"]
    reference.metadata = meta
    return reference

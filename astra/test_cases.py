"""Materialize an agent-authored JSON suite once; reuse its reference snapshot."""
import copy
import hashlib
import json
import math
from pathlib import Path
import random


SPEC_GUIDE = """Submit JSON with cases and coverage lists through generate_comprehensive_test_cases(test_spec).
Each case: {"name":"unique name", "base_case":0, "seed":43, "inputs":[...]}.
base_case indexes the ORIGINAL dataset cases. inputs describes ALL positional arguments.
Tensor descriptor: {"kind":"tensor", "shape":[2,33], "distribution":"normal", "mean":0.0, "std":1.0}.
Other distributions: uniform (low/high), randint (integer low/high, high exclusive), zeros, ones.
Use randint low=0 high=2 for bool masks. Dtype and device stay the same as the base input.
Optional transpose:[axis1,axis2] is applied AFTER allocating shape; use it for legal noncontiguous cases.
The descriptor shape is the PRE-transpose shape. Before the reference model is called, the tool
computes the FINAL shape after transpose and rejects malformed axes or fixed-axis size changes.
For example, shape [8192,12] with transpose [0,1] produces final shape [12,8192].
This is a generic structural check; the reference model remains the authority for operator-specific
contracts such as normalized dimensions, channel counts and reduction legality.
Copy descriptor: {"kind":"copy"} clones the original argument, including nested structures.
Scalar descriptor: {"kind":"scalar", "value":1.0} preserves the original Python scalar type.
Do not change model weights, constructor arguments, computation semantics or original dataset cases.
Keep coupled dimensions/broadcasting valid; class labels must stay in range, attention masks must be legal.
For constrained parameterized models vary legal batch/spatial sizes or just input values, not fixed channel dimensions.
Cover mixed signs and tails only where the reference contract permits them. Do not insert NaNs or infinities.
All proposed cases must run successfully on the reference model before freezing. Invalid cases return errors for repair.
The suite is generated ONCE before candidate evaluation; never add/remove cases in response to candidate failures.
Only propose additional input tensors within the supplied total byte budget; the original dataset is retained unchanged.
Submit exactly 15 additional cases, even when boundary/layout coverage is partial.
Once a valid reference suite of exactly 15 cases is materialized, it is frozen.
Partial coverage does not trigger a repair loop or a missing-items report, under either profile.
coverage must declare EVERY axis listed in coverage_requirements.axes, using the input's FINAL axes AFTER transpose:
{"base_case":0,"input":0,"axis":1,"kind":"variable","reason":"Source reduces this resizable dimension"}.
For a dimension fixed by weights/constructor/semantics use kind:"fixed" with a specific source-based reason.
Follow coverage_requirements.profile. In compact mode, cover each supplied boundary length across
the legal layouts JOINTLY, not a length-by-layout Cartesian product. Include the multi-vector
remainder in that union and include every legal layout at least once: contiguous, unit_stride
(stride[axis]==1), and strided (stride[axis]>1). Full mode requires every boundary length in every layout.
For a middle reduction axis, transpose to make that
axis unit-stride, not just an unrelated axis. Use small other dimensions and preserve coupled dimensions.
Strided coverage at length 1 is not required: singleton axes have no stride-dependent traversal.
One case may cover multiple axes/layouts. Reuse cases across inputs/axes and combine value variation
with shape/layout checks. Both profiles require exactly fifteen additional cases.
Where the budget permits, reserve one case with original input shapes and varied nonzero values
so generated-case performance selection also samples the original workload scale.
Nonzero-capable distributions must supply boundary coverage; zero-only cases do not count.
Include ones to expose missing terms, and mixed-sign inputs where legal to check accuracy.
Only when required by the REFERENCE contract, a variable entry may supply minimum/maximum/multiple
(positive integers), or layout_exemptions:{"strided":"Source requires contiguous ..."}; explain these
restrictions in reason. A layout structurally impossible with the input descriptor and fixed dimensions
may also be exempted with that specific reason. Never exempt a representable legal layout or mark
resizable axes fixed to avoid testing.
When no SVE VL is readable, supplied generic sizes are diagnostic coverage, not an assumed hardware VL.
The tool checks actual shapes and strides before freezing. This samples vector and common unroll
boundaries when possible; compact coverage may be partial under the fixed case cap and does not
prove arbitrary candidate tile/loop coverage. The tool reports only partial status, not a list
of untested items.
"""


def validate_spec(spec, count, maximum=15):
    if not isinstance(spec, dict) or set(spec) != {"cases", "coverage"}:
        raise ValueError("Test specification must contain cases and coverage lists; legacy specs need a coverage declaration")
    cases = spec["cases"]
    if not isinstance(cases, list):
        raise ValueError('cases must be a list')
    if len(cases) < count or len(cases) > maximum:
        raise ValueError(f"Expected {count} to {maximum} additional test cases")
    if not isinstance(spec['coverage'], list):
        raise ValueError('coverage must be a list of axis declarations')
    names, signatures = set(), set()
    for case in cases:
        if not isinstance(case, dict) or set(case) - {"name", "base_case", "seed", "inputs"}:
            raise ValueError("Invalid test case fields")
        name = case.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Each case needs a unique nonempty name")
        names.add(name)
        for field in ("seed", "base_case"):
            value = case.get(field)
            if type(value) is not int or value < 0 or value >= 2**32:
                raise ValueError(field + " must be a nonnegative 32-bit integer")
        if not isinstance(case.get("inputs"), list):
            raise ValueError("inputs must be a descriptor list")
        signature = json.dumps({key: value for key, value in case.items() if key != "name"}, sort_keys=True)
        if signature in signatures:
            raise ValueError("Duplicate test specification")
        signatures.add(signature)
    # Enforce JSON-compatible finite scalar values before remote transport/materialization.
    json.dumps(spec, allow_nan=False)
    return spec


def suite_limits(settings):
    return 15, 15


def build_input(base, descriptor, remaining):
    import torch
    from .reference import map_tree
    if not isinstance(descriptor, dict):
        raise ValueError("Input descriptor must be an object")
    kind = descriptor.get("kind")
    if kind == "copy":
        if set(descriptor) != {"kind"}:
            raise ValueError("copy accepts only kind")
        def clone(tensor):
            remaining[0] -= tensor.numel() * tensor.element_size()
            if remaining[0] < 0:
                raise ValueError("Additional test inputs exceed test-suite-max-mib")
            return tensor.detach().clone(memory_format=torch.preserve_format)
        return map_tree(base, clone)
    if kind == "scalar":
        if set(descriptor) != {"kind", "value"} or type(descriptor["value"]) is not type(base):
            raise ValueError("Scalar descriptor must preserve the original scalar type")
        if not isinstance(base, (int, float, bool, str)):
            raise ValueError("Unsupported scalar input")
        return descriptor["value"]
    allowed = {"kind", "shape", "distribution", "low", "high", "mean", "std", "transpose"}
    if kind != "tensor" or not isinstance(base, torch.Tensor) or set(descriptor) - allowed:
        raise ValueError("Invalid tensor descriptor or argument type")
    shape = descriptor.get("shape")
    if not isinstance(shape, list) or len(shape) > 16 or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("shape must be a list of nonnegative integers")
    remaining[0] -= math.prod(shape) * base.element_size()
    if remaining[0] < 0:
        raise ValueError("Additional test inputs exceed test-suite-max-mib")
    distribution = descriptor.get("distribution")
    options = dict(dtype=base.dtype, device=base.device)
    if distribution in ("zeros", "ones"):
        tensor = getattr(torch, distribution)(shape, **options)
    elif distribution == "normal" and base.is_floating_point():
        std, mean = descriptor.get("std", 1.0), descriptor.get("mean", 0.0)
        if std < 0:
            raise ValueError("normal std must be nonnegative")
        tensor = torch.randn(shape, **options).mul_(std).add_(mean)
    elif distribution == "uniform" and base.is_floating_point():
        low, high = descriptor.get("low", 0.0), descriptor.get("high", 1.0)
        if not low < high:
            raise ValueError("uniform requires low < high")
        tensor = torch.rand(shape, **options).mul_(high-low).add_(low)
    elif distribution == "randint" and not (base.is_floating_point() or base.is_complex()):
        low, high = descriptor.get("low"), descriptor.get("high")
        if type(low) is not int or type(high) is not int or not low < high:
            raise ValueError("randint requires integer low < high")
        tensor = torch.randint(low, high, shape, **options)
    else:
        raise ValueError("Distribution incompatible with original dtype")
    if "transpose" in descriptor:
        axes = descriptor["transpose"]
        if not isinstance(axes, list) or len(axes) != 2 or any(type(n) is not int for n in axes):
            raise ValueError("transpose must contain two axes")
        tensor = tensor.transpose(*axes)
    return tensor


def _preflight_descriptor(base, descriptor, path, fixed_axes):
    """Check descriptor structure and final shape without allocating or invoking the model."""
    if not isinstance(descriptor, dict):
        raise ValueError(f"{path}: input descriptor must be an object")
    kind = descriptor.get("kind")
    if kind == "copy":
        if set(descriptor) != {"kind"}:
            raise ValueError(f"{path}: copy accepts only kind")
        return
    if kind == "scalar":
        if set(descriptor) != {"kind", "value"} or type(descriptor.get("value")) is not type(base):
            raise ValueError(f"{path}: scalar descriptor must preserve the original scalar type")
        if not isinstance(base, (int, float, bool, str)):
            raise ValueError(f"{path}: unsupported scalar input")
        return

    allowed = {"kind", "shape", "distribution", "low", "high", "mean", "std", "transpose"}
    if kind != "tensor" or not hasattr(base, "shape") or not hasattr(base, "dtype"):
        raise ValueError(f"{path}: invalid tensor descriptor or argument type")
    if set(descriptor) - allowed:
        raise ValueError(f"{path}: unknown tensor descriptor field")
    shape = descriptor.get("shape")
    if not isinstance(shape, list) or len(shape) > 16 or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError(f"{path}: shape must be a list of nonnegative integers")
    axes = descriptor.get("transpose")
    final_shape = list(shape)
    if axes is not None:
        if not isinstance(axes, list) or len(axes) != 2 or any(type(n) is not int for n in axes):
            raise ValueError(f"{path}: transpose must contain two axes")
        if axes[0] == axes[1] or any(axis < 0 or axis >= len(shape) for axis in axes):
            raise ValueError(f"{path}: transpose axes must be distinct and within the pre-transpose rank")
        final_shape[axes[0]], final_shape[axes[1]] = final_shape[axes[1]], final_shape[axes[0]]
    for axis in fixed_axes:
        if axis < 0 or axis >= len(final_shape):
            raise ValueError(f"{path}: fixed coverage axis {axis} is outside final rank {len(final_shape)}")
        original_shape = list(base.shape)
        if axis >= len(original_shape) or final_shape[axis] != original_shape[axis]:
            raise ValueError(
                f"{path}: final shape {final_shape} changes fixed axis {axis}; "
                "choose a pre-transpose shape that preserves the reference contract"
            )


def preflight_spec(reference, spec):
    """Validate all descriptors before any generated input reaches the reference model."""
    original_count = len(reference.cases)
    fixed_axes = {}
    for declaration in spec.get("coverage", []):
        if not isinstance(declaration, dict) or declaration.get("kind") != "fixed":
            continue
        selectors = tuple(declaration.get(key) for key in ("base_case", "input", "axis"))
        if all(type(value) is int for value in selectors):
            fixed_axes.setdefault(selectors[:2], set()).add(selectors[2])
    for index, case in enumerate(spec["cases"]):
        base_case = case["base_case"]
        if base_case >= original_count:
            raise ValueError(f"Generated case {index}: base_case is outside original dataset cases")
        base = reference.cases[base_case]
        if len(case["inputs"]) != len(base):
            raise ValueError(f"Generated case {index}: wrong number of positional input descriptors")
        if all(isinstance(item, dict) and item.get("kind") == "copy" for item in case["inputs"]):
            raise ValueError(f"Generated case {index}: a generated case must vary at least one argument")
        for input_index, (value, descriptor) in enumerate(zip(base, case["inputs"])):
            _preflight_descriptor(value, descriptor, f"case {index} input {input_index}",
                                  fixed_axes.get((base_case, input_index), set()))


def freeze_suite(reference, spec, folder, settings):
    """Commit only after every proposed case validates and all baselines measure."""
    import torch
    from .reference import Reference, clone_inputs, describe, map_tree, save_reference
    from .timing import measure
    from .reports import atomic_write
    spec = validate_spec(spec, *suite_limits(settings))
    folder = Path(folder)
    frozen = folder / "test_suite.json"
    if frozen.exists():
        raise RuntimeError("Test suite is frozen; it cannot be regenerated during iterations")
    if reference.metadata.get("suite_id"):
        raise RuntimeError("Reference already contains a frozen suite")
    # Reject malformed descriptors and impossible fixed-axis layouts before any
    # generated input reaches the reference model.
    preflight_spec(reference, spec)
    original_count = len(reference.cases)
    cases, expected, mutations, labels = [], [], [], []
    remaining = [int(settings.test_suite_max_mib * 2**20)]
    for index, case in enumerate(spec["cases"]):
        try:
            if case["base_case"] >= original_count:
                raise ValueError("base_case is outside original dataset cases")
            base = reference.cases[case["base_case"]]
            if len(case["inputs"]) != len(base):
                raise ValueError("Wrong number of positional input descriptors")
            if all(item.get("kind") == "copy" for item in case["inputs"]):
                raise ValueError("A generated case must vary at least one argument")
            random.seed(case["seed"])
            torch.manual_seed(case["seed"])
            inputs = type(base)(build_input(value, desc, remaining) for value, desc in zip(base, case["inputs"]))
            if settings.target != 'cuda' and reference.metadata.get('cpu_execution'):
                from .cpu_execution import apply_case_execution, build_thread_plan
                execution = reference.metadata['cpu_execution']
                apply_case_execution(execution, build_thread_plan([inputs], execution['threads'])[0]['threads'], torch)
            working = clone_inputs(inputs)
            with torch.inference_mode():
                output = copy.deepcopy(reference.model)(*working)
            # A finite test specification should not silently produce invalid reference expectations.
            def check_finite(tensor):
                if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
                    raise ValueError("Reference produced non-finite outputs; repair input domain")
                return tensor.detach().clone()
            expected.append(map_tree(output, check_finite))
            cases.append(inputs)
            mutations.append(clone_inputs(working))
            labels.append(dict(name=case["name"], origin="generated", seed=case["seed"]))
        except Exception as error:
            raise ValueError(f"Generated case {index} ({case['name']}) failed reference validation: {error}") from error
    from .coverage import check_coverage
    coverage = check_coverage(spec, cases, reference.metadata, getattr(settings, 'test_coverage', 'compact'))
    metadata = copy.deepcopy(reference.metadata)
    metadata['boundary_coverage'] = coverage
    identity = dict(spec=spec, source_sha256=metadata["source_sha256"], seed=metadata["seed"])
    metadata["suite_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    metadata["dataset_case_count"] = original_count
    metadata["case_labels"] = [dict(name=f"dataset_{i}", origin="dataset") for i in range(original_count)] + labels
    combined = Reference(reference.model, reference.cases + cases, reference.expected + expected,
                         reference.mutated_inputs + mutations, metadata)
    combined.metadata["cases"] = [describe(case) for case in combined.cases]
    combined.metadata["outputs"] = [describe(output) for output in combined.expected]
    if settings.target != 'cuda' and metadata.get('cpu_execution'):
        from .cpu_execution import build_thread_plan
        metadata['thread_plan'] = build_thread_plan(combined.cases, metadata['cpu_execution']['threads'])
    baseline = [measure(combined.model, case, settings.target, settings.warmup, settings.trials,
                        execution=metadata.get('cpu_execution'),
                        case_threads=metadata.get('thread_plan', [])[index]['threads'] if metadata.get('thread_plan') else None)
                for index, case in enumerate(combined.cases)]
    save_reference(combined, folder)
    atomic_write(folder / "baseline.json", json.dumps(baseline, indent=2))
    atomic_write(frozen, json.dumps(dict(spec=spec, suite_id=metadata["suite_id"],
        dataset_case_count=original_count, cases=metadata["case_labels"], boundary_coverage=coverage), indent=2))
    atomic_write(folder / 'coverage.json', json.dumps(coverage, indent=2))
    # Update the caller only once all reference execution and timing have succeeded.
    reference.cases, reference.expected = combined.cases, combined.expected
    reference.mutated_inputs, reference.metadata = combined.mutated_inputs, combined.metadata
    return dict(reference=metadata, baseline=baseline)


def generate_suite(task, reference, folder, settings, llm, materialize):
    from agents import Agent, Runner, function_tool
    from .prompts import TESTER, execution_context, public_metadata
    folder = Path(folder)
    if (folder / "test_suite.json").exists():
        raise RuntimeError("Suite generation was already completed for this run")
    outcome = {}

    @function_tool
    def generate_comprehensive_test_cases(test_spec: str) -> str:
        """Validate a complete JSON test specification on the reference, then freeze it for all versions."""
        if outcome:
            return "Suite already frozen; reuse it unchanged for every version."
        try:
            spec = validate_spec(json.loads(test_spec), *suite_limits(settings))
            result = materialize(spec)
            outcome.update(result)
            return json.dumps(dict(status="frozen", case_count=len(result["reference"]["cases"]),
                                   cases=result["reference"]["case_labels"]))
        except Exception as error:
            return "ERROR: " + str(error)

    from .coverage import coverage_requirements
    instructions = TESTER + "\n" + SPEC_GUIDE
    prompt = dict(action="generate_and_freeze_test_suite", task=public_metadata(task.metadata()),
                  reference_source=task.source, execution=execution_context(reference.metadata, settings),
                  minimum_additional_cases=suite_limits(settings)[0],
                  maximum_additional_cases=suite_limits(settings)[1],
                  coverage_requirements=coverage_requirements(reference.metadata, getattr(settings, 'test_coverage', 'compact')),
                  total_additional_input_bytes_limit=int(settings.test_suite_max_mib * 2**20))
    (folder / "test_generation_prompt.json").write_text(json.dumps(prompt, indent=2), encoding="utf-8")
    (folder / "test_generation_instructions.txt").write_text(instructions, encoding="utf-8")
    agent = Agent(name="TestingAgent", model=llm.model, instructions=instructions,
                  tools=[generate_comprehensive_test_cases])
    try:
        result = Runner.run_sync(agent, json.dumps(prompt), max_turns=llm.max_turns)
        (folder / "test_generation_response.txt").write_text(str(result.final_output), encoding="utf-8")
    except Exception as error:
        (folder / "test_generation_error.txt").write_text(str(error), encoding="utf-8")
        if not outcome:
            raise
    if not outcome:
        raise RuntimeError("TestingAgent did not freeze a valid test suite; no candidates will be evaluated")
    return outcome

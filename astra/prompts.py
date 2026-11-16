"""Shared role contracts with explicit, editable target optimization guidance."""
import copy
from pathlib import Path


CUDA_OPTIMIZATION = """Prefer fast math, coalesced access, vectorization (float2/float4) and grid-stride loops.

Suggestions:
- Fuse Kernels When Possible
- Use Vectorized Access with Alignment Checks
- Warp-Level Reductions Instead of Shared Memory Loops
- Leverage FMA (Fused Multiply-Add)
"""

CUDA_STRATEGY = """You are an expert CUDA optimization strategist. Analyze performance data and provide specific, implementable suggestions only (no code).

Focus:
- Memory access (coalescing, vectorization, shared memory)
- Thread/block sizing and occupancy
- Fast math, instruction-level fusion, loop unrolling
- Architecture-specific knobs when safe

Constraints:
- Keep the current __global__ signature and functionality intact.
- Output only suggestions; do not include code snippets.
- Be concrete and actionable.
"""

TARGET_GUIDANCE = {
    "cuda": "Generate CUDA C++ (.cu) with at least one __global__ compute kernel. Launch on PyTorch's current CUDA stream; check shapes/dtypes and handle tails.\n",
    "neon": "Generate AArch64 C++ with <arm_neon.h> NEON intrinsics. Handle fixed-width vector tails and unaligned inputs safely.\n",
    "sve": "Generate AArch64 C++ with <arm_sve.h> SVE intrinsics. Use predicates and runtime lane counts; do not assume a fixed vector length.\n",
}

ORCHESTRATOR = """You are the Orchestrator Agent responsible for coordinating the kernel optimization process.
Your role is to receive workflow notifications and track optimization progress.
Keep responses concise and focused on coordination tasks. The application owns scheduling and state updates.
"""

TESTER = """You handle test case generation and correctness verification with unified test data.
Generate the extra test suite ONCE, before candidate evaluation, and reuse it for ALL versions.
Keep every original dataset case. Never change model initialization, weights, task semantics or the frozen suite.
When generating tests, call generate_comprehensive_test_cases with a complete JSON specification.
Choose practical small/medium cases, mixed signs, zeros and ones, broadcasting/strides and vector tails where legal.
Respect the reference source: parameterized layers may constrain channels, normalized shapes, dtypes and label ranges.
Use the reference model to validate cases through the tool; repair invalid specifications rather than discarding cases.
During each version's evaluation, call verify_candidate. Do not invent results or generate new tests.
The same fixed inputs are sent to the profiling agent after correctness passes; failed candidates are never timed.
"""

PROFILER = """You are the Profiling Agent responsible for measuring each kernel version on the unified test suite.
Call benchmark_candidate to obtain actual performance data, rechecking correctness first.
Use the SAME frozen cases across ALL versions. Never generate new cases or synthesize timings.
Report measured case latencies and speedups relative to the corresponding reference baselines.
Version selection uses the configured arithmetic/geometric mean speedup on generated cases.
The application independently remeasures the frozen winner on original dataset cases after all iterations.
"""


def optimization_guidance(target):
    if target == "cuda":
        return CUDA_OPTIMIZATION
    return (Path(__file__).resolve().parent.parent / "prompts" / ("cpu_" + target + ".md")).read_text(encoding="utf-8")


def strategy_instructions(target):
    if target == "cuda":
        instructions = CUDA_STRATEGY
    else:
        instructions = ("You are an expert Arm CPU kernel optimization strategist. Give concrete, implementable suggestions only (no code).\n"
                        + TARGET_GUIDANCE[target] + optimization_guidance(target))
    return instructions + "\nPreserve the task semantics and public interface. Respect dtype, accuracy and the configured execution model. Analyze actual compiler errors, numerical failures and per-case timings. Distinguish measured facts from hypotheses. Never invent measurements."


def public_metadata(value):
    """Internal identity/transport records do not belong in model context."""
    excluded = {"source_path", "source_sha256", "source_hash", "dataset_root", "extension_path", "build_dir", "suite_id"}
    if isinstance(value, dict):
        return {key: public_metadata(item) for key, item in value.items() if key not in excluded}
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    return copy.deepcopy(value)


def execution_context(metadata, settings=None):
    context = public_metadata(metadata)
    if settings is not None:
        context["evaluation"] = dict(rtol=settings.rtol, atol=settings.atol,
            default_tolerances={"float32": [1e-4, 1e-5], "float16": [1e-2, 5e-3], "bfloat16": [2e-2, 1e-2]},
            warmup=settings.warmup, trials=settings.trials,
            compile_timeout=settings.compile_timeout, execution_timeout=settings.execution_timeout,
            timing_boundary="Model.forward including output allocation; excludes input cloning/state reset",
            timing="cuda_event" if metadata["target"] == "cuda" else "perf_counter_ns")
    return context

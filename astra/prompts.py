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

INDEPENDENT_KERNEL_POLICY = """
Independent-kernel experiment constraints (apply to every candidate and refinement):
- The objective is independently generated native compute kernels, not delegation
  to an existing framework operator or numerical-library implementation. Implement
  ALL task computation in the generated C++/CUDA sources for ALL valid task inputs.
- Do not compute results with Python torch, torch.nn, torch.nn.functional, NumPy,
  or other tensor/numerical APIs. The same prohibition applies inside C++:
  no ATen/LibTorch computational operators via at::, torch::, Tensor methods,
  torch.ops, dispatcher calls, or indirect wrappers around such operators.
  Moving a PyTorch operation from Python to a C++ extension does not make it
  an independently generated kernel.
- PyTorch/ATen may only provide integration plumbing: Tensor handles, metadata
  (shape, stride, dtype, device), data pointers, empty output/workspace allocation,
  metadata-only views, and independent copies of reference parameters/buffers
  during initialization. Python may bind arguments and invoke the extension.
  Perform numerical initialization, casts, packing and layout transformations in
  your own native code when needed; allocation is not permission to compute.
- Do not call the reference model, its forward/submodules, the original operator,
  or a saved baseline callable on ANY execution path. No shape-, size-, stride-,
  dtype- or error-dependent fallback to an original/framework/library implementation.
  Shape dispatch, small-input serial paths and generic/tail paths are allowed only
  when every path uses your own generated native computation. Preserve support
  for the task's valid inputs, including legal broadcasting and noncontiguous data.
- Do not delegate task computation to BLAS, CBLAS, LAPACK, MKL, OpenBLAS, BLIS,
  oneDNN, Eigen numerical routines, cuBLAS, cuDNN, cuFFT, or other library-provided
  compute kernels, directly or indirectly. Do not add dependencies or load a
  prebuilt compute implementation to bypass this restriction. Scalar standard
  math functions, compiler intrinsics, NEON/SVE/CUDA instruction intrinsics,
  OpenMP, and CUDA stream/launch/runtime APIs are allowed within the target ISA,
  accuracy and execution constraints; they do not authorize library compute calls.
- Planning must propose only compliant implementations. If previous candidate
  sources delegate computation or contain a prohibited fallback, explicitly plan
  their replacement with generated native code rather than retaining that path.
"""

BOUNDARY_CORRECTNESS = """
Boundary and layout correctness requirements for every generated/refined kernel:
- Account for every logical element exactly once on every shape/stride dispatch path,
  including scalar, contiguous, noncontiguous, packed, SIMD, threaded and CUDA paths.
- After a loop unrolled by U vectors, the remainder can be almost U full vectors.
  Do not assume a single masked vector completes the remainder. Use a complete
  vector cleanup loop and a final predicate/scalar cleanup as appropriate.
- For reductions, preserve partial sums in inactive lanes and include every valid
  contribution exactly once; initialize accumulators and never read unwritten output.
  For GEMM, handle M/N/K tile remainders independently. Mask all tail loads/stores.
- When changing vector width, unroll factor, tile size, packing or dispatch thresholds,
  reason explicitly about lengths below/equal/above the boundary and all affected strides.
  Use runtime SVE lane counts; observed VL is for tests, not a hard-coded kernel assumption.
- Planning must audit each proposed transformation and previous source's loop bounds,
  index updates, remainder loops and layout branches before proposing performance changes.
  Fixed-suite success is evidence only for those inputs, not proof of all paths.
  Audit these paths while implementing the kernel, but do not enumerate compact-suite
  coverage gaps or change the frozen suite. Never invent test outcomes.
"""

TARGET_GUIDANCE = {
    "cuda": "Generate CUDA C++ (.cu) with at least one __global__ compute kernel. Launch on PyTorch's current CUDA stream; check shapes/dtypes and handle tails.\n",
    "neon": "Generate AArch64 C++ using <arm_neon.h> NEON intrinsics within the supplied compiler flags. Do not use SVE, SVE2 or SME even if the host supports them. Handle fixed-width vector tails and unaligned inputs safely.\n",
    "sve": "Generate AArch64 C++ for an SVE-enabled target. NEON (<arm_neon.h>), SVE (<arm_sve.h>) and mixed implementations are allowed within the supplied compiler flags. Choose according to task shapes, strides and measured performance; using both ISAs or using SVE is not mandatory. For SVE code, use predicates and runtime lane counts; do not assume a fixed vector length. Do not assume SVE2, SME or unsupported compiler intrinsics.\n",
}

ORCHESTRATOR = """You are the Orchestrator Agent responsible for coordinating the kernel optimization process.
Your role is to receive workflow notifications and track optimization progress.
Keep responses concise and focused on coordination tasks. The application owns scheduling and state updates.
"""

TESTER = """You handle test case generation and correctness verification with unified test data.
Generate the extra test suite ONCE, before candidate evaluation, and reuse it for ALL versions.
Keep every original dataset case. Never change model initialization, weights, task semantics or the frozen suite.
When generating tests, call generate_comprehensive_test_cases with a complete JSON specification.
Generate exactly 15 additional cases under either coverage profile. Mix practical
small/medium cases, signs, zeros and ones, broadcasting/strides and vector tails where legal.
For tasks with very large original tensors, use smaller legal batch or spatial dimensions for most
cases and preserve only the dimensions fixed by the constructor or operator contract. Include at most
one original-scale case for representative performance; do not duplicate a huge original tensor across
the suite. Each case must be practical to materialize, benchmark and profile independently.
Use the supplied boundary coverage requirements and declare every input axis variable or fixed from the reference source.
The case count is a hard budget: combine checks across cases and stop after the fixed-size suite is submitted.
Coverage may be partial and does not block the frozen suite; do not add repair cases after
freezing or enumerate untested items.
Follow the configured compact/full coverage profile and strict additional-case cap. In compact mode,
combine boundary sizes, value patterns and layouts within fifteen cases; do not enumerate every size/layout pair.
Cover the multi-vector remainder in each legal stride category; a single odd length or a transpose
of unrelated axes does not establish tail coverage. Include nonzero values to reveal omitted contributions.
Respect the reference source: parameterized layers may constrain channels, normalized shapes, dtypes and label ranges.
Use the reference model to validate cases through the tool; repair invalid specifications rather than discarding cases.
During each version's evaluation, call verify_candidate. Do not invent results or generate new tests.
Review candidate loop bounds, cleanup loops, inactive reduction lanes, M/N/K tile tails and shape/stride
dispatch paths. Cross-check them against execution.boundary_coverage. Report suspected counterexamples
separately from actual pass/fail; tool success does not prove general correctness.
Never mutate the frozen suite mid-run. Coverage status is retained without enumerating gaps.
The same fixed inputs are sent to the profiling agent after correctness passes; failed candidates are never timed.
"""

PROFILER = """You are the Profiling Agent responsible for measuring each kernel version on the unified test suite.
Call benchmark_candidate to obtain actual performance data, rechecking correctness first.
Use the SAME frozen cases across ALL versions. Never generate new cases or synthesize timings.
Report measured case latencies and speedups relative to the corresponding reference baselines.
Version selection uses the configured arithmetic/geometric mean speedup on generated cases.
The application independently remeasures the frozen winner on original dataset cases after all iterations.
When execution.evaluation.cpu_perf.enabled is true, also call profile_candidate after
benchmark_candidate. Each frozen case has a separate perf-stat session, repeated independently;
the baseline is profiled once per case and reused. Compare matching case IDs only.
Report per-case IPC, cycles/instructions and cache/branch counters with their running percentages.
Unsupported or missing events are unavailable, never zero. Generic cache events do not
identify cache levels. Counters include wrapper/loop/control overhead and may multiplex;
interpret them as diagnostic evidence and hypotheses, not pure-kernel cycles or ranking times.
Never aggregate all cases into one counter measurement. Never invent measurements.
"""


def optimization_guidance(target):
    if target == "cuda":
        return CUDA_OPTIMIZATION
    return (Path(__file__).resolve().parent.parent / "prompts" / ("cpu_" + target + ".md")).read_text(encoding="utf-8")


def threading_guidance(target, metadata=None):
    if target == "cuda":
        return ""
    threads = (metadata or {}).get("threads", 1)
    plan = (metadata or {}).get('thread_plan')
    if plan:
        return (f"\nCPU execution: automatic per-case thread plan, even counts up to {threads} threads. "
                "The PyTorch baseline and generated kernel use the same thread count for each case; "
                "the plan is selected from input bytes and is recorded in execution.thread_plan. "
                "All selected workers stay on one NUMA node. Use OpenMP/SIMD for sufficiently large cases "
                "and avoid assuming every case uses the maximum. Respect the runtime's per-case team size: "
                "do not hard-code num_threads, change OpenMP settings, create other pools or alter binding.\n")
    if threads == 1:
        return "\nCPU execution: exactly one thread. Do not use OpenMP, worker pools or external threaded libraries.\n"
    return (f"\nCPU execution: at most {threads} threads, on the supplied fixed CPU affinity. "
            f"The PyTorch baseline also uses {threads} intra-op threads and one inter-op thread. "
            "OpenMP is enabled by the framework's compiler and linker flags; <omp.h> and #pragma omp are allowed. "
            "Use OpenMP for sufficiently large independent outer loops and NEON/SVE within each thread. "
            "Prefer static scheduling for regular work, avoid false sharing, and use size thresholds to avoid "
            "parallel startup costs on small inputs. Preserve the runtime thread/affinity settings: do not "
            "increase thread counts, spawn separate pools, enable nested parallelism or change CPU binding. "
            "Serial paths for small cases are allowed. Do not call external numerical libraries.\n")


def strategy_instructions(target, metadata=None):
    if target == "cuda":
        instructions = CUDA_STRATEGY
    else:
        instructions = ("You are an expert Arm CPU kernel optimization strategist. Give concrete, implementable suggestions only (no code).\n"
                        + TARGET_GUIDANCE[target] + optimization_guidance(target))
    instructions += BOUNDARY_CORRECTNESS
    return instructions + threading_guidance(target, metadata) + INDEPENDENT_KERNEL_POLICY + "\nPreserve the task semantics and public interface. Respect dtype, accuracy and the configured execution model. Analyze actual compiler errors, numerical failures and per-case timings. Use CPU perf data only for matching case IDs and supported events; account for running percentages, wrapper overhead and multiplexing. Do not infer a specific cache level or causal bottleneck from generic events alone. Distinguish measured facts from hypotheses. Never invent measurements."


def public_metadata(value):
    """Internal identity/transport records do not belong in model context."""
    excluded = {"source_path", "source_sha256", "source_hash", "dataset_root", "extension_path", "build_dir", "suite_id", "samples_ms"}
    if isinstance(value, dict):
        from .cpu_profiling import perf_context
        return {key: public_metadata(perf_context(item) if key in ('cpu_perf', 'baseline_perf', 'base_version_perf') and isinstance(item, dict) and 'events' in item else item)
                for key, item in value.items() if key not in excluded}
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    return copy.deepcopy(value)


def execution_context(metadata, settings=None):
    context = public_metadata(metadata)
    if context.get("hardware", {}).get("cpu"):
        from .hardware import hardware_context
        context["hardware"] = hardware_context(context["hardware"])
    if settings is not None:
        context["evaluation"] = dict(rtol=settings.rtol, atol=settings.atol,
            default_tolerances={"float32": [1e-4, 1e-5], "float16": [1e-2, 5e-3], "bfloat16": [2e-2, 1e-2]},
            warmup=settings.warmup, trials=settings.trials,
            compile_timeout=settings.compile_timeout, execution_timeout=settings.execution_timeout,
            timing_boundary="Model.forward including output allocation; excludes input cloning/state reset",
            timing="cuda_event" if metadata["target"] == "cuda" else "perf_counter_ns")
        context['evaluation']['cpu_perf'] = dict(enabled=metadata['target'] != 'cuda' and getattr(settings, 'cpu_perf', True),
            repeats=getattr(settings, 'perf_repeats', 64), batch_mib=getattr(settings, 'perf_batch_mib', 64),
            protocol='Separate diagnostic session for EACH frozen case; baseline once, candidate every correct version; never used for ranking')
    return context

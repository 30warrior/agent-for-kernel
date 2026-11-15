# Astra: Dataset-driven GPU and Arm CPU Kernel Optimization

This independent copy adds **local LLM orchestration with an offline SSH worker**.
Windows/Linux/macOS controllers need only Python, `openai-agents`, SSH and SCP.
The execution server supplies PyTorch, Ninja and the native CPU/CUDA toolchain.
See [the remote workflow and Windows commands](docs/remote-execution.md).

```powershell
# No model/API call: upload and evaluate the handwritten NEON Add example.
.\run_remote.ps1 -Target neon -EvaluateOnly "$PWD/examples/remote_add/neon"
# Full multi-agent run after installing requirements-controller.txt and setting
# OPENAI_API_KEY locally (and OPENAI_BASE_URL if using a compatible endpoint).
.\run_remote.ps1 -Python .\.venv\Scripts\python.exe -Target sve -Model YOUR_MODEL -ModelApi chat_completions
```

Astra is being extended to generate and iteratively optimize kernels from
KernelBench and AKG benchmark PyTorch tasks. Planned generation targets are
CUDA, AArch64 NEON and AArch64 SVE. Each target runs natively on its corresponding
machine, using a PyTorch reference on the same device.

## Development status

This branch contains the migration in separate, ordered commits:

1. Remove mandatory sgl-kernel dependencies and implicit library baselines.
2. Load KernelBench and AKG tasks through a common dataset interface.
3. Prepare fixed reference inputs, model state, outputs and baseline measurements.
4. Generate the initial candidate from task source instead of requiring a .cu file.
5. Compile CUDA and Arm C++ candidates with a common ModelNew contract.
6. Verify outputs and measure latency with device-appropriate timing.
7. Share the multi-agent refinement workflow across generation targets.
8. Report reference speedup, refinement speedup and batch task outcomes.

All eight migration steps have code implementations. The remote-execution branch
adds native smoke validation, recorded in [the remote workflow](docs/remote-execution.md).
Actual LLM generation still needs a locally configured API and controller SDK.
The legacy CUDA entry point is separate from the new workflow.

## Dependencies

`requirements.txt` no longer installs sgl-kernel or PyCUDA. Install a PyTorch
build appropriate to the machine, together with the remaining Python dependencies.
No CUDA-only reference library is required by the new dataset-based design.

The existing `cuda_kernel_optimizer_multi.py` remains a legacy CUDA-source
entry point while the common workflow is introduced. It requires explicit
`--baseline-module` and `--baseline-func` arguments, plus an initial CUDA source.
Reference callables are imported lazily from that module, with no sgl-kernel
fallback. See [the legacy CUDA workflow](docs/legacy-cuda.md) for details.

Existing specialized examples under `test/` are retained as historical CUDA
examples. They are not the new dataset runner and may require optional libraries.

## Development environment

Original Arm development is confined to `hzh/Astra`; remote runs use the separate
`hzh/Astra-remote` directory. Existing shared Python environments
and other users' directories must not be modified. Generated sources, build
artifacts, reports and local environment files must not be committed.

## Inspect dataset definitions

`python kernel_optimizer.py --dataset akg --dataset-root /path/to/akg_kernels_bench --list-tasks`

`python kernel_optimizer.py --dataset kernelbench --dataset-root /path/to/KernelBench/KernelBench --list-tasks`

Listing reads Python syntax only. It does not establish CPU compatibility or
run correctness checks. Select explicit task paths; accelerator-specific tasks
must be excluded from CPU manifests. Dynamic AKG tasks use get_inputs_dyn_list.

## Prepare the reference

Add `--task static_shape/elemwise/elemwise_add_001.py --target neon --prepare-only`
to an AKG invocation to materialize fixed inputs and reference outputs and measure
the baseline. This explicitly runs the reference, so only invoke it when desired.
The default is single-threaded inference (`eval`), five warmups and thirty samples.
Files stay under `optimization_runs/` inside this repository. Stateful/stochastic
training tasks are outside the initial inference contract. Reference snapshots
are trusted local PyTorch artifacts and must not be loaded from untrusted sources.

## Generate the first candidate

Use `--generate-only` with a selected task to produce `v1/kernel.cpp` (NEON/SVE)
or `v1/kernel.cu` (CUDA) plus `v1/model.py`. This option calls the model; it is not
a dry run. API settings use `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `--model`
and `--model-api`. Tracing is disabled unless explicitly enabled. Secrets are
not written into run metadata. The ModelNew wrapper contract is recorded in
`astra/generation.py`; compilation and refinement are connected by the shared workflow.

## Native compilation

Generation now exposes a real compile tool. CUDA uses nvcc and the active GPU
architecture; NEON/SVE use the native AArch64 compiler with explicit ISA flags.
No fast-math flag is enabled by default. Compiler output and exact options are
saved per candidate. Compilation runs in a subprocess with a process-group timeout
(`--compile-timeout`, seconds), with two build jobs and a candidate-local build
directory. C++ headers/toolchain matching the installed PyTorch build are required.
Generated code is executable native/Python code and runs as the current account;
this implementation does not provide a security sandbox for untrusted tasks.

## Candidate evaluation

Use `--evaluate-only optimization_runs/<run>/v1` with the same task and target to
compile and check saved sources without model calls. Tensor shape, dtype, device,
sequence/dictionary outputs and post-call input mutations are checked. Floating
outputs use dtype-specific tolerances (overridable with --rtol/--atol); integer and
boolean results are exact. Matching NaNs are allowed; infinity signs must match.
Only candidates passing every case are timed. CPU runs use one thread and pin to
the first allowed logical CPU by default; override it with --cpu-core. Cases are
pre-cloned and model state restored before timing. Output allocations and wrapper
call overhead remain inside the boundary on CPU; CUDA events measure stream time.
CPU and CUDA numbers are not directly compared to one another.

## Shared refinement workflow

Without an *-only switch, the entry point runs Orchestrator planning, initial code
generation, CorrectnessTester, Benchmarker and OptimizationStrategist refinement.
The host enforces actual compilation/checks/timing even if an agent skips its tool.
`--max-iterations 3` means v1 plus up to three further candidates; failed candidates
consume a round and feed repair diagnostics to the next attempt. Valid regressions
refine from the best valid candidate; invalid sources are repaired directly.
Progress is saved after each attempted version. Candidate execution uses subprocesses
with --execution-timeout so a segfault or hang is recorded as a failure. Compiler and
execution timeouts are separate. Task inputs and reference weights are restored from
the same run snapshot in evaluation workers; only the reference device is used.

## Run a selected task on Arm

```bash
cd /mnt/data_nvme4n1/lab317_cnndev/hzh/Astra
export PYTHONDONTWRITEBYTECODE=1
# Install dependencies into a project-local environment if needed; shared envs are read-only.
# Supply OPENAI_API_KEY through the environment. Never commit it.
python -B kernel_optimizer.py --dataset akg --dataset-root benchmarks/akg \
  --task static_shape/elemwise/elemwise_add_001.py --target sve \
  --model YOUR_MODEL --model-api chat_completions --max-iterations 3
```

Use `--target neon` for NEON. On a CUDA host, use `--target cuda` and a CUDA-enabled
PyTorch build. For KernelBench use `--dataset kernelbench --dataset-root benchmarks/kernelbench`.
The bundled datasets retain their original sources/licenses and are not all CPU-compatible;
see benchmarks/PROVENANCE.md and SHA256SUMS.json. The supplied four-task manifest is a
starting selection, not a claim that generated code has passed validation.

## Batch runs and reports

```bash
python -B kernel_optimizer.py --dataset akg --dataset-root benchmarks/akg \
  --manifest manifests/akg_fp32.json --target sve --model YOUR_MODEL
```

A manifest contains a `tasks` list of relative paths. An object entry can use
`{"task": "path.py", "skip_reason": "explicit reason"}` to declare a skip. Runs are
serial; failures remain visible and do not stop later tasks. No benchmark has been
run merely by copying/listing the dataset.

Each run atomically updates benchmark_results.json, summary.json,
benchmark_summary.csv and benchmark_details.csv after every attempted candidate.
Per-case CPU latency is the median of host-clock samples; GPU latency is the median
of CUDA-event samples. Version ranking uses the mean of per-case medians on the
same complete case set. Aggregate reference speedup is sum(reference medians) /
sum(candidate medians); refinement speedup uses v1's sum instead. If v1 is invalid,
refinement speedup remains empty. Incorrect or incompletely measured candidates
cannot win. Each case has its own ID, even if shapes match.

Batch reports include compile rate, correctness rate and fast_1 (correct, completely
measured tasks faster than their reference). Denominators include failed attempted
tasks; declared skips are reported separately and excluded. The geometric mean is
explicitly limited to successful tasks and must be read alongside coverage. CPU and
GPU results are evaluated against their own native references, not each other's times.

## Validation status and boundaries

No unit tests, native compilation, generated kernels or model calls were run during
implementation at the user's request. Python source parsing, CLI help, dataset
interface listing, diff checks and source snapshot checksums can be reviewed without
running kernels. End-to-end correctness/performance on Arm and CUDA remains to be
validated in the intended environments. CPU execution currently supports single-thread
inference. Additional edge shapes, thread scaling, stronger numerical stress testing,
repeatability across hardware, and RAG are later work; no performance gain is claimed.

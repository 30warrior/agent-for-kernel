# Fixed-suite multi-agent workflow

The workflow uses the same scheduling and evaluation rules for CUDA, NEON and SVE.
`--target` selects hardware guidance, compiler flags and the native backend.
RAG is not implemented in this revision.

## Mapping to the supplied Astra workflow figure

| Figure component or arrow | Implementation |
| --- | --- |
| Source kernel | KernelBench/AKG `Model`, constructor inputs and reference `forward` source |
| Testing Agent | `generate_suite` in `astra/test_cases.py`; per-version `TestingAgent` in `astra/workflow.py` |
| Test Input → Profiling Agent | One persisted `reference.pt` with original/generated inputs, expected outputs and mutations |
| Profiling Agent | `ProfilingAgent` invokes real benchmarking after verification passes |
| Test Result → Planning Agent | Per-case pass/fail, failure tracebacks and latest version result |
| Speedup → Planning Agent | Measured latency, corresponding reference baseline and speedup |
| Planning Agent → Coding Agent | Hardware-specific suggestions, previous source and diagnostics |
| Coding Agent → Testing Agent | Every generated version is compiled and checked on the same snapshot |
| Orchestrator | Notifications/progress, no planning tool; host Python schedules the agents |

This maps the supplied figure, not unspecified details of the full paper. The
intentional difference is the starting representation: this fork first implements
a PyTorch reference as native v1; original Astra starts with an existing CUDA
implementation. Tests are frozen from the reference before v1 so candidate errors
cannot influence suite selection.

## Test creation and reuse

1. Prepare unmodified dataset cases and reference model/weights.
2. TestingAgent proposes exactly `--extra-test-cases` additional cases (default 8).
   Descriptors support argument copies, tensor distributions/shapes, legal
   transposes and scalars of the original type. Dtype/device stay fixed.
3. Validate all proposed cases on the reference. Errors return to TestingAgent
   for repair. Freeze only after the complete proposal validates and all
   reference baselines can be measured.
4. Save `test_suite.json` and the input/output snapshot. Reject regeneration.
   `--test-suite-max-mib` defaults to 512 MiB of additional input tensor storage;
   this does not bound reference temporaries/outputs or resize original inputs.
5. Verify every compiled version. If any test fails, return diagnostics to
   planning and skip timing for that version. Otherwise profile every case with
   identical inputs, weights and timing settings.
6. Check returned suite identity, count and order on the host.

Correctness must pass **all** original and generated cases. Best-version ranking
and headline speedup use **original dataset cases**, preserving the benchmark's
workload. Generated-case timings are reported separately and remain available to
planning. They do not redefine KernelBench/AKG performance with arbitrary small
shapes. No separate operator workflows or batches are introduced.

Test generation must respect model constraints. Generic descriptors do not
guarantee coverage for every task; unsupported inputs/reference failures remain
visible. The current CPU contract is single-threaded inference, not training.

`--test-spec path/to/test_suite.json` replays a saved specification (or plain
`{"cases": [...]}` object), without model test generation. Set `--extra-test-cases`
to its additional-case count. Inputs are rematerialized using the specification
and seeds; each new run gets its own reference snapshot.
The focused `--generate-only`/`--evaluate-only` actions do not ask the model to
generate tests; add `--test-spec` to explicitly replay a suite for those actions.
`--prepare-only` only prepares original cases.

## Prompt files and review artifacts

- CPU suggestions: `prompts/cpu_neon.md`, `prompts/cpu_sve.md`. Both coding and
  planning read the selected file. Edit before starting a run.
- Original CUDA optimization/strategy text and shared role contracts:
  `astra/prompts.py`. CUDA retains its four original Suggestions and CUDA Focus.
- Run context: `test_generation_prompt.json`, `role_instructions.json`.
- Version context: `generation_prompt.json`, `generation_instructions.txt`,
  `testing_prompt.json`, `profiling_prompt.json`, `strategy_prompt.json`.
- Notifications: `orchestrator_events.json`.
- Reports: `benchmark_results.json`, `benchmark_summary.csv`,
  `benchmark_details.csv`; case details include name and origin.

Source paths/checksums and suite identity remain internal provenance and are
filtered out of structured LLM context. Reference source, hardware, test
descriptions, accuracy requirements and measurements are supplied. Compiler
diagnostics can naturally contain compiler file paths.

## Arm execution validation (2026-11-16)

A real offline SVE worker run used one original AKG Add case and eight additional
cases, with 1-, 3-, 7-, 31-, 33-, 65-, 1025- and 2048-column shapes. An invalid
nonbroadcastable proposal was rejected before freezing. A correct handwritten
SVE candidate passed and was timed on all nine cases; a subtraction variant
failed and received no timings; a restored correct variant passed on the same
suite. This checks execution/feedback, not LLM generation.

The local result directory is
`optimization_runs/suite_20261116_124956_411068/`. Raw run artifacts stay ignored.
CUDA Suggestions and strategy instructions were also compared with the original
repository snapshot; report checks used the actual nine-case Arm measurements.

## API relay connectivity during validation

The attempted new end-to-end LLM run did not complete. Default asynchronous TLS
connections to the configured relay timed out. Connection-only probes returned
HTTP 401 with TLS 1.2 and certificate verification enabled. The optional
`--tls-max-version 1.2` (PowerShell: `-TlsMaxVersion 1.2`) makes that setting
explicit; the default TLS behavior is unchanged. A subsequent real orchestrator
request succeeded, but test generation encountered a connection read failure.
TLS compatibility does not guarantee that the relay remains available.

Therefore the new nine-case execution checks must not be presented as a successful
new LLM generation run. Earlier actual LLM NEON/SVE runs in `remote-execution.md`
used the preceding workflow.

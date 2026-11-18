# Hardware optimization guidance

Edit `cpu_neon.md` or `cpu_sve.md` to replace the initial CPU suggestions. Keep
the `Suggestions:` bullet list format if convenient; the following `Focus:`
section is also editable. Files are read as UTF-8 when agents are constructed.
Restart the workflow after editing so coding and planning use the same revision.

Both the CodeGenerator and PlanningAgent receive the selected file. Selection is
explicit: `--target neon`, `--target sve`, or `--target cuda`. The framework checks
worker ISA capabilities before generation; it does not choose NEON because the
host process runs on a non-Arm laptop.

CUDA guidance lives in `astra/prompts.py`: `CUDA_OPTIMIZATION` retains the
original Astra optimization suggestions, and `CUDA_STRATEGY` retains its CUDA
strategy instructions. Wrapper/tool contracts differ because this fork accepts
KernelBench/AKG PyTorch tasks instead of an existing sgl-kernel implementation.
Accuracy requirements still apply to any fast-math suggestion.

`astra/prompts.py:INDEPENDENT_KERNEL_POLICY` defines the shared independent-kernel
constraints for Coding and Planning on CUDA, NEON and SVE, including generate-only
and every refinement. Task computation must use generated native code: no Python
PyTorch/NumPy or C++ ATen/LibTorch compute, original-implementation fallbacks on
selected shapes, or BLAS/oneDNN/cuBLAS and similar library compute. Tensor metadata,
empty allocations and initialization-time parameter copies remain allowed for
integration. All shape dispatch and serial/generic paths must compute in generated
native code. These are prompt constraints, not an automatic compliance checker;
correctness and timing checks alone do not establish compliance. Existing saved
candidates and runs are not retroactively validated by changing these prompts.

Shared testing/profiling/orchestrator role contracts also live in
`astra/prompts.py`. Editing optimization suggestions does not change fixed test
inputs, correctness tolerances, compiler flags or the measurement boundary.

Each run saves actual role instructions and generation instructions for review.
`astra/prompts.py:BOUNDARY_CORRECTNESS` adds shared loop/tail/layout requirements
to Coding and Planning. Testing generates a boundary-covered suite before v1,
then reviews each candidate against the frozen coverage report without adding cases.
`astra/coverage.py` checks actual materialized shapes and strides before freezing;
fixed-axis and layout exemptions retain source justifications for human review.
The minimum extra case count is 8 and the default maximum is 128; the input byte
budget still applies. See `docs/server-execution.md` for the new coverage schema.
CPU threading instructions are added from the run configuration: `--threads 1`
requires serial kernels, while `--threads N` (N > 1) enables OpenMP with the same
thread budget and CPU allocation as the PyTorch baseline. Do not hard-code a
single-thread requirement in the editable CPU Markdown files.
No RAG retrieval is implemented yet; this file selection supplies the initial
hardware-specific context and can be extended by a later retrieval stage.

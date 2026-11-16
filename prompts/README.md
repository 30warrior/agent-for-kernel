# Hardware optimization guidance

Edit `cpu_neon.md` or `cpu_sve.md` to replace the initial CPU suggestions. Keep
the `Suggestions:` bullet list format if convenient; the following `Focus:`
section is also editable. Files are read as UTF-8 at the start of a workflow.
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

Shared testing/profiling/orchestrator role contracts also live in
`astra/prompts.py`. Editing optimization suggestions does not change fixed test
inputs, correctness tolerances, compiler flags or the measurement boundary.

Each run saves actual role instructions and generation instructions for review.
No RAG retrieval is implemented yet; this file selection supplies the initial
hardware-specific context and can be extended by a later retrieval stage.

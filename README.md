# Astra: Dataset-driven GPU and Arm CPU Kernel Optimization

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

Only step 1 is implemented at this commit. Dataset execution and Arm generation
are not yet available; do not treat the existing CUDA entry point as CPU-ready.

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

All Arm development is confined to `hzh/Astra`. Existing shared Python environments
and other users' directories must not be modified. Generated sources, build
artifacts, reports and local environment files must not be committed.

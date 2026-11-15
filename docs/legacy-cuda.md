> Historical CUDA-source workflow. Requires a CUDA host and an explicitly selected reference library.
> sgl-kernel is optional and must be installed separately if selected.

# Astra: A Multi-Agent System for GPU Kernel Performance Optimization
[![arXiv](https://img.shields.io/badge/arXiv-2509.21629-b31b1b.svg)](https://arxiv.org/abs/2509.07506) [![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/license/apache-2-0) 

## Usage

```bash
python3 cuda_kernel_optimizer_multi.py \
    --api-key <OPENAI_API_KEY> \
    --initial-kernel-path <PATH_TO_INIT_CUDA_FILE> \
    --compare-kind <KERNEL_TYPE> \
    --baseline-module sgl_kernel \
    --baseline-func <BASELINE_FUNCTION_NAME> \
    --generated-export-func <EXPORT_FUNCTION_NAME>
```

## Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--api-key` | No | `OPENAI_API_KEY` | API key; provide this option or set the environment variable |
| `--base-url` | No | `OPENAI_BASE_URL` or OpenAI default | OpenAI-compatible API base URL, including `/v1` |
| `--model` | No | `o4-mini` | Model name for all five agents |
| `--model-api` | No | `responses` | Provider interface: `responses` or `chat_completions` |
| `--disable-tracing` | No | Off | Disable OpenAI trace uploads; recommended for third-party providers |
| `--initial-kernel-path` | Yes | - | Path to the initial CUDA kernel file (`.cu`) to optimize |
| `--max-iterations` | No | 5 | Maximum number of optimization iterations |
| `--baseline-module` | Yes | - | Explicit Python module containing the baseline function |
| `--baseline-func` | Yes | - | Name of the baseline function to compare against |
| `--compare-kind` | No | `generic` | Comparison mode tag that guides test generation prompts (e.g., `rmsnorm`, `silu`, `mergestate`) |
| `--generated-export-func` | No | Same as `baseline-func` | Export function name expected in the generated PyBind module |

## Examples

### 1. RMSNorm Kernel Optimization

Optimize a fused add RMSNorm kernel:

```bash
python3 cuda_kernel_optimizer_multi.py \
    --api-key $OPENAI_API_KEY \
    --initial-kernel-path /path/to/rms_v1.cu \
    --compare-kind rmsnorm \
    --baseline-module sgl_kernel \
    --baseline-func fused_add_rmsnorm \
    --generated-export-func sgl_fused_add_rmsnorm
```

### 2. SiLU (Swish) Kernel Optimization

Optimize a SiLU activation with multiplication kernel:

```bash
python3 cuda_kernel_optimizer_multi.py \
    --api-key $OPENAI_API_KEY \
    --initial-kernel-path /path/to/silu_mul_v1.cu \
    --compare-kind silu \
    --baseline-module sgl_kernel \
    --baseline-func silu_and_mul \
    --generated-export-func sgl_silu_mul
```

### 3. Merge State Kernel Optimization

Optimize a merge state kernel (commonly used in attention mechanisms):

```bash
python3 cuda_kernel_optimizer_multi.py \
    --api-key $OPENAI_API_KEY \
    --initial-kernel-path /path/to/merge_v1.cu \
    --compare-kind mergestate \
    --baseline-module sgl_kernel \
    --baseline-func merge_state \
    --generated-export-func merge_state
```

### Third-party OpenAI-compatible providers

Use the provider's base URL and an available model with tool calling support.
For a provider that supports Chat Completions, enter the key interactively in Bash:

```bash
read -rsp "API Key: " OPENAI_API_KEY; echo
export OPENAI_API_KEY
python3 cuda_kernel_optimizer_multi.py \
    --base-url https://your-provider.example/v1 \
    --model your-provider-model \
    --model-api chat_completions \
    --disable-tracing \
    --max-iterations 1 \
    --initial-kernel-path test/silu/silu_mul_v1.cu \
    --compare-kind silu \
    --baseline-module sgl_kernel \
    --baseline-func silu_and_mul \
    --generated-export-func sgl_silu_mul
```

If the provider supports the Responses API instead, use `--model-api responses`.
The baseline callable (`silu_and_mul`) and generated PyBind export (`sgl_silu_mul`)
have different names; specify both. An initial compilation failure stops the run
before any model request.

## Output

Results are saved in a timestamped directory under the project's
`cuda_optimization_runs/`, regardless of the working directory used to launch it.
Initial source is stored in `kernels/<initial-name>_v1.cu`; generated versions use
`kernels/kernel_v2.cu`, `kernels/kernel_v3.cu`, etc. Their compiled extensions are
stored alongside them in `kernels/build_v2/`, `kernels/build_v3/`, etc.
The save tool accepts only a version and source code; agents cannot choose an
output directory. `cuda_optimization_runs/latest` links to the latest run.

Each measured version is saved immediately to reports in its run directory:

- `performance.log`: the displayed `Benchmark Summary` for each version, with
  the `CURRENT PERFORMANCE` context appended before each strategy request.
- `benchmark_summary.csv`: one row per version, including correctness status,
  successful/failed test counts, average kernel and baseline time in milliseconds,
  aggregate speedup, and source path.
- `benchmark_details.csv`: one row per version and test case, including shape,
  timing variation, throughput, speedup, correctness status, and errors.
- `benchmark_results.json`: full benchmark and correctness results, test
  specifications, and kernel paths.

Summary averages include successful benchmarks only. Aggregate speedup is the sum
of baseline times divided by the sum of generated-kernel times for those cases;
values above 1 mean faster than the baseline. Failed/missing measurements are left
blank, not recorded as zero. Compare versions using the same cases and check
correctness and failure counts before judging performance. Detailed rows retain
distinct test names even when shapes match. Reports update after every benchmark
and correctness recheck; completed measurements remain on disk if a later step
fails. Each file is replaced atomically so readers do not see partially written
contents. These reports apply to new runs; old in-memory results cannot be
recovered automatically.

For example, inspect the latest version summary on the server:

```bash
cat ~/Astra/cuda_optimization_runs/latest/benchmark_summary.csv
tail -f ~/Astra/cuda_optimization_runs/latest/performance.log
```

Profiling samples use the active CUDA device's UUID, so `CUDA_VISIBLE_DEVICES`
remapping also applies to NVML queries. Optional NVML failures are recorded as
notes and do not prevent CUDA event timing. GPU utilization and memory-controller
activity are device-wide samples, not per-kernel measurements; memory capacity
usage is reported separately. Effective bandwidth is estimated bytes per call
divided by event time (GiB/s). Occupancy, arithmetic intensity, theoretical peak
bandwidth, and bandwidth efficiency are not measured or synthesized. Missing
samples no longer produce an unconditional package-install instruction.

## Citation

If our research inspires you, please cite our paper:

```bibtex
@inproceedings{wei2025astra,
  title={Astra: A Multi-Agent System for GPU Kernel Performance Optimization},
  author={Wei, Anjiang and Sun, Tianran and Seenichamy, Yogesh and Song, Hang and Ouyang, Anne and Mirhoseini, Azalia and Wang, Ke and Aiken, Alex},
  booktitle={NeurIPS 2025 Fourth Workshop on Deep Learning for Code},
  year={2025}
}
```

## License

This project is licensed under the [Apache License 2.0](https://opensource.org/license/apache-2-0).

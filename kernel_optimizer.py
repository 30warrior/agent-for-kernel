"""Dataset-driven Astra entry point; imports execution dependencies lazily."""
import argparse
import os
import json
from pathlib import Path
from datetime import datetime
from astra.datasets import list_tasks, load_task


def build_parser():
    parser = argparse.ArgumentParser(description="Generate and optimize dataset kernels for CUDA, NEON or SVE")
    parser.add_argument("--dataset", choices=("akg", "kernelbench"), required=True)
    parser.add_argument("--dataset-root", required=True, help="AKG benchmark root or KernelBench task root")
    parser.add_argument("--task", help="Python task path relative to dataset-root")
    parser.add_argument("--target", choices=("cuda", "neon", "sve"), default="neon")
    parser.add_argument("--list-tasks", action="store_true", help="List definitions without importing tasks or running kernels")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and benchmark the PyTorch reference; no LLM calls")
    parser.add_argument("--threads", type=int, choices=(1,), default=1, help="Initial CPU backend is single-threaded")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--generate-only", action="store_true", help="Generate v1 sources and expose compilation tools; makes LLM requests")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "o4-mini"))
    parser.add_argument("--model-api", choices=("responses", "chat_completions"), default="responses")
    parser.add_argument("--enable-tracing", action="store_true", help="Opt in to SDK trace uploads")
    parser.add_argument("--tls-max-version", choices=("1.2",),
                        help="Optional TLS 1.2 API relay compatibility; certificate verification stays enabled")
    parser.add_argument("--max-agent-turns", type=int, default=12)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument("--evaluate-only", metavar="CANDIDATE_DIR", help="Compile and evaluate saved kernel/model.py sources without LLM requests")
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--cpu-core", type=int, help="CPU affinity; defaults to the first allowed core for CPU targets")
    parser.add_argument("--max-iterations", type=int, default=3, help="Refinement rounds after v1; zero generates and evaluates v1 only")
    parser.add_argument("--execution-timeout", type=int, default=300)
    parser.add_argument("--extra-test-cases", type=int, default=8,
                        help="Additional cases generated once and reused for all versions")
    parser.add_argument("--test-suite-max-mib", type=float, default=512,
                        help="Additional input storage budget; never resizes original dataset cases")
    parser.add_argument("--test-spec", help="Replay a saved test suite JSON instead of model test generation")
    parser.add_argument("--manifest", help="JSON task list; runs complete workflows serially")
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    return parser


def execute_task(args, parser):
    import math
    if args.extra_test_cases < 2 or not math.isfinite(args.test_suite_max_mib) or args.test_suite_max_mib <= 0:
        parser.error("extra-test-cases must be at least two and test-suite-max-mib must be positive and finite")
    if args.max_iterations < 0 or args.trials < 1 or args.warmup < 0:
        parser.error("max-iterations/warmup must be nonnegative and trials positive")
    if args.compile_timeout <= 0 or args.execution_timeout <= 0 or args.max_agent_turns < 1:
        parser.error("timeouts and max-agent-turns must be positive")
    if any(value is not None and value < 0 for value in (args.rtol, args.atol)):
        parser.error("tolerances must be nonnegative")
    if args.list_tasks:
        for task in list_tasks(args.dataset, args.dataset_root):
            print(json.dumps(task.metadata(), ensure_ascii=False))
        return 0
    if args.manifest:
        if args.task or args.prepare_only or args.generate_only or args.evaluate_only:
            parser.error("--manifest runs complete workflows and cannot be combined with --task or *-only")
        from astra.batch import run_batch
        return run_batch(args)
    if not args.task:
        parser.error("--task is required unless --list-tasks or --manifest is selected")
    if sum(bool(value) for value in (args.prepare_only, args.generate_only, args.evaluate_only)) > 1:
        parser.error("prepare-only, generate-only and evaluate-only are mutually exclusive")
    if not args.prepare_only and not args.evaluate_only and not args.api_key:
        parser.error("Set OPENAI_API_KEY or pass --api-key before model generation")
    if args.target != "cuda" and hasattr(os, "sched_getaffinity"):
        allowed = os.sched_getaffinity(0)
        core = args.cpu_core if args.cpu_core is not None else min(allowed)
        if core not in allowed:
            parser.error("--cpu-core is outside this process's allowed affinity")
        os.sched_setaffinity(0, {core})
    from astra.reference import prepare_reference, save_reference
    from astra.backends import Backend
    from astra.timing import measure
    task = load_task(args.dataset, args.dataset_root, args.task)
    backend = Backend(args.target, args.compile_timeout)
    hardware = backend.capabilities()
    reference = prepare_reference(task, args.target, args.seed, args.threads)
    reference.metadata["hardware"] = hardware
    folder = Path(__file__).resolve().parent / "optimization_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_reference(reference, folder)
    baseline = [measure(reference.model, case, args.target, args.warmup, args.trials) for case in reference.cases]
    (folder / "baseline.json").write_text(json.dumps(baseline, indent=2))
    print("Run directory:", folder, flush=True)
    if args.prepare_only:
        return 0
    from astra.test_cases import freeze_suite
    suite_builder = lambda spec: freeze_suite(reference, spec, folder, args)
    if args.test_spec:
        saved = json.loads(Path(args.test_spec).read_text(encoding="utf-8"))
        suite = suite_builder(saved.get("spec", saved))
        reference.metadata, baseline = suite["reference"], suite["baseline"]
    if args.evaluate_only:
        from astra.execution import execute
        candidate_dir = Path(args.evaluate_only).resolve()
        project = Path(__file__).resolve().parent
        if project not in candidate_dir.parents:
            parser.error("Candidate directory must be inside this project")
        source = candidate_dir / ("kernel.cu" if args.target == "cuda" else "kernel.cpp")
        backend.compile(source, candidate_dir)
        result = execute(folder, candidate_dir, "benchmark", args)
        (folder / "evaluation.json").write_text(json.dumps(result, indent=2))
        return 0 if result["complete"] else 1
    from astra.generation import LLMConfig, configure_llm, generate_candidate
    llm = LLMConfig(args.api_key, args.base_url, args.model, args.model_api,
                    args.enable_tracing, args.max_agent_turns, args.tls_max_version)
    configure_llm(llm)
    if args.generate_only:
        generate_candidate(task, reference.metadata, args.target, folder / "v1", llm,
                           compiler=backend.compile_feedback)
        return 0
    from astra.workflow import run
    from astra.reports import write_reports
    result = run(task, reference, baseline, folder, args, llm,
                 on_update=lambda payload: write_reports(payload, folder), suite_builder=suite_builder)
    summary = write_reports(result, folder)
    summary.update(status="completed" if summary["success"] else "failed", run_directory=str(folder))
    if args.result_file:
        from astra.reports import atomic_write
        atomic_write(Path(args.result_file), json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 1


def main():
    from astra.config import load_environment
    load_environment(Path(__file__).resolve().parent / ".env")
    parser = build_parser()
    args = parser.parse_args()
    if args.result_file:
        result_file = Path(args.result_file).resolve()
        project = Path(__file__).resolve().parent
        if project not in result_file.parents or not result_file.parent.is_dir():
            parser.error("Result file must be inside an existing project directory")
    try:
        return execute_task(args, parser)
    except BrokenPipeError:
        return 0
    except Exception as error:
        import traceback
        details = traceback.format_exc()
        if args.result_file:
            from astra.reports import atomic_write
            atomic_write(Path(args.result_file), json.dumps(dict(status="error", success=False, error=details), indent=2))
        print(details, file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

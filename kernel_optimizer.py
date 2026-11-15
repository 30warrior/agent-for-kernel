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
    parser.add_argument("--generate-only", action="store_true", help="Generate v1 sources from the task; makes LLM requests")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--model", default="o4-mini")
    parser.add_argument("--model-api", choices=("responses", "chat_completions"), default="responses")
    parser.add_argument("--enable-tracing", action="store_true", help="Opt in to SDK trace uploads")
    parser.add_argument("--max-agent-turns", type=int, default=12)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument("--evaluate-only", metavar="CANDIDATE_DIR", help="Compile and evaluate saved kernel/model.py sources without LLM requests")
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--cpu-core", type=int, help="CPU affinity; defaults to the first allowed core for CPU targets")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.list_tasks:
        for task in list_tasks(args.dataset, args.dataset_root):
            print(json.dumps(task.metadata(), ensure_ascii=False))
        return 0
    if not args.task:
        parser.error("--task is required unless --list-tasks is selected")
    task = load_task(args.dataset, args.dataset_root, args.task)
    if args.prepare_only or args.generate_only or args.evaluate_only:
        if args.target != "cuda" and hasattr(os, "sched_getaffinity"):
            allowed = os.sched_getaffinity(0)
            core = args.cpu_core if args.cpu_core is not None else min(allowed)
            if core not in allowed:
                parser.error("--cpu-core is outside this process's allowed affinity")
            os.sched_setaffinity(0, {core})
        from astra.reference import prepare_reference, save_reference
        from astra.timing import measure
        from astra.backends import Backend
        backend = Backend(args.target, args.compile_timeout)
        hardware = backend.capabilities()
        reference = prepare_reference(task, args.target, args.seed, args.threads)
        reference.metadata["hardware"] = hardware
        folder = Path(__file__).resolve().parent / "optimization_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_reference(reference, folder)
        baseline = [measure(reference.model, case, args.target, args.warmup, args.trials) for case in reference.cases]
        (folder / "baseline.json").write_text(json.dumps(baseline, indent=2))
        if args.generate_only:
            from astra.generation import LLMConfig, configure_llm, generate_candidate
            config = LLMConfig(args.api_key, args.base_url, args.model, args.model_api,
                               args.enable_tracing, args.max_agent_turns)
            configure_llm(config)
            generate_candidate(task, reference.metadata, args.target, folder / "v1", config, compiler=backend.compile_feedback)
        if args.evaluate_only:
            from astra.evaluation import evaluate
            candidate_dir = Path(args.evaluate_only).resolve()
            project = Path(__file__).resolve().parent
            if project not in candidate_dir.parents:
                parser.error("Candidate directory must be inside this project")
            source = candidate_dir / ("kernel.cu" if args.target == "cuda" else "kernel.cpp")
            artifact = backend.compile(source, candidate_dir)
            candidate = backend.load_candidate(artifact, candidate_dir / "model.py", reference.model)
            result = evaluate(reference, candidate, args.target, args.warmup, args.trials, args.rtol, args.atol)
            (folder / "evaluation.json").write_text(json.dumps(result, indent=2))
            print(folder)
            return 0 if result["complete"] else 1
        print(folder)
        return 0
    print(json.dumps(task.metadata(), indent=2, ensure_ascii=False))
    parser.error("Dataset execution is introduced in the following migration steps; no model request was made")


if __name__ == "__main__":
    raise SystemExit(main())

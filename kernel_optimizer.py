"""Dataset-driven Astra entry point; imports execution dependencies lazily."""
import argparse
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
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
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
    if args.prepare_only:
        from astra.reference import prepare_reference, save_reference
        from astra.timing import measure
        reference = prepare_reference(task, args.target, args.seed, args.threads)
        folder = Path(__file__).resolve().parent / "optimization_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_reference(reference, folder)
        baseline = [measure(reference.model, case, args.target, args.warmup, args.trials) for case in reference.cases]
        (folder / "baseline.json").write_text(json.dumps(baseline, indent=2))
        print(folder)
        return 0
    print(json.dumps(task.metadata(), indent=2, ensure_ascii=False))
    parser.error("Dataset execution is introduced in the following migration steps; no model request was made")


if __name__ == "__main__":
    raise SystemExit(main())

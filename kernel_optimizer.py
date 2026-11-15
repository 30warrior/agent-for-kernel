"""Dataset-driven Astra entry point; imports execution dependencies lazily."""
import argparse
import json
from astra.datasets import list_tasks, load_task


def build_parser():
    parser = argparse.ArgumentParser(description="Generate and optimize dataset kernels for CUDA, NEON or SVE")
    parser.add_argument("--dataset", choices=("akg", "kernelbench"), required=True)
    parser.add_argument("--dataset-root", required=True, help="AKG benchmark root or KernelBench task root")
    parser.add_argument("--task", help="Python task path relative to dataset-root")
    parser.add_argument("--target", choices=("cuda", "neon", "sve"), default="neon")
    parser.add_argument("--list-tasks", action="store_true", help="List definitions without importing tasks or running kernels")
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
    print(json.dumps(task.metadata(), indent=2, ensure_ascii=False))
    parser.error("Dataset execution is introduced in the following migration steps; no model request was made")


if __name__ == "__main__":
    raise SystemExit(main())

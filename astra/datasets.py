"""Read task definitions without importing PyTorch or executing dataset code."""
import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    dataset: str
    task_id: str
    path: Path
    source: str
    source_hash: str
    dynamic: bool

    def metadata(self):
        return dict(dataset=self.dataset, task_id=self.task_id,
                    source_path=str(self.path), source_sha256=self.source_hash,
                    dataset_root=str(self.path.parents[len(Path(self.task_id).parts)-1]),
                    dynamic=self.dynamic)


def load_task(dataset, dataset_root, task_id):
    root = Path(dataset_root).resolve(strict=True)
    path = (root / task_id).resolve(strict=True)
    if root not in path.parents or path.suffix != ".py":
        raise ValueError("Task must be a Python file inside dataset-root")
    if dataset not in ("akg", "kernelbench"):
        raise ValueError("Unsupported dataset: " + dataset)
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    dynamic = "get_inputs_dyn_list" in functions
    if "Model" not in classes or "get_init_inputs" not in functions:
        raise ValueError("Task needs Model and get_init_inputs")
    if not dynamic and "get_inputs" not in functions:
        raise ValueError("Task needs get_inputs or get_inputs_dyn_list")
    return Task(dataset, path.relative_to(root).as_posix(), path, source,
                hashlib.sha256(source.encode()).hexdigest(), dynamic)


def list_tasks(dataset, dataset_root):
    root = Path(dataset_root).resolve(strict=True)
    directories = ("static_shape", "dynamic_shape") if dataset == "akg" else ("level1", "level2", "level3")
    # Also support a root already pointing at one level/category of a dataset.
    roots = [root / name for name in directories if (root / name).is_dir()] or [root]
    seen = set()
    for directory in roots:
        for path in sorted(directory.rglob("*.py")):
            if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(root).parts):
                continue
            task_id = path.relative_to(root).as_posix()
            if task_id in seen:
                continue
            seen.add(task_id)
            try:
                yield load_task(dataset, root, task_id)
            except (ValueError, SyntaxError, UnicodeError):
                continue

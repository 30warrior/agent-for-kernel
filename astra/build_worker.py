"""Isolated compiler subprocess, launched only by Backend.compile."""
import json
import os
from pathlib import Path
import sys


def main():
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text())
    from torch.utils.cpp_extension import load
    cuda = request["target"] == "cuda"
    if not cuda:
        from .cpu_execution import restore_cpu_affinity
        restore_cpu_affinity()
    if cuda:
        import torch
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    module = load(name=request["name"], sources=[request["source"]],
                  build_directory=request["build_dir"],
                  extra_cflags=request["cflags"],
                  extra_ldflags=request.get("ldflags", []),
                  extra_cuda_cflags=["-O3", "-std=c++17"] if cuda else None,
                  with_cuda=cuda, verbose=True)
    artifact = dict(request, extension_path=str(Path(module.__file__).resolve()))
    (request_path.parent / "build.json").write_text(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()

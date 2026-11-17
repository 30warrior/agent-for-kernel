"""Native compilation and loading for CUDA, NEON and SVE targets."""
import copy
import hashlib
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path


class ExtensionHandle:
    """Share the immutable native module when candidate model state is copied."""
    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __deepcopy__(self, memo):
        return self


class Backend:
    def __init__(self, target, compile_timeout=300, threads=1):
        if target not in ("cuda", "neon", "sve"):
            raise ValueError("Unknown target: " + target)
        self.target = target
        self.compile_timeout = compile_timeout
        self.threads = threads

    def capabilities(self):
        import torch
        info = dict(target=self.target, machine=platform.machine(),
                    torch_version=torch.__version__, compiler=os.environ.get("CXX", "c++"))
        if self.target == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            device = torch.cuda.current_device()
            info.update(device=torch.cuda.get_device_name(device),
                        capability=list(torch.cuda.get_device_capability(device)))
        else:
            if platform.machine().lower() not in ("aarch64", "arm64"):
                raise RuntimeError("NEON/SVE candidates require a native AArch64 host")
            text = Path("/proc/cpuinfo").read_text().lower()
            features = next((line.split(":",1)[1].split() for line in text.splitlines() if line.startswith("features")), [])
            required = "sve" if self.target == "sve" else "asimd"
            if required not in features:
                raise RuntimeError("Host does not report " + required)
            info["features"] = features
            info["openmp_enabled"] = self.threads > 1
            vector = Path("/proc/sys/abi/sve_default_vector_length")
            if vector.exists():
                info["sve_default_vector_bytes"] = int(vector.read_text())
        try:
            result = subprocess.run([info["compiler"], "--version"], capture_output=True, text=True, timeout=10)
            info["compiler_version"] = result.stdout.splitlines()[0] if result.stdout else result.stderr
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("Compiler unavailable: " + str(error)) from error
        info["cflags"] = self.cflags()
        info["ldflags"] = self.ldflags()
        return info

    def cflags(self):
        flags = ["-O3", "-std=c++17"]
        if self.target == "neon":
            flags += ["-march=armv8-a+simd"]
        elif self.target == "sve":
            flags += ["-march=armv8.2-a+sve", "-msve-vector-bits=scalable"]
        if self.target != "cuda" and self.threads > 1:
            flags.append("-fopenmp")
        return flags

    def ldflags(self):
        return ["-fopenmp"] if self.target != "cuda" and self.threads > 1 else []

    def compile(self, source, folder):
        folder = Path(folder).resolve()
        source = Path(source).resolve()
        if source.parent != folder:
            raise ValueError("Source must belong to this candidate directory")
        info = self.capabilities()
        identity = source.read_bytes() + json.dumps(info, sort_keys=True).encode()
        name = "astra_kernel_" + hashlib.sha256(identity).hexdigest()[:20]
        build = folder / "build"
        build.mkdir(exist_ok=True)
        request = dict(name=name, source=str(source), build_dir=str(build),
                       target=self.target, cflags=self.cflags(), ldflags=self.ldflags(), hardware=info)
        request_path = folder / "build_request.json"
        request_path.write_text(json.dumps(request, indent=2))
        project = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["MAX_JOBS"] = "2"
        # Invoking a venv's Python directly does not activate its script PATH.
        # Include the base environment too when using --system-site-packages.
        env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent),
                                      str(Path(sys.base_prefix) / "bin"), env.get("PATH", "")])
        process = subprocess.Popen([sys.executable, "-B", "-m", "astra.build_worker", str(request_path)],
                                   cwd=project, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            output, _ = process.communicate(timeout=self.compile_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                output, _ = process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
            (folder / "build.log").write_text(output)
            raise RuntimeError("Compilation timed out; compiler process group terminated")
        (folder / "build.log").write_text(output)
        if process.returncode:
            raise RuntimeError("Compilation failed (see build.log):\n" + output[-24000:])
        artifact = json.loads((folder / "build.json").read_text())
        return artifact

    def compile_feedback(self, source, folder):
        try:
            artifact = self.compile(source, folder)
            return "Compiled successfully: " + artifact["extension_path"]
        except Exception as error:
            return "ERROR: " + str(error)

    def load_candidate(self, artifact, wrapper_path, reference):
        if self.target != "cuda":
            from .cpu_execution import restore_cpu_affinity
            restore_cpu_affinity()
        name = artifact["name"]
        extension = sys.modules.get(name)
        if extension is None:
            spec = importlib.util.spec_from_file_location(name, artifact["extension_path"])
            extension = importlib.util.module_from_spec(spec)
            sys.modules[name] = extension
            spec.loader.exec_module(extension)
        wrapper_path = Path(wrapper_path)
        module_name = name + "_wrapper_" + hashlib.sha256(str(wrapper_path).encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(module_name, wrapper_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        model = module.ModelNew(copy.deepcopy(reference), ExtensionHandle(extension))
        import torch
        if not isinstance(model, torch.nn.Module):
            raise TypeError("ModelNew must be a torch.nn.Module")
        return model.eval()

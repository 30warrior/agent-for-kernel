"""SSH transport: keep model clients local and execute native work remotely."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
from types import SimpleNamespace
import uuid


# Run before any project code exists on the worker. Reject links/path traversal.
UNPACK = """import pathlib,sys,tarfile
base=pathlib.Path(sys.argv[1]).resolve()
archive=pathlib.Path(sys.argv[2]).resolve()
if base not in archive.parents: raise ValueError('Archive outside workspace')
with tarfile.open(archive, 'r:gz') as bundle:
    members=bundle.getmembers()
    for item in members:
        target=(base/item.name).resolve()
        if base not in target.parents or not item.isfile():
            raise ValueError('Invalid archive entry: '+item.name)
    for item in members:
        target=base/item.name
        target.parent.mkdir(parents=True,exist_ok=True)
        with bundle.extractfile(item) as source, target.open('wb') as dest:
            import shutil
            shutil.copyfileobj(source,dest)
"""


def remote_path(value):
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or any(c in value for c in "\n\r\x00"):
        raise ValueError("Remote paths must be absolute POSIX paths without '..'")
    return path


class RemoteRunner:
    def __init__(self, settings, folder):
        self.settings = settings
        self.folder = Path(folder).resolve()
        self.transfer = self.folder / ".transfer"
        self.transfer.mkdir(parents=True, exist_ok=True)
        self.host = settings.ssh_host
        if not self.host or self.host.startswith("-") or any(c.isspace() for c in self.host):
            raise ValueError("Invalid SSH host/alias")
        root = remote_path(settings.remote_root)
        if len(root.parts) < 3:
            raise ValueError("Use a dedicated remote project directory")
        self.python = str(remote_path(settings.remote_python))
        self.workspace = str(root / "runs" / (self.folder.name + "_" + uuid.uuid4().hex[:12]))
        self.reference = None
        self.baseline = None
        self.artifacts = {}
        self._uploaded = {}

    def _ssh(self, command, label, timeout=60):
        options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                   "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
        try:
            result = subprocess.run(["ssh", *options, self.host, command],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                                    timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"SSH transport timed out during {label}; remote worker has its own deadline") from error
        (self.transfer / (label + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"SSH {label} failed ({result.returncode}):\n{(result.stdout + result.stderr)[-12000:]}")
        return result

    def _scp(self, local, remote, upload=True):
        # Restrict generated remote paths to shell-safe characters for legacy scp.
        # Our workspace is unique, and no credentials enter this command.
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-" for c in remote):
            raise ValueError("Remote transfer paths must use letters, digits, '_', '-', '.' and '/'")
        spec = self.host + ":" + remote
        command = ["scp", "-B", "-q", "-o", "ConnectTimeout=20"]
        command += [str(local), spec] if upload else [spec, str(local)]
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=self.settings.transfer_timeout)
        if result.returncode:
            raise RuntimeError("SCP failed: " + result.stderr[-12000:])

    def _upload_archive(self, archive, label):
        destination = self.workspace + "/" + archive.name
        self._scp(archive, destination)
        command = shlex.join([self.python, "-B", "-c", UNPACK, self.workspace, destination])
        self._ssh(command, label, self.settings.transfer_timeout)

    def prepare(self, task):
        project = Path(__file__).resolve().parent.parent
        dataset = Path(task.metadata()["dataset_root"]).resolve()
        archive = self.transfer / "payload.tar.gz"
        excluded = {".git", ".venv", "__pycache__", ".cache", "build", "optimization_runs", ".publish"}
        with tarfile.open(archive, "w:gz") as bundle:
            for source in sorted((project / "astra").glob("*.py")):
                bundle.add(source, "astra/" + source.name, recursive=False)
            for source in sorted(dataset.rglob("*")):
                relative = source.relative_to(dataset)
                if any(part.startswith(".") or part in excluded for part in relative.parts):
                    continue
                if source.is_symlink():
                    raise ValueError("Dataset symlinks are unsupported for remote transfer: " + str(source))
                if source.is_file():
                    bundle.add(source, "dataset/" + relative.as_posix(), recursive=False)
        create = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=False)"
        self._ssh(shlex.join([self.python, "-B", "-c", create, self.workspace]), "create_workspace")
        self._upload_archive(archive, "unpack_payload")
        response = self._request("prepare", dataset=task.dataset, task_id=task.task_id,
                                 source_sha256=task.source_hash)
        metadata = response["reference"]
        if metadata["source_sha256"] != task.source_hash:
            raise RuntimeError("Remote task source differs from the local task")
        self.reference = SimpleNamespace(metadata=metadata)
        self.baseline = response["baseline"]
        (self.folder / "reference.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (self.folder / "baseline.json").write_text(json.dumps(self.baseline, indent=2), encoding="utf-8")
        (self.folder / "remote.json").write_text(json.dumps(dict(host=self.host, workspace=self.workspace,
                    python=self.python, target=self.settings.target), indent=2), encoding="utf-8")
        return self.reference, self.baseline

    def _request(self, action, **extra):
        token = action + "_" + uuid.uuid4().hex
        request = dict(action=action, target=self.settings.target, seed=self.settings.seed,
                       threads=self.settings.threads, cpu_core=self.settings.cpu_core,
                       warmup=self.settings.warmup, trials=self.settings.trials,
                       rtol=self.settings.rtol, atol=self.settings.atol,
                       compile_timeout=self.settings.compile_timeout,
                       execution_timeout=self.settings.execution_timeout,
                       result_name=token + ".result.json", **extra)
        local = self.transfer / (token + ".request.json")
        local.write_text(json.dumps(request, indent=2), encoding="utf-8")
        remote = self.workspace + "/" + local.name
        self._scp(local, remote)
        env = ["env", "TORCH_DEVICE_BACKEND_AUTOLOAD=0", "PYTHONDONTWRITEBYTECODE=1",
               "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "XDG_CACHE_HOME=" + self.workspace + "/.cache"]
        command = "cd " + shlex.quote(self.workspace) + " && " + shlex.join(
            env + [self.python, "-B", "-m", "astra.remote_worker", local.name])
        deadline = self.settings.compile_timeout if action == "compile" else self.settings.execution_timeout
        self._ssh(command, token, deadline + 45)
        response_path = self.transfer / request["result_name"]
        self._scp(response_path, self.workspace + "/" + request["result_name"], upload=False)
        response = json.loads(response_path.read_text(encoding="utf-8"))
        for name, content in response.get("logs", {}).items():
            if Path(name).name != name:
                raise RuntimeError("Invalid worker log name")
            destination = self.folder / extra["candidate"] if "candidate" in extra else self.folder
            destination.mkdir(parents=True, exist_ok=True)
            (destination / ("remote_" + name)).write_text(content, encoding="utf-8")
        if not response.get("ok"):
            raise RuntimeError("Remote " + action + " failed:\n" + response.get("error", "Missing worker error"))
        return response["result"]

    def _candidate(self, folder):
        folder = Path(folder).resolve()
        relative = folder.relative_to(self.folder)
        if not relative.parts or any(part.startswith(".") for part in relative.parts):
            raise ValueError("Candidate must be inside the local run directory")
        return folder, relative.as_posix()

    def compile(self, source, folder):
        folder, name = self._candidate(folder)
        source = Path(source).resolve()
        suffix = "kernel.cu" if self.settings.target == "cuda" else "kernel.cpp"
        if source != folder / suffix:
            raise ValueError("Unexpected candidate source")
        wrapper = folder / "model.py"
        digest = hashlib.sha256(source.read_bytes() + b"\0" + wrapper.read_bytes()).hexdigest()
        if self._uploaded.get(name) == digest:
            return self.artifacts[name]
        # A failed replacement must not leave a cache entry for older remote files.
        self._uploaded.pop(name, None)
        self.artifacts.pop(name, None)
        archive = self.transfer / ("candidate_" + uuid.uuid4().hex + ".tar.gz")
        with tarfile.open(archive, "w:gz") as bundle:
            for path in (source, wrapper):
                bundle.add(path, "candidates/" + name + "/" + path.name, recursive=False)
        self._upload_archive(archive, "unpack_" + archive.stem)
        artifact = self._request("compile", candidate=name)
        self._uploaded[name], self.artifacts[name] = digest, artifact
        (folder / "remote_build.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return artifact

    def compile_feedback(self, source, folder):
        try:
            artifact = self.compile(source, folder)
            return "Compiled on " + self.host + ": " + artifact["extension_path"]
        except Exception as error:
            return "ERROR: " + str(error)

    def execute(self, run_dir, candidate_dir, mode, settings):
        if Path(run_dir).resolve() != self.folder or mode not in ("verify", "benchmark"):
            raise ValueError("Invalid remote evaluation request")
        folder, name = self._candidate(candidate_dir)
        source = folder / ("kernel.cu" if self.settings.target == "cuda" else "kernel.cpp")
        self.compile(source, folder)
        result = self._request(mode, candidate=name)
        (folder / (mode + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

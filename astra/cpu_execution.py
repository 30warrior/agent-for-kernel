"""One CPU resource budget shared by reference, native kernel and worker processes."""
import ctypes
import os
from pathlib import Path


def parse_cpu_list(value):
    cpus = set()
    try:
        for item in value.split(","):
            bounds = item.strip().split("-")
            if len(bounds) == 1:
                first = last = int(bounds[0])
            elif len(bounds) == 2:
                first, last = map(int, bounds)
            else:
                raise ValueError()
            if first < 0 or last < first or last - first > 65536:
                raise ValueError()
            cpus.update(range(first, last + 1))
    except (ValueError, AttributeError):
        raise ValueError("--cpu-cores expects CPU IDs/ranges, for example 0,2,4,6 or 0-3") from None
    if not cpus:
        raise ValueError("--cpu-cores must not be empty")
    return sorted(cpus)


def cpu_topology(cpu):
    path = Path("/sys/devices/system/cpu") / f"cpu{cpu}"
    def read(name):
        try:
            return int((path / "topology" / name).read_text().strip())
        except (OSError, ValueError):
            return None
    nodes = sorted(int(node.name[4:]) for node in path.glob("node[0-9]*"))
    return dict(cpu=cpu, package=read("physical_package_id"), core=read("core_id"),
                numa_node=nodes[0] if nodes else None)


def select_cpu_execution(threads=None, cpu_core=None, cpu_cores=None):
    if not hasattr(os, "sched_getaffinity"):
        raise ValueError("CPU execution requires Linux CPU affinity support")
    allowed = set(os.sched_getaffinity(0))
    if cpu_core is not None and cpu_cores is not None:
        raise ValueError("Use either --cpu-core or --cpu-cores")
    if cpu_cores is not None:
        selected = parse_cpu_list(cpu_cores)
        if threads is None:
            threads = len(selected)
        if len(selected) != threads:
            raise ValueError("--cpu-cores must contain exactly --threads logical CPU IDs")
        if not set(selected) <= allowed:
            raise ValueError("--cpu-cores contains CPUs outside this process's allowed affinity")
        policy = "explicit"
    else:
        if threads is None:
            # Keep automatic runs bounded while still using multiple physical
            # cores for large cases. Per-case plans may select fewer cores.
            threads = min(8, len({(cpu_topology(cpu)["package"], cpu_topology(cpu)["core"]) for cpu in allowed}))
            threads = max(1, threads)
        anchor = min(allowed) if cpu_core is None else cpu_core
        if anchor not in allowed:
            raise ValueError("--cpu-core is outside this process's allowed affinity")
        topology = {cpu: cpu_topology(cpu) for cpu in allowed}
        base = topology[anchor]
        selected, seen = [], set()
        for cpu in [anchor] + sorted(allowed - {anchor}):
            info = topology[cpu]
            if info["numa_node"] != base["numa_node"]:
                continue
            if base["numa_node"] is None and info["package"] != base["package"]:
                continue
            key = (info["package"], info["core"]) if info["core"] is not None else ("cpu", cpu)
            if key not in seen:
                selected.append(cpu)
                seen.add(key)
            if len(selected) == threads:
                break
        if len(selected) != threads:
            raise ValueError(f"Requested {threads} threads, but only {len(selected)} distinct physical cores "
                             "are available in the selected NUMA node/package. Use fewer threads or explicit --cpu-cores.")
        policy = "auto_physical_same_numa"
    topology = [cpu_topology(cpu) for cpu in selected]
    environment = dict(OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads),
        OMP_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1", OMP_PROC_BIND="close",
        OMP_PLACES=",".join("{" + str(cpu) + "}" for cpu in selected),
        MKL_NUM_THREADS=str(threads), MKL_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS=str(threads),
        BLIS_NUM_THREADS=str(threads), ASTRA_CPU_AFFINITY=",".join(map(str, selected)))
    return dict(threads=threads, affinity=selected, affinity_policy=policy, topology=topology,
        numa_nodes=sorted({info["numa_node"] for info in topology if info["numa_node"] is not None}),
        torch_interop_threads=1, openmp_enabled=threads > 1, environment=environment)


def value_nbytes(value):
    """Approximate tensor storage used to choose a case's CPU thread budget."""
    if hasattr(value, 'numel') and hasattr(value, 'element_size'):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(value_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(value_nbytes(item) for item in value)
    return 0


def threads_for_bytes(nbytes, maximum):
    if maximum <= 1 or nbytes <= 4 * 2**20:
        return 1
    if nbytes <= 32 * 2**20:
        return min(2, maximum)
    if nbytes <= 256 * 2**20:
        return min(4, maximum)
    return maximum


def build_thread_plan(cases, maximum):
    return [dict(case_index=index, input_bytes=value_nbytes(case),
                 threads=threads_for_bytes(value_nbytes(case), maximum))
            for index, case in enumerate(cases)]


def apply_case_execution(configuration, threads, torch=None):
    """Apply the same case-specific affinity/thread count to baseline and candidate."""
    threads = max(1, min(int(threads), int(configuration['threads'])))
    affinity = list(configuration['affinity'][:threads])
    os.sched_setaffinity(0, set(affinity))
    environment = dict(configuration['environment'])
    environment.update(OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads),
                       MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                       BLIS_NUM_THREADS=str(threads),
                       OMP_PLACES=','.join('{' + str(cpu) + '}' for cpu in affinity),
                       ASTRA_CPU_AFFINITY=','.join(map(str, affinity)))
    os.environ.update(environment)
    # OpenMP runtimes are often loaded before the per-case environment changes;
    # set their active team size explicitly when the runtime is available.
    for library in ('libgomp.so.1', 'libomp.so', 'libiomp5.so'):
        try:
            runtime = ctypes.CDLL(library)
            runtime.omp_set_num_threads.argtypes = [ctypes.c_int]
            runtime.omp_set_num_threads(threads)
            break
        except (OSError, AttributeError):
            continue
    if torch is not None:
        configure_torch_threads(torch, threads)
    return dict(threads=threads, affinity=affinity, environment=environment)


def apply_cpu_execution(configuration, environment=None):
    """Apply before importing torch/OpenMP; a worker may inherit a master bound to one CPU."""
    env = os.environ if environment is None else environment
    # Avoid inherited runtime-specific bindings overriding the explicit common places.
    for name in ("GOMP_CPU_AFFINITY", "KMP_AFFINITY", "KMP_HW_SUBSET", "OMP_NESTED"):
        env.pop(name, None)
    env.update(configuration["environment"])
    if environment is None:
        requested = set(configuration["affinity"])
        os.sched_setaffinity(0, requested)
        if set(os.sched_getaffinity(0)) != requested:
            raise RuntimeError("Could not apply the complete saved CPU affinity")


def configure_torch_threads(torch, threads):
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def restore_cpu_affinity():
    """Let a newly loaded OpenMP runtime see the whole budget, not just torch's master place."""
    value = os.environ.get("ASTRA_CPU_AFFINITY")
    if value:
        requested = set(parse_cpu_list(value))
        os.sched_setaffinity(0, requested)
        if set(os.sched_getaffinity(0)) != requested:
            raise RuntimeError("Could not restore the complete CPU affinity before loading an extension")


def runtime_snapshot(torch, configuration):
    return dict(configuration,
        torch_num_threads=torch.get_num_threads(), torch_num_interop_threads=torch.get_num_interop_threads(),
        main_thread_affinity=sorted(os.sched_getaffinity(0)),
        runtime_environment={name: os.environ.get(name) for name in configuration["environment"]})

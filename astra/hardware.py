"""Read-only Linux CPU facts; unavailable specifications remain unknown."""
import ctypes
from collections import Counter
import json
import os
from pathlib import Path
import platform
import re
import subprocess

from .cpu_execution import cpu_topology, parse_cpu_list


def _read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def _integer(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _cpus(value):
    if not value:
        return []
    try:
        return parse_cpu_list(value)
    except ValueError:
        return []


def _bytes(value):
    match = re.fullmatch(r"(\d+)\s*([KMGT]?)(?:i?B)?", value or "", re.IGNORECASE)
    if not match:
        return None
    exponent = {'': 0, 'K': 1, 'M': 2, 'G': 3, 'T': 4}[match[2].upper()]
    return int(match[1]) * 1024 ** exponent


def _cpuinfo():
    records = {}
    for block in (_read('/proc/cpuinfo') or '').split('\n\n'):
        fields = {key.strip().lower(): value.strip() for line in block.splitlines()
                  if ':' in line for key, value in [line.split(':', 1)]}
        number = _integer(fields.get('processor'))
        if number is not None:
            records[number] = fields
    return records


def _lscpu():
    try:
        result = subprocess.run(['lscpu', '--json'], env=dict(os.environ, LC_ALL='C'),
                                capture_output=True, text=True, timeout=5, check=True)
        return {row['field'].rstrip(':').lower(): row['data'] for row in json.loads(result.stdout)['lscpu']}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return {}


def _supplement():
    path = Path(__file__).resolve().parent.parent / 'cpu_hardware.json'
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or set(data) - {'cpu_identity', 'source'}:
        raise ValueError('cpu_hardware.json accepts only cpu_identity and source')
    identity = data.get('cpu_identity')
    if not isinstance(identity, dict) or set(identity) - {'model_name'}:
        raise ValueError('cpu_hardware.json cpu_identity accepts only model_name')
    if not isinstance(identity.get('model_name'), str) or not identity['model_name'].strip():
        raise ValueError('cpu_hardware.json requires a nonempty model_name')
    if not isinstance(data.get('source'), str) or not data['source'].strip():
        raise ValueError('cpu_hardware.json requires a source for the supplied model name')
    return data


def _sve(features):
    current, error = None, None
    if 'sve' in features and platform.machine().lower() in ('aarch64', 'arm64'):
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
            libc.prctl.restype = ctypes.c_int
            # Linux UAPI: PR_SVE_GET_VL=51, PR_SVE_VL_LEN_MASK=0xffff. Never change VL here.
            value = libc.prctl(51, 0, 0, 0, 0)
            if value < 0:
                error = 'PR_SVE_GET_VL unavailable: ' + os.strerror(ctypes.get_errno())
            else:
                current = value & 0xffff
        except (OSError, AttributeError) as failure:
            error = str(failure)
    return dict(supported='sve' in features, current_thread_vector_bytes=current,
                current_thread_vector_bits=current * 8 if current else None,
                system_default_vector_bytes=_integer(_read('/proc/sys/abi/sve_default_vector_length')),
                source='prctl(PR_SVE_GET_VL); /proc/sys/abi/sve_default_vector_length',
                query_error=error,
                note='Observed VL is diagnostic only. Generate vector-length-agnostic SVE; worker threads may have different VL.')


def _caches(selected):
    caches, missing = {}, []
    for cpu in selected:
        entries = sorted((Path('/sys/devices/system/cpu') / f'cpu{cpu}/cache').glob('index[0-9]*'))
        if not entries:
            missing.append(cpu)
        for path in entries:
            sharing = _read(path / 'shared_cpu_list')
            level, kind = _integer(_read(path / 'level')), _read(path / 'type')
            # Deduplicate shared instances, never sum whole-machine cache totals as per-core capacity.
            key = (level, kind, sharing if sharing else f'cpu{cpu}/{path.name}')
            if key not in caches:
                shared = _cpus(sharing)
                caches[key] = dict(level=level, type=kind, size_bytes=_bytes(_read(path / 'size')),
                    coherency_line_bytes=_integer(_read(path / 'coherency_line_size')),
                    associativity_ways=_integer(_read(path / 'ways_of_associativity')),
                    shared_cpu_list=sharing, shared_logical_cpu_count=len(shared) if shared else None,
                    selected_cpus_using_instance=[], source='Linux sysfs cpu/cache')
            caches[key]['selected_cpus_using_instance'].append(cpu)
    return dict(instances=list(caches.values()), unavailable_for_cpus=missing,
                scope='Cache instances used by allocated CPUs; size_bytes is capacity per shared instance, not reserved capacity.')


def collect_cpu_hardware():
    """Capture the allocated CPUs even after an OpenMP runtime binds its master to one CPU."""
    selected = _cpus(os.environ.get('ASTRA_CPU_AFFINITY')) or sorted(os.sched_getaffinity(0))
    records, reported = _cpuinfo(), _lscpu()
    sample = records.get(selected[0], {})
    supplement = _supplement()
    reported_model = reported.get('model name') or sample.get('model name')
    supplied_model = supplement.get('cpu_identity', {}).get('model_name')
    feature_sets = [set(records.get(cpu, {}).get('features', '').split()) for cpu in selected]
    features = sorted(set.intersection(*feature_sets)) if feature_sets else []
    identities = []
    for cpu in selected:
        record = records.get(cpu, {})
        identity = {name: record.get('cpu ' + name) for name in ('implementer', 'architecture', 'variant', 'part', 'revision')}
        identity['midr_el1'] = _read(Path('/sys/devices/system/cpu') / f'cpu{cpu}/regs/identification/midr_el1')
        if identity not in identities:
            identities.append(identity)
    online = _cpus(_read('/sys/devices/system/cpu/online'))
    topology = [cpu_topology(cpu) for cpu in online]
    known_cores = {(row['package'], row['core']) for row in topology if row['package'] is not None and row['core'] is not None}
    packages = {row['package'] for row in topology if row['package'] is not None}
    complete_topology = bool(topology) and all(row['core'] is not None and row['package'] is not None for row in topology)
    core_widths = Counter((row['package'], row['core']) for row in topology) if complete_topology else None
    node_count = len(list(Path('/sys/devices/system/node').glob('node[0-9]*')))
    online_node_ids = _cpus(_read('/sys/devices/system/node/online'))
    selected_topology = [cpu_topology(cpu) for cpu in selected]
    nodes = []
    for node in sorted({row['numa_node'] for row in selected_topology if row['numa_node'] is not None}):
        path = Path('/sys/devices/system/node') / f'node{node}'
        memory = re.search(r'MemTotal:\s*(\d+)\s*kB', _read(path / 'meminfo') or '')
        distances = [_integer(n) for n in (_read(path / 'distance') or '').split()]
        nodes.append(dict(node=node, cpu_list=_read(path / 'cpulist'),
            memory_total_bytes=int(memory[1]) * 1024 if memory else None,
            distances_to_online_nodes=dict(zip(map(str, online_node_ids), distances)) if online_node_ids and len(distances) == len(online_node_ids) else None,
            source='Linux sysfs node'))
    host_memory = re.search(r'^MemTotal:\s*(\d+)\s*kB', _read('/proc/meminfo') or '', re.MULTILINE)
    frequency = Path('/sys/devices/system/cpu') / f'cpu{selected[0]}/cpufreq'
    return dict(
        cpu_identity=dict(model_name=reported_model or supplied_model,
            model_name_os_reported=reported_model, model_name_user_supplied=supplied_model,
            model_name_source='OS-reported' if reported_model else supplement.get('source'),
            vendor=reported.get('vendor id'), architecture=platform.machine(),
            identification_groups=identities, source='lscpu --json; /proc/cpuinfo; sysfs MIDR',
            note='OS and explicitly supplied model names retain their provenance. No product name is inferred from an SSH alias or CPU part.'),
        topology=dict(online_logical_cpus=len(online) if online else None,
            online_physical_cores=len(known_cores) if complete_topology else None,
            online_packages=len(packages) if complete_topology else None,
            online_threads_per_physical_core=sorted(set(core_widths.values())) if core_widths else None,
            online_numa_node_count=node_count or None,
            allocated_logical_cpus=selected, allocated_topology=selected_topology,
            source='Linux sysfs cpu/topology and node'),
        caches=_caches(selected),
        isa=dict(host_features_common_to_allocated_cpus=features, neon_vector_bits=128 if 'asimd' in features else None,
            sve=_sve(features), source='/proc/cpuinfo features intersection across allocated CPUs'),
        frequency=dict(representative_cpu=selected[0],
            hardware_min_khz=_integer(_read(frequency / 'cpuinfo_min_freq')),
            hardware_max_khz=_integer(_read(frequency / 'cpuinfo_max_freq')),
            source='Linux sysfs cpufreq; unknown if unavailable; no current clock or throughput is inferred'),
        memory=dict(host_total_bytes=int(host_memory[1]) * 1024 if host_memory else None,
            allocated_numa_nodes=nodes, source='/proc/meminfo; Linux sysfs node',
            note='Host/node totals are not the job memory limit, available memory or an exclusive allocation.'),
        performance_limits=dict(peak_flops=None, peak_memory_bandwidth_bytes_per_second=None,
            note='Not measured or inferred by hardware discovery. Kernel timings belong to profiling results.'))


def hardware_context(hardware):
    """Compact repeated cache instances while keeping their sharing and per-instance sizes explicit."""
    import copy
    result = copy.deepcopy(hardware)
    if 'cpu' not in result:
        return result
    caches = result['cpu']['caches']
    if 'instances' not in caches:
        return result  # Coding receives an execution context already prepared by the workflow.
    groups = {}
    for cache in caches.get('instances', []):
        key = tuple(cache.get(field) for field in ('level', 'type', 'size_bytes', 'coherency_line_bytes',
                                                  'associativity_ways', 'shared_logical_cpu_count'))
        if key not in groups:
            groups[key] = {field: cache.get(field) for field in ('level', 'type', 'size_bytes',
                'coherency_line_bytes', 'associativity_ways', 'shared_logical_cpu_count')}
            groups[key].update(instance_count_used_by_allocation=0, sharing_groups=[])
        groups[key]['instance_count_used_by_allocation'] += 1
        groups[key]['sharing_groups'].append(dict(shared_cpu_list=cache['shared_cpu_list'],
            allocated_cpus=cache['selected_cpus_using_instance']))
    result['cpu']['caches'] = dict(groups=list(groups.values()), scope=caches['scope'],
                                   unavailable_for_cpus=caches['unavailable_for_cpus'])
    result['interpretation'] = ('CPU specs describe the observed host. Use only compilation_target.allowed_isa and supplied compiler flags; '
        'host feature presence does not enable an instruction in this build. Respect cpu_execution thread/affinity limits. '
        'Shared caches are not exclusive; use per-instance capacities for tiling. Null means unknown. '
        'Observed SVE VL must not be hard-coded. Performance measurements come from profiling, not these specifications.')
    return result

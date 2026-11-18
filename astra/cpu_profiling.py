"""Independent perf-stat sessions for each frozen CPU input case."""
import copy
import json
import math
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import time

EVENTS = ('cycles:u', 'instructions:u', 'branches:u', 'branch-misses:u',
          'cache-references:u', 'cache-misses:u', 'task-clock', 'context-switches')


def parse_stat(text, repeats):
    counters = {}
    for line in text.splitlines():
        fields = [v.strip() for v in line.split(';')]
        if len(fields) < 3 or fields[2] not in EVENTS:
            continue
        value, unit, event = fields[:3]
        def number(value):
            try:
                result = float(value)
                return result if math.isfinite(result) else None
            except ValueError:
                return None
        count = number(value)
        counters[event] = dict(status='ok' if count is not None else value.strip('<> ').replace(' ', '_'),
            count=count, per_call=count / repeats if count is not None else None, unit=unit or 'count',
            counter_runtime_ns=number(fields[3]) if len(fields) > 3 else None,
            percent_running=number(fields[4].rstrip('%')) if len(fields) > 4 else None,
            scaled_by_perf=True)
    for event in EVENTS:
        counters.setdefault(event, dict(status='missing', count=None, per_call=None, unit=None,
                                       percent_running=None))
    cycles, instructions = counters['cycles:u']['count'], counters['instructions:u']['count']
    ipc = instructions / cycles if cycles and instructions is not None else None
    return counters, ipc


def _tensor_bytes(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _command(process, descriptor, acknowledgement, command, timeout=10):
    os.write(descriptor, (command + '\n').encode())
    ready, _, _ = select.select([acknowledgement], [], [], timeout)
    reply = os.read(acknowledgement, 64) if ready else b''
    if reply.rstrip(b'\x00\r\n ') != b'ack':
        raise RuntimeError(f'perf control {command} did not acknowledge: {reply!r}; exit={process.poll()}')


def profile_case(model, case, expected, destination, warmup, repeats, batch_mib):
    import torch
    from .reference import clone_inputs, describe
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    raw, log = destination / 'perf_stat.csv', destination / 'perf.log'
    result = dict(status='unavailable', repeats=repeats, inputs=describe(case), events=list(EVENTS),
        counters={}, ipc=None, raw_path=str(raw), log_path=str(log),
        scope='Model.forward including output allocation, Python loop/result retention and perf control boundary overhead; excludes model/input preparation, warmup, loading and compilation',
        thread_scope='perf -p enumerates target process threads; default inheritance includes new workers; counters sum participating threads',
        interpretation='Generic cache events are architecture dependent, not identified L1/L2/L3 events. IPC under multiplexing is approximate. This is a separate diagnostic run, not the ranking timer.')
    executable = shutil.which('perf')
    if executable is None:
        result['error'] = 'perf executable not found'
        return result
    state = copy.deepcopy(model.state_dict())
    process = None
    descriptors = []
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                model.load_state_dict(state)
                output = model(*clone_inputs(case))
                del output
            model.load_state_dict(state)
            # Inputs, state and retained outputs all contribute to the batching budget.
            per_call_bytes = _tensor_bytes(case) + _tensor_bytes(state) + _tensor_bytes(expected)
            batch_size = min(repeats, max(1, int(batch_mib * 2**20) // max(1, per_call_bytes)))
            result.update(batch_size=batch_size, preparation_bytes_per_call=per_call_bytes,
                          threads_at_attach=sorted(int(p.name) for p in Path('/proc/self/task').iterdir()))
            ctl_read, ctl_write = os.pipe()
            ack_read, ack_write = os.pipe()
            descriptors = [ctl_read, ctl_write, ack_read, ack_write]
            command = [executable, 'stat', '-x', ';', '-o', str(raw), '-D', '-1',
                       '--control', f'fd:{ctl_read},{ack_write}', '-e', ','.join(EVENTS), '-p', str(os.getpid())]
            with log.open('w') as stream:
                process = subprocess.Popen(command, pass_fds=(ctl_read, ack_write),
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stream,
                    env=dict(os.environ, LC_ALL='C'))
                os.close(ctl_read)
                os.close(ack_write)
                descriptors = [ctl_write, ack_read]
                _command(process, ctl_write, ack_read, 'disable')
                wall, completed = 0.0, 0
                while completed < repeats:
                    count = min(batch_size, repeats - completed)
                    prepared = [(copy.deepcopy(model), clone_inputs(case)) for _ in range(count)]
                    outputs = []
                    _command(process, ctl_write, ack_read, 'enable')
                    started = time.perf_counter()
                    for instance, inputs in prepared:
                        outputs.append(instance(*inputs))
                    wall += time.perf_counter() - started
                    _command(process, ctl_write, ack_read, 'disable')
                    outputs.clear()
                    prepared.clear()
                    completed += count
                process.send_signal(signal.SIGINT)
                process.wait(timeout=5)
            counters, ipc = parse_stat(raw.read_text(), repeats)
            supported = sum(c['status'] == 'ok' for c in counters.values())
            result.update(status='ok' if supported == len(EVENTS) else 'partial' if supported else 'unavailable',
                          counters=counters, ipc=ipc, completed_calls=completed, loop_wall_seconds=wall,
                          perf_exit_code=process.returncode)
    except Exception as error:
        result['error'] = str(error)
        if log.exists():
            result['diagnostic'] = log.read_text(errors='replace')[-2500:]
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for descriptor in descriptors:
            os.close(descriptor)
        model.load_state_dict(state)
    return result


def profile_suite(reference, model, destination, warmup=5, repeats=64, batch_mib=64):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    labels = reference.metadata['case_labels']
    results = []
    for index, (case, expected, label) in enumerate(zip(reference.cases, reference.expected, labels)):
        result = profile_case(copy.deepcopy(model), case, expected, destination / f'case_{index:04d}',
                              warmup, repeats, batch_mib)
        result.update(case_id=index, **label)
        (destination / f'case_{index:04d}' / 'perf.json').write_text(json.dumps(result, indent=2))
        results.append(result)
    return dict(status='ok' if results and all(r['status'] == 'ok' for r in results) else 'partial' if any(r['status'] in ('ok', 'partial') for r in results) else 'unavailable',
                suite_id=reference.metadata['suite_id'], events=list(EVENTS), repeats=repeats,
                batch_mib=batch_mib, warmup=warmup, cases=results,
                cpu_execution=reference.metadata.get('cpu_execution'),
                protocol='fresh perf-stat session per case; counters disabled during preparation; same case repeated within each session')


def perf_context(profile):
    """Keep per-case attribution without sending raw totals/logs to the model."""
    return dict(status=profile['status'], repeats=profile['repeats'], protocol=profile['protocol'],
        interpretation='per_call counters normalized by repeats; compare same case and event only. Unsupported/missing counts are null. Counts include wrapper/loop/control overhead. No cache level or causal bottleneck is inferred.',
        cases=[dict(case_id=c['case_id'], name=c['name'], origin=c['origin'], status=c['status'],
            ipc=c['ipc'], batch_size=c.get('batch_size'), error=c.get('error'),
            counters={event:dict(status=v['status'], per_call=v['per_call'], unit=v['unit'],
                                percent_running=v.get('percent_running')) for event,v in c['counters'].items()})
               for c in profile['cases']])

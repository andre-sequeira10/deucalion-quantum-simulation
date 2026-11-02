#!/usr/bin/env python3
"""Fresh-process, statevector gradient memory measurement (Linux compute nodes)."""
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time


def proc_status():
    data = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM', 'VmSize', 'Threads'):
            data[key] = int(value.split()[0]) * (1024 if key != 'Threads' else 1)
    return data


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qubits', type=int, required=True)
    ap.add_argument('--groups', type=int, default=20)
    ap.add_argument('--method', choices=['backprop', 'adjoint', 'parameter-shift', 'forward'], required=True)
    ap.add_argument('--repeat', type=int, default=0)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    assert 2 <= args.qubits <= 20 and args.groups in (10, 20, 40)
    affinity_before = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {affinity_before[0]})
    import numpy as np
    import torch
    import pennylane as qml
    from threadpoolctl import threadpool_info
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Each group is exactly five elementary gates and three independent angles.
    # All methods use the same device family, wires, observable and angles.
    def make_node(n, groups, method):
        dev = qml.device('default.qubit', wires=n, shots=None, max_workers=None)
        kwargs = dict(interface='torch', diff_method=None if method == 'forward' else method,
                      cache=False, max_diff=1)
        if method == 'parameter-shift':
            kwargs['gradient_kwargs'] = {'broadcast': False}
        @qml.qnode(dev, **kwargs)
        def circuit(theta):
            for i in range(groups):
                a, b = i % n, (i + 1) % n
                qml.RY(theta[3*i], wires=a)
                qml.RX(theta[3*i+1], wires=b)
                qml.CNOT(wires=[a, b])
                qml.RY(theta[3*i+2], wires=b)
                qml.CNOT(wires=[b, a])
            return qml.expval(qml.PauliZ(0))
        return dev, circuit

    # Tiny, method-matched warmup loads lazy derivative machinery, never a large state.
    wd, warm = make_node(2, 2, args.method)
    wt = torch.tensor(np.random.default_rng(20261102).uniform(-1, 1, 6), requires_grad=args.method != 'forward')
    wy = warm(wt)
    if args.method != 'forward':
        wy.backward()
    del wd, warm, wt, wy
    # Record actual state precision outside the measured region using a tiny probe.
    pd = qml.device('default.qubit', wires=2, shots=None)
    @qml.qnode(pd, interface='torch', diff_method='backprop')
    def precision_probe(x):
        qml.RY(x, 0)
        return qml.state()
    probe = precision_probe(torch.tensor(0.1, requires_grad=True))
    assert probe.dtype == torch.complex128
    del probe, pd, precision_probe
    dev, circuit = make_node(args.qubits, args.groups, args.method)
    angles = np.random.default_rng(20261102).uniform(-1, 1, 3*args.groups)
    theta = torch.tensor(angles, requires_grad=args.method != 'forward')
    circuit.construct((theta,), {})
    tape = circuit._tape
    assert len(tape.operations) == 5*args.groups
    assert len(tape.trainable_params) == (3*args.groups if args.method != 'forward' else 0)
    operation_names = [op.name for op in tape.operations]
    del tape
    gc.collect()
    libc = ctypes.CDLL(None)
    trim_available = hasattr(libc, 'malloc_trim')
    if trim_available:
        libc.malloc_trim(0)
    before_reset = proc_status()
    # Linux clear_refs=5 resets ONLY this process's RSS high-water mark.
    # It neither drops page caches nor changes any other process.
    try:
        Path('/proc/self/clear_refs').write_text('5\n')
        reset_error = None
    except OSError as e:
        reset_error = str(e)
    baseline = proc_status()
    wall0, cpu0 = time.perf_counter(), time.process_time()
    with qml.Tracker(dev) as tracker:
        value = circuit(theta)
        forward_status = proc_status()
        if args.method != 'forward':
            value.backward()
    elapsed, cpu = time.perf_counter()-wall0, time.process_time()-cpu0
    end = proc_status()
    grad = None if args.method == 'forward' else theta.grad.detach().numpy().copy()
    assert np.isfinite(float(value.detach()))
    if grad is not None:
        assert grad.shape == (3*args.groups,) and np.isfinite(grad).all()
        assert float(np.linalg.norm(grad)) > 1e-12
    pools = threadpool_info()
    assert all(p['num_threads'] == 1 for p in pools), pools
    result = dict(
        schema=1, qubits=args.qubits, groups=args.groups, gates=5*args.groups,
        parameters=3*args.groups, method=args.method, repeat=args.repeat,
        backend='default.qubit', interface='torch', state_dtype='complex128',
        statevector_bytes=16*2**args.qubits, seed=20261102,
        angles_sha256=hashlib.sha256(angles.tobytes()).hexdigest(),
        value=float(value.detach()), gradient=None if grad is None else grad.tolist(),
        elapsed_seconds=elapsed, cpu_seconds=cpu,
        baseline=baseline, before_reset=before_reset, after_forward=forward_status, end=end,
        rss_peak_bytes=end['VmHWM'], incremental_peak_bytes=max(0,end['VmHWM']-baseline['VmRSS']),
        hwm_reset_error=reset_error, malloc_trim_available=trim_available,
        lifetime_ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        tracker_totals=tracker.totals, operation_names=operation_names,
        affinity_before=affinity_before, affinity=sorted(os.sched_getaffinity(0)),
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        threadpools=pools, versions=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__, pennylane=qml.__version__),
        host=platform.node(), machine=platform.machine(), job_id=os.environ.get('SLURM_JOB_ID'),
        source_sha256=digest(__file__), protocol_sha256=digest('protocol.json'))
    Path(args.output).write_text(json.dumps(result, indent=2, default=lambda x: x.item())+'\n')
    print(json.dumps({k: result[k] for k in ('qubits','groups','method','repeat','rss_peak_bytes','incremental_peak_bytes','elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()

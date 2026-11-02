"""Four-action parity-policy variance from a full MPI-distributed product state.

The state is a tensor product of RY(theta_i)|0>, not a compressed simulation.
Two disjoint wire-group parities define actions 00, 01, 10, 11. Three Pauli-Z
expectations reconstruct their exact Born probabilities without gathering the
statevector. Independent analytic probabilities validate every sample.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import time
import traceback

from mpi4py import MPI
import numpy as np
import pennylane as qml

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()
FIELDS = ['sample', 'p00', 'p01', 'p10', 'p11', 'exact00', 'exact01',
          'exact10', 'exact11', 'max_abs_error', 'seconds', 'max_rank_rss_kib']


def exact_policy(theta):
    """Independent product-state expression; groups are even and odd wires."""
    even = np.prod(np.cos(theta[0::2]))
    odd = np.prod(np.cos(theta[1::2]))
    return np.array([(1 + se * even) * (1 + so * odd) / 4
                     for se, so in [(1, 1), (1, -1), (-1, 1), (-1, -1)]])


def reconstruct(moments):
    even, odd, both = moments
    return np.array([(1 + se * even + so * odd + se * so * both) / 4
                     for se, so in [(1, 1), (1, -1), (-1, 1), (-1, -1)]])


def atomic_json(path, data):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qubits', type=int, required=True)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reference-check', action='store_true')
    args = parser.parse_args()
    n = args.qubits
    if n < 2 or args.samples < 2 or args.checkpoint_every < 1:
        raise ValueError('Require qubits>=2, samples>=2 and checkpoint-every>=1')
    if SIZE & (SIZE - 1) or SIZE > 2**n:
        raise ValueError('MPI size must be a power of two and <= 2**qubits')
    if args.reference_check and n > 16:
        raise ValueError('Replicated reference checks are limited to 16 qubits')
    if not args.reference_check and platform.machine() != 'aarch64':
        raise RuntimeError('Production runs require a Deucalion ARM compute node')
    identity = dict(qubits=n, samples=args.samples, seed=args.seed,
                    initialization='independent Uniform(-pi,pi) RY angles',
                    actions=['even/even', 'even/odd', 'odd/even', 'odd/odd'],
                    parity_groups=[list(range(0, n, 2)), list(range(1, n, 2))],
                    precision='complex128', reference_check=args.reference_check)
    rows = []
    if RANK == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        meta_path = args.output / 'experiment.json'
        csv_path = args.output / 'samples.csv'
        if meta_path.exists():
            if not args.resume or json.loads(meta_path.read_text()) != identity:
                raise RuntimeError('Output already exists or resume configuration differs')
            if csv_path.exists():
                with csv_path.open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                for i, row in enumerate(rows):
                    assert int(row['sample']) == i
                    assert all(np.isfinite(float(row[k])) for k in FIELDS)
                if len(rows) > args.samples:
                    raise RuntimeError('Checkpoint contains too many samples')
        else:
            if csv_path.exists():
                raise RuntimeError('samples.csv exists without metadata')
            atomic_json(meta_path, identity)
    start_index = COMM.bcast(len(rows) if RANK == 0 else None, root=0)
    # Device construction allocates the statevector: do this only on compute nodes.
    COMM.Barrier()
    started = time.perf_counter()
    dev = qml.device('default.qubit' if args.reference_check else 'lightning.kokkos',
                     wires=n, shots=None, **({} if args.reference_check else
                                             {'mpi': True, 'c_dtype': np.complex128}))
    initialization_seconds = COMM.allreduce(time.perf_counter() - started, op=MPI.MAX)
    local_amplitudes = None
    if not args.reference_check:
        state = dev._statevector
        assert state._mpi
        local_amplitudes = int(state._qubit_state.getLocalBlockSize())
        assert local_amplitudes * SIZE == 2**n

    @qml.qnode(dev, interface=None, diff_method=None)
    def moments(theta):
        for wire in range(n):
            qml.RY(theta[wire], wires=wire)
        # Pauli words use the backend's specialized expectation implementation.
        # Do not build a dense matrix or request qml.state()/qml.probs(all wires).
        even = qml.prod(*(qml.PauliZ(w) for w in range(0, n, 2)))
        odd = qml.prod(*(qml.PauliZ(w) for w in range(1, n, 2)))
        both = qml.prod(*(qml.PauliZ(w) for w in range(n)))
        return qml.expval(even), qml.expval(odd), qml.expval(both)

    hosts = COMM.gather(platform.node(), root=0)
    metadata = dict(identity, mpi_ranks=SIZE, hosts=hosts,
                    local_amplitudes=local_amplitudes,
                    statevector_gib=16 * 2**n / 2**30,
                    statevector_gib_per_rank=None if local_amplitudes is None else
                                             16 * local_amplitudes / 2**30,
                    omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    pennylane=qml.__version__, numpy=np.__version__,
                    lightning=importlib.metadata.version('pennylane-lightning'),
                    mpi_library=MPI.Get_library_version().strip(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    initialization_seconds=initialization_seconds,
                    resumed_from_sample=start_index)
    if RANK == 0:
        atomic_json(args.output / f'runtime-{os.environ.get("SLURM_JOB_ID", "local")}.json', metadata)
        handle = (args.output / 'samples.csv').open('a', newline='', buffering=1)
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        print(json.dumps(dict(event='initialized', **metadata)), flush=True)

    def save_summary(complete):
        if RANK != 0:
            return
        p = np.array([[float(r[k]) for k in FIELDS[1:5]] for r in rows])
        exact = np.array([[float(r[k]) for k in FIELDS[5:9]] for r in rows])
        count = len(rows)
        m_even, m_odd = (n + 1)//2, n//2
        theoretical = (2.0**(-m_even) + 2.0**(-m_odd) + 2.0**(-n)) / 16
        summary = dict(qubits=n, requested_samples=args.samples, completed_samples=count,
                       status='complete' if complete else 'in_progress',
                       mean_probabilities=p.mean(axis=0).tolist(),
                       unbiased_policy_variance=p.var(axis=0, ddof=1).tolist() if count > 1 else None,
                       analytic_sample_variance=exact.var(axis=0, ddof=1).tolist() if count > 1 else None,
                       theoretical_mean=0.25, theoretical_population_variance=theoretical,
                       max_probability_error=max(float(r['max_abs_error']) for r in rows),
                       total_sample_seconds=sum(float(r['seconds']) for r in rows),
                       max_rank_rss_gib=max(float(r['max_rank_rss_kib']) for r in rows) / 2**20,
                       note='Variance over initializations of exact action probabilities; '
                            'not shot variance or gradient variance. Product-state validation only.')
        atomic_json(args.output / 'summary.json', summary)

    for sample in range(start_index, args.samples):
        theta = None
        if RANK == 0:
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, n, sample]))
            theta = rng.uniform(-np.pi, np.pi, n)
        theta = COMM.bcast(theta, root=0)
        COMM.Barrier()
        tic = time.perf_counter()
        measured = np.asarray(moments(theta), dtype=float)
        elapsed = COMM.allreduce(time.perf_counter() - tic, op=MPI.MAX)
        lower, upper = np.empty(3), np.empty(3)
        COMM.Allreduce(measured, lower, op=MPI.MIN)
        COMM.Allreduce(measured, upper, op=MPI.MAX)
        if np.max(upper - lower) > 1e-10:
            raise RuntimeError('Measurement results disagree across ranks')
        p, expected = reconstruct(measured), exact_policy(theta)
        error = float(np.max(np.abs(p - expected)))
        if not np.isfinite(p).all() or error > 1e-9 or np.min(p) < -1e-10 or abs(p.sum()-1) > 1e-10:
            raise RuntimeError(f'Probability validation failed at sample {sample}: error={error}')
        rss = COMM.allreduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, op=MPI.MAX)
        # Linux reports KiB; macOS reports bytes (reference-check only).
        if platform.system() == 'Darwin':
            rss /= 1024
        if RANK == 0:
            row = dict(zip(FIELDS, [sample, *p.tolist(), *expected.tolist(), error, elapsed, rss]))
            writer.writerow(row)
            rows.append(row)
            if (sample + 1) % args.checkpoint_every == 0 or sample + 1 == args.samples:
                handle.flush()
                os.fsync(handle.fileno())
                save_summary(sample + 1 == args.samples)
                print(json.dumps(dict(event='progress', completed=sample + 1,
                                      requested=args.samples, last_seconds=elapsed,
                                      max_error=error, max_rank_rss_gib=rss/2**20)), flush=True)
    if RANK == 0:
        save_summary(True)
        handle.close()
        print('COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        COMM.Abort(1)

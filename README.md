# Quantum simulation experiments on Deucalion ARM

Standalone PennyLane experiments supporting the EPICURE case study on dense
statevector simulation and gradient-based training of parameterized quantum
reinforcement learning agents on Deucalion ARM.

**All reported experiments ran on Deucalion's ARM partition (`normal-arm`),
using Fujitsu A64FX CPU nodes.** They did not run on Deucalion's x86 or GPU
partitions, or on quantum hardware.

**These programs do not import or use QASP.** Circuits are constructed directly
with PennyLane gates, observables and QNodes. The surrounding programs use
Autograd or PyTorch, Gymnasium, NumPy and, for distributed execution, MPI.

## Experiments

| Directory | Measurement | Recorded campaign |
|---|---|---|
| [Born timing](experiments/born-timing/) | Six-qubit training; 30 paired seeds at 1, 8, 16, 32 and 48 active threads | 2152965, 27–28 October 2026 |
| [MPI variance](experiments/mpi-variance/) | Dense 30/31/32/34-qubit simulation on 2/4/8/32 nodes; 2,000 initializations each | 2152291–2152294, 27 October 2026 |
| [Hybrid training](experiments/hybrid-training/) | Six-qubit, 86-parameter softmax agent; three seeds, 80,000 transitions each, episode cap 500 | 2160932, 29 October 2026 |
| [Gradient memory and time](experiments/gradient-memory/) | Backpropagation, adjoint and sequential parameter-shift, plus forward control; 6–20 qubits | 2177536, 2 November 2026 |

The Python experiment programs and all seven Slurm scripts are **exact copies
of the executed/submitted files**. `controlled_training.slurm` was originally
submitted as `run.slurm`; only its filename has changed. Source hashes and copy
origins are recorded in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json).

## Recorded findings and limits

- For the small Born-policy workload, one active thread reduced median paired
  training time by 14.16% relative to 48 threads. Both settings reserved the
  same exclusive node. This does not establish optimal settings for larger
  circuits or for the separate hybrid policy.
- The MPI experiment evolved the full dense statevector and checked its outputs
  against an independent product-state reference. The product-state circuit
  itself can be evaluated efficiently without a dense statevector: this is a
  simulator capacity/validation test, not evidence that this model needs HPC.
- Hybrid seeds 7/17/27 reached final cap-500 mean returns of
  492.22/165.14/486.75. The all-three learning criterion failed. These development
  seeds are not independent confirmation. No full controlled hybrid thread
  comparison was executed.
- At 20 qubits, median output-plus-gradient time and added peak process memory
  were 5.576 s / 1020.25 MiB for backpropagation, 5.325 s / 103.5625 MiB for
  adjoint, and 140.466 s / 61.25 MiB for parameter-shift. An initialized baseline
  of about 670 MiB is excluded from these increments. RSS includes backend
  buffers, metadata and allocator effects; it is not a count of live states.
- The gradient benchmark's 12 nominal 16-qubit/50-gate controls use only 11
  active wires. They are excluded from fixed-16-qubit depth claims. All main
  qubit-sweep cases use all declared wires; the 100/200-gate depth comparison is
  valid. The archived protocol describes the original declaration; this note
  records the subsequent analysis correction without rewriting that protocol.
- A preliminary global-parity adjoint test attempted a 64 GiB dense observable
  allocation at 16 qubits. The final benchmark uses local `Z(0)` for every method.
  This was an implementation/observable issue, not an inherent adjoint memory law.

[recorded-results/](recorded-results/) contains selected existing summaries and
measurements, unchanged from the study. It is not the full raw evidence archive:
checkpoints, per-process traces, failed-smoke logs and complete offline validators
are not included here. Historical submission/protocol records retain their
original pre-run wording; the findings above describe the completed runs.

## Software and hardware

Recorded hardware: Deucalion ARM, Fujitsu A64FX, 48 compute cores per node.
The first three campaigns use `lightning.kokkos`; the MPI campaign requires an
MPI-enabled build. The gradient benchmark uses `default.qubit` with the Torch
interface, CPU execution and complex128 states.

The recorded environment included Python 3.12.3, PennyLane 0.45.0.dev82,
PennyLane-Lightning-Kokkos 0.45.0.dev33, NumPy 2.4.4, Gymnasium 1.2.3 and
PyTorch 2.11.0. The gradient worker recorded the PyTorch build as
`2.11.0+cu130`, but ran on the CPU. Kokkos 5.1.0 was recorded for the timing study.
These are historical development builds, **not an assertion that equivalent
wheels are available from PyPI**. The exact backend build, MPI integration and
thread runtime affect reproducibility. A generic `pip install` is not a complete
reconstruction of the recorded Deucalion environment.

The Slurm scripts retain the actual account, partition, modules and shared
environment path used on Deucalion. They require access to that account and
environment. They are historical execution records, not portable templates.
For another environment, make separately named adapted copies and retain the
originals for comparison. Build guidance is available in the
[Lightning-Kokkos documentation](https://docs.pennylane.ai/projects/lightning/en/stable/lightning_kokkos/device.html).

`threadpoolctl` 3.6.0 is bundled beside the gradient worker, with its original
[license](experiments/gradient-memory/threadpoolctl-LICENSE) and
[dependency provenance](experiments/gradient-memory/dependency.json).

## Recorded submission commands

These commands allocate compute resources. Run each campaign from its own
directory so relative paths resolve correctly. Create `logs/` before submission.
The commands document the completed campaigns; cloning this repository does
not submit jobs or run experiments.

### Born timing

```bash
cd experiments/born-timing
mkdir -p logs
sbatch --array=0-29%2 controlled_training.slurm
```

Each array task executes the five thread settings sequentially, in balanced
order, on the same exclusive node. The launcher creates fresh processes and
refuses to reuse an existing seed output directory. The array specification was
supplied at submission rather than embedded in the original script.

### MPI variance

```bash
cd experiments/mpi-variance
mkdir -p logs
sbatch q30.slurm
sbatch q31.slurm
sbatch q32.slurm
sbatch q34.slurm
```

The scripts request one MPI rank per node and 48 active OpenMP threads per rank.
Qubit count and node count change together: these runs are not a strong-scaling
study. The programs use resumable output directories; do not mix output from
different configurations.

### Hybrid training

```bash
cd experiments/hybrid-training
mkdir -p logs
sbatch --array=0-2%2 learning_validation.slurm
```

The exact script supplies `--limit 500`; the Python default alone is 200.
Seeds are 7, 17 and 27. The script permits at most two concurrent array tasks.
Use a fresh working copy for reruns: this trainer is not a general checkpoint
resume tool and existing output files must not be reused.

### Gradient memory and time

```bash
cd experiments/gradient-memory
mkdir -p logs
sbatch --export=ALL,PHASE=smoke --time=00:15:00 memory.slurm
```

After the eight-case smoke has completed and its validation passes, inspect its
memory/time costs against the declared resource bounds before production:

```bash
sbatch --export=ALL,PHASE=production memory.slurm
```

Production reads `smoke-results/validation.json`; it is not an independent
first command. Cases execute sequentially in fresh processes, with a 24 GiB RSS
guard and a 15-minute per-case limit. Each worker pins itself to one core; the
sampler remains separately schedulable. Linux `/proc`, CPU affinity APIs and
high-water-mark reset support are required; this measurement method is not
portable unchanged to macOS or Windows. The declared depth controls, including
the flagged 11-active-wire case, remain in the exact runner.

## Thread configuration

Allocated CPUs and active simulator workers are different settings. The timing
and hybrid scripts use `OMP_PLACES=cores` and `OMP_PROC_BIND=spread`; the gradient
benchmark uses `cores/close` with one explicitly pinned worker. No controlled
`close` versus `spread` experiment was performed. Classical thread pools are
restricted separately. Check effective worker counts and affinity, rather than
inferring them from Slurm allocations alone.

## Check this source snapshot

```bash
python3 verify_sources.py
```

This offline check verifies the copied-file hashes and parses the Python and
JSON files. It does not import the simulator, execute a circuit, submit a job,
or repeat the scientific validation. Full reruns require the documented runtime
and compute allocation.

## License

The original experiment code and documentation are available under the
[MIT license](LICENSE). The bundled `threadpoolctl` module retains its separate
BSD 3-Clause license and copyright notices in
[threadpoolctl-LICENSE](experiments/gradient-memory/threadpoolctl-LICENSE).

#!/bin/bash -l
#SBATCH --job-name=parity4-q31
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --account=eehpc-dev-2026d02-152a
#SBATCH --partition=normal-arm
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Set --nodes and --ntasks together at submission. One rank per ARM node.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load CMake/3.29.3-GCCcore-13.3.0 OpenMPI/5.0.3-GCC-13.3.0 \
            Ninja/1.12.1-GCCcore-13.3.0 Python/3.12.3-GCCcore-13.3.0
source /projects/EEHPC-DEV-2026D02-152/pl-kokkos-mpi/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export N_QUBITS=31
export RUN_LABEL="${RUN_LABEL:-q31}"
[[ "$(uname -m)" == aarch64 ]]
[[ "$SLURM_NTASKS" == "$SLURM_JOB_NUM_NODES" ]]
echo "job=$SLURM_JOB_ID qubits=$N_QUBITS ranks=$SLURM_NTASKS nodes=$SLURM_JOB_NUM_NODES threads=$OMP_NUM_THREADS"
module list
command -v python
command -v mpirun
mpirun -np "$SLURM_NTASKS" \
    --map-by ppr:1:node --bind-to none --report-bindings \
    python -u variance_product_mpi.py --qubits "$N_QUBITS" \
    --samples "${N_SAMPLES:-2000}" --seed 20261027 \
    --output "results/${RUN_LABEL}" --resume --checkpoint-every 10

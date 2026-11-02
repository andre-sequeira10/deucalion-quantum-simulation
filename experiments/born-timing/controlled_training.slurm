#!/bin/bash -l
#SBATCH --job-name=parity-thread-control
#SBATCH --account=eehpc-dev-2026d02-152a
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load CMake/3.29.3-GCCcore-13.3.0 OpenMPI/5.0.3-GCC-13.3.0 \
            Ninja/1.12.1-GCCcore-13.3.0 Python/3.12.3-GCCcore-13.3.0
source /projects/EEHPC-DEV-2026D02-152/pl-kokkos-mpi/bin/activate
[[ "$(uname -m)" == aarch64 ]]
module list
echo "job=$SLURM_JOB_ID node=$(hostname)"
# One fresh process per thread count. Exclusive identical node allocation avoids
# co-tenant effects; only the active Kokkos/OpenMP thread count changes.
if [[ "${PILOT:-0}" == 1 ]]; then
    python -u run_pair.py --index 0 --pilot
else
    python -u run_pair.py --index "$SLURM_ARRAY_TASK_ID"
fi

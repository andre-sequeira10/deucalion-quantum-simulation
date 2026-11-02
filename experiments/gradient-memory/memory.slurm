#!/bin/bash -l
#SBATCH --job-name=gradient-memory
#SBATCH --account=eehpc-dev-2026d02-152a
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load CMake/3.29.3-GCCcore-13.3.0 OpenMPI/5.0.3-GCC-13.3.0 \
            Ninja/1.12.1-GCCcore-13.3.0 Python/3.12.3-GCCcore-13.3.0
source /projects/EEHPC-DEV-2026D02-152/pl-kokkos-mpi/bin/activate
[[ "$(uname -m)" == aarch64 ]]
module list
export OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
echo "job=$SLURM_JOB_ID node=$(hostname) phase=${PHASE:?}"
sha256sum measure.py run.py memory.slurm protocol.json
python -u run.py "$PHASE"

#!/bin/bash -l
#SBATCH --job-name=horizon-v3-learning
#SBATCH --account=eehpc-dev-2026d02-152a
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=08:00:00
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
export OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=spread
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
SEEDS=(7 17 27)
export OMP_NUM_THREADS=1
python -u train_annealed.py --seed "${SEEDS[$SLURM_ARRAY_TASK_ID]}" --threads 1 --transitions 80000 --limit 500 --eval-episodes 100 --eval-every 10000 --decay-start 30000 --decay-end 80000 --final-rate-factor .1 --output "learning-validation/seed${SEEDS[$SLURM_ARRAY_TASK_ID]}"

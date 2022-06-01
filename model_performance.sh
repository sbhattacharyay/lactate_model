#!/bin/bash
#SBATCH --job-name=dynAPM_performance
#SBATCH --time=00:20:00
#SBATCH --array=0-999
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/performance/v1-0_dynAPM_performance_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python model_performance.py $SLURM_ARRAY_TASK_ID
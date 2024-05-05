#!/bin/sh

#SBATCH --array               19-135
#SBATCH --cpus-per-task       8
#SBATCH --error               /scratch/nia4240/attention-scratch/slurm_logs/%A/%A_%a.err
#SBATCH --gpus-per-node       1
#SBATCH --mem                 32G
#SBATCH --open-mode           append
#SBATCH --output              /scratch/nia4240/attention-scratch/slurm_logs/%A/%A_%a.out
#SBATCH --time                06:00:00


singularity exec --nv --overlay /scratch/nia4240/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate attention
python -m fire slurm run_job paper_experiments/fig1_config.yaml --grid_path=paper_experiments/fig1_sweep1.yaml --sweep_id=$SLURM_ARRAY_JOB_ID --job_id=$SLURM_ARRAY_TASK_ID 
"


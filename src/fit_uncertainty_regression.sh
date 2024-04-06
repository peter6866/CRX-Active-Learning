#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --array=1-20%5
#SBATCH -o log/uncertainty_regression_%a.out
#SBATCH -e log/uncertainty_regression_%a.err
#SBATCH --mail-type=END,FAIL

python3 src/fit_uncertainty_regression.py $SLURM_ARRAY_TASK_ID


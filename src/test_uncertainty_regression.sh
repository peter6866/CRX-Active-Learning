#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH -o log/uncertainty_test_%j.out
#SBATCH -e log/uncertainty_test_%j.err
#SBATCH --mail-type=FAIL


python3 src/test_uncertainty_regression.py

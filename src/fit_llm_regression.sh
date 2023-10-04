#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -p gpu
#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH -o log/transformer_regression.out
#SBATCH -e log/transformer_regression.err
#SBATCH --mail-type=END,FAIL


python3 src/fit_llm_regression.py

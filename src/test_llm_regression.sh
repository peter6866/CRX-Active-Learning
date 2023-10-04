#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH -o log/transformer_regression_test_%j.out
#SBATCH -e log/transformer_regression_test_%j.err
#SBATCH --mail-type=END,FAIL


python3 src/test_llm_regression.py

#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -o log/wgan_eval_%j.out
#SBATCH -e log/wgan_eval_%j.err
#SBATCH --mail-type=END,FAIL


python3 src/gan_eval.py
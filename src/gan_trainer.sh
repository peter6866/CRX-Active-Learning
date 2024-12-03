#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -o log/wgan_%j.out
#SBATCH -e log/wgan_%j.err
#SBATCH --mail-type=END,FAIL


python3 src/gan_trainer.py
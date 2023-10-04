#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --mem=8G
#SBATCH --time=4-00:10:00
#SBATCH -o log/importance_analysis_nrl.out
#SBATCH -e log/importance_analysis_nrl.err
#SBATCH --mail-type=BEGIN,END,FAIL

eval $(spack load --sh miniconda3)
source activate active-learning

which python3

model_dir=ModelFitting/CNN_Reg/best_model/
dirname=ImportanceAnalysis/CrxNrl

mkdir -p $dirname
python3 src/importance_analysis_crx_nrl.py "${model_dir}" "${dirname}"

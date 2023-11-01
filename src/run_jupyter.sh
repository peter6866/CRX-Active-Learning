#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -o log/jupyterFromCluster-log-%A.txt
#SBATCH -e log/jupyterFromCluster-log-%A.txt
eval $(spack load --sh miniconda3)
unset PYTHONPATH
unset XDG_RUNTIME_DIR
source activate active-learning
port=$(shuf -i9000-9999 -n1)
host=$(hostname)
# Print tunneling instructions to ~/logs/jupyterFromCluster-log-{jobid}.txt
echo -e "
    Run in your local terminal to create an SSH tunnel to $host
    -----------------------------------------------------------
    ssh -N -L $port:$host:$port $USER@login.htcf.wustl.edu
    -----------------------------------------------------------
    Go to the following address in a browser on your local machine
    --------------------------------------------------------------
    https://localhost:$port
    --------------------------------------------------------------
    "
# Launch Jupyter lab server
jupyter lab --no-browser --port=${port} --ip=${host}
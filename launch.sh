#!/bin/bash
#  salloc -q interactive -C gpu -t 4:00:00 -A nintern -N 1 --gpu-bind=none --image=nersc/pytorch:24.06.01 --module=gpu,nccl-plugin

#SBATCH --image=nersc/pytorch:24.06.01
#SBATCH --module=gpu,nccl-plugin
#SBATCH --output=out/%j.out
#SBATCH --time=28:00 -q debug

set -u  # exit if you try to use an uninitialized variable
set -e  # bash exits if any statement returns a non-true return value

# get inside
# shifter --image=nersc/pytorch:24.06.01 --module gpu,nccl-plugin --env PYTHONUSERBASE=$SCRATCH/qml_env
srun -l -N 1 shifter --env PYTHONUSERBASE=$SCRATCH/qml_env python main.py
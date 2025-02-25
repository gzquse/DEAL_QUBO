#!/bin/bash
# filename          : mar_test.sh
# description           : testing scripts
# author            : Ziqing Guo
# email             : ziqguo@ttu.edu
# date              : July 5, 2024
# version           : 1.3
# usage             : ./mar_test.sh
# notes             : This script is built to run on Quanah.hpcc.ttu.edu.
# license           : MIT License
#==============================================================================

#SBATCH --job-name=QUBO
#SBATCH --output=out/%x.o%j
#SBATCH --error=out/%x.e%j
#SBATCH --partition=nocona
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=5370MB  #5.3GB per core
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ziqguo@ttu.edu

#Override field separators to line terminators and store the original field separators.
OIFS="$IFS"
IFS=$'\n'

#Load the Bioinformatics conda environment.
. $HOME/conda/etc/profile.d/conda.sh
conda activate qaoa

#Delete the executable then re-compile the source code.
echo -e "\n\nBenchmarking crosstalking 1.\n"
python ./benchmark/benchmark.py -m hybrid -i 10
#Return the field separator to its original value and deactivate conda.
IFS="$OIFS"
conda deactivate

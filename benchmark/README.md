# PM setup
module load conda
conda activate /pscratch/sd/g/gzquse/qaoa/

conda activate 
# local conda env
pip install --user -r .\requirements.txt

dwave auth login
dwave auth get
dwave setup --auth

# HPCC

. $HOME/conda/etc/profile.d/conda.sh

conda activate qaoa

# run
./benchmark.py  -m hybrid -s qiskit.statevector_simulator
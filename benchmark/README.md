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

# pipeline 

## get optimized circ qubo 
1. ./benchmark.py  -m hybrid -s qiskit.statevector_simulator --prjName kp

## switch conda env
2. conda activate qiskit
## dry run qiskit locally
3. ./np_backends.py  --prjName maxcut

## dry run qpy circuit compiled from benchmark
4. ./np_backends.py  --prjName qpy --infPath circ/hybrid_95a850.qpy --dryRun

## send to real quantum computer
5. export QISKIT_IBM_TOKEN="MY_IBM_CLOUD_API_KEY"
6. ./np_backends.py  --prjName qpy --infPath circ/hybrid_95a850.qpy
wait until it finished and retrieve the results
7. ./np_backends.py  --prjName plot --jobID cz4ynz710wx0008bhvvg --backend ibm_kyiv


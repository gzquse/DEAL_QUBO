conda activate qiskit

export QISKIT_IBM_TOKEN=""

./np_backends.py -v 1 --prjName qpy

./np_backends.py -v 1 --prjName qpy --infPath circ/hybrid_78dff2.qpy --dryRun True

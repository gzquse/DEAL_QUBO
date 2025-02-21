#!/usr/bin/env python3
import argparse
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'toolbox')))
from PlotterBackbone import PlotterBackbone

# useful additional packages
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit_algorithms import SamplingVQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler as RuntimeSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import CouplingMap, Layout
from qiskit.providers.fake_provider import GenericBackendV2

# backends
# FakeAlgiers
#     FakeAlmadenV2
#     FakeArmonkV2
#     FakeAthensV2
#     FakeAuckland
#     FakeBelemV2
#     FakeBoeblingenV2
#     FakeBogotaV2
#     FakeBrisbane
#     FakeBrooklynV2
#     FakeBurlingtonV2
#     FakeCairoV2
#     FakeCambridgeV2
#     FakeCasablancaV2
#     FakeCusco
#     FakeEssexV2
#     FakeFez
#     FakeGeneva
#     FakeGuadalupeV2
#     FakeHanoiV2
#     FakeJakartaV2
#     FakeJohannesburgV2
#     FakeKawasaki
#     FakeKolkataV2
#     FakeKyiv
#     FakeKyoto
#     FakeLagosV2
#     FakeLimaV2
#     FakeFractionalBackend
#     FakeLondonV2
#     FakeManhattanV2
#     FakeManilaV2
#     FakeMarrakesh
#     FakeMelbourneV2
#     FakeMontrealV2
#     FakeMumbaiV2
#     FakeNairobiV2
#     FakeOsaka
#     FakeOslo
#     FakeOurenseV2
#     FakeParisV2
#     FakePeekskill
#     FakePerth
#     FakePrague
#     FakePoughkeepsieV2
#     FakeQuebec
#     FakeQuitoV2
#     FakeRochesterV2
#     FakeRomeV2
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeFez, FakeMarrakesh

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")

    parser.add_argument("-p", "--showPlots",  default='b', nargs='+',help="abc-string listing shown plots")
    parser.add_argument("-m", "--penMed", choices=["X", "X2", "hybrid"], default="hybrid",help='hamiltonian method')
    parser.add_argument("--outPath",default='out/',help="all outputs from experiment")
    parser.add_argument("--proName",  default='maxcut', help='problem name')
    parser.add_argument("-s", "--simName", choices=["qiskit.statevector_simulator", "qiskit.shot_simulator", "classic"], default='qiskit.statevector_simulator', help='simulators')
    parser.add_argument("-j", "--jobID",  default=None,help='(optional) jobID assigned during submission')
    args = parser.parse_args()
    # make arguments  more flexible
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    args.showPlots=''.join(args.showPlots)

    return args

def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)


def compute_max_cut(args):
    # Generating a graph of 4 nodes
    n = 4  # Number of nodes in graph
    G = nx.Graph()
    G.add_nodes_from(np.arange(0, n, 1))
    elist = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    # tuple is (i,j,weight) where (i,j) is the edge
    G.add_weighted_edges_from(elist)

    colors = ["r" for node in G.nodes()]
    pos = nx.spring_layout(G)

    # Computing the weight matrix from the random graph
    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = temp["weight"]
    print(w)

    max_cut = Maxcut(w)
    qp = max_cut.to_quadratic_program()
    print(qp.prettyprint())

    qubitOp, offset = qp.to_ising()
    print("Offset:", offset)
    print("Ising Hamiltonian:")
    print(str(qubitOp))

    # solving Quadratic Program using exact classical eigensolver
    exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    result = exact.solve(qp)
    print(result.prettyprint())

    # Making the Hamiltonian in its full form and getting the lowest eigenvalue and eigenvector
    ee = NumPyMinimumEigensolver()
    result = ee.compute_minimum_eigenvalue(qubitOp)

    x = max_cut.sample_most_likely(result.eigenstate)
    print("energy:", result.eigenvalue.real)
    print("max-cut objective:", result.eigenvalue.real + offset)
    print("solution:", x)
    print("solution objective:", qp.objective.evaluate(x))

    colors = ["r" if x[i] == 0 else "c" for i in range(n)]
    # draw_graph(G, colors, pos)

    algorithm_globals.random_seed = 123
    seed = 10598

    # construct SamplingVQE
    optimizer = SPSA(maxiter=300)
    ry = TwoLocal(qubitOp.num_qubits, "ry", "cz", reps=5, entanglement="linear")
    vqe = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=optimizer)

    # run SamplingVQE
    result = vqe.compute_minimum_eigenvalue(qubitOp)

    # print results
    x = max_cut.sample_most_likely(result.eigenstate)
    print("energy:", result.eigenvalue.real)
    print("time:", result.optimizer_time)
    print("max-cut objective:", result.eigenvalue.real + offset)
    print("solution:", x)
    print("solution objective:", qp.objective.evaluate(x))

    # plot results
    colors = ["r" if x[i] == 0 else "c" for i in range(n)]

    # TODO
    # save the last graph 
    #draw_graph(G, colors, pos)

    return vqe

def transpile_backend(vqe):
    service = QiskitRuntimeService(channel="ibm_quantum") # or your preferred channel
    circuit = vqe.ansatz
    # Initialize IBM Provider
    #IBMProvider.save_account(token='fill your token')
    # Get the backend (replace 'ibm_torino' with the correct backend name if needed)
    coupling_map = backend.configuration().coupling_map

    # Generate the optimization level 3 pass manager for local test
    # pm = generate_preset_pass_manager(3, backend)
    # Transpile the circuit using the pass manager
    transpiled_circuit = transpile(circuit, backend=backend)
    # Count CZ gates in the transpiled circuit
    cz_count = transpiled_circuit.count_ops().get('cz', 0)
    # Update VQE ansatz and run
    vqe._ansatz = transpiled_circuit
    # result = vqe.compute_minimum_eigenvalue(qubitOp)
    print("CZ gate count (after transpilation):", cz_count)
    # Draw the transpiled circuit (optional)
    transpiled_circuit.draw('mpl', idle_wires=False)

    # Save the transpiled circuit to a file

if __name__ == "__main__":
    args=get_parser()
    # change to name based on reading the config file
    #reading  => Faketorino
    # current we hard coded
    backend = FakeTorino()

    if args.proName == 'maxcut':
        vqe = compute_max_cut(args)
        # save output in the sub directory
        transpile_backend(vqe)
        
        
    
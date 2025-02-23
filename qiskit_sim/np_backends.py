#!/usr/bin/env python3
import yaml
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'toolbox')))
from PlotterBackbone import PlotterBackbone
import argparse

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

from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeFez, FakeMarrakesh
from datetime import datetime

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
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    args.showPlots=''.join(args.showPlots)
    return args

# Read backend configuration from the YAML file
def read_config():
    with open("backend_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config.get("backend", "FakeTorino")  # Default to FakeTorino

# Read the backend name from config file
backend_name = read_config()

# Dynamically load the backend class using the name from the config
backend = getattr(sys.modules[__name__], backend_name)()

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
    exact_result = exact.solve(qp)
    print(exact_result.prettyprint())

    # Making the Hamiltonian in its full form and getting the lowest eigenvalue and eigenvector
    ee = NumPyMinimumEigensolver()
    ee_result = ee.compute_minimum_eigenvalue(qubitOp)

    x_exact = max_cut.sample_most_likely(ee_result.eigenstate)
    print("energy:", ee_result.eigenvalue.real)
    print("max-cut objective:", ee_result.eigenvalue.real + offset)
    print("solution:", x_exact)
    print("solution objective:", qp.objective.evaluate(x_exact))

    colors = ["r" if x_exact[i] == 0 else "c" for i in range(n)]

    algorithm_globals.random_seed = 123
    seed = 10598

    # construct SamplingVQE
    optimizer = SPSA(maxiter=300)
    ry = TwoLocal(qubitOp.num_qubits, "ry", "cz", reps=5, entanglement="linear")
    vqe = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=optimizer)

    # run SamplingVQE
    vqe_result = vqe.compute_minimum_eigenvalue(qubitOp)

    # print results can be commented out if not needed
    x_vqe = max_cut.sample_most_likely(vqe_result.eigenstate)
    print("energy:", vqe_result.eigenvalue.real)
    print("time:", vqe_result.optimizer_time)
    print("max-cut objective:", vqe_result.eigenvalue.real + offset)
    print("solution:", x_vqe)
    print("solution objective:", qp.objective.evaluate(x_vqe))

    # plot results
    colors = ["r" if x_vqe[i] == 0 else "c" for i in range(n)]

    # Save the last graph
    draw_graph(G, colors, pos)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(args.outPath, f'maxcut_graph_{timestamp}.png'))
    plt.close()

    # Collect parameters for YAML
    params = {
        'weight_matrix': w.tolist(),
        'offset': float(offset),
        'ising_hamiltonian': str(qubitOp),
        'exact_solution': exact_result.prettyprint(),  # Only use prettyprint() for exact result
        'exact_energy': float(ee_result.eigenvalue.real),
        'exact_maxcut_objective': float(ee_result.eigenvalue.real + offset),
        'exact_solution_vector': x_exact.tolist(),
        'exact_solution_objective': float(qp.objective.evaluate(x_exact)),
        'vqe_energy': float(vqe_result.eigenvalue.real),
        'vqe_time': float(vqe_result.optimizer_time),
        'vqe_maxcut_objective': float(vqe_result.eigenvalue.real + offset),
        'vqe_solution': x_vqe.tolist(),
        'vqe_solution_objective': float(qp.objective.evaluate(x_vqe)),
        'timestamp': timestamp
    }

    return vqe, params

def transpile_backend(vqe, args, params):
    QiskitRuntimeService.save_account(token="ADD IBM QUANTUM API TOKEN HERE", overwrite=True, channel="ibm_quantum")

    service = QiskitRuntimeService(channel="ibm_quantum")
    circuit = vqe.ansatz
    coupling_map = backend.configuration().coupling_map

    transpiled_circuit = transpile(circuit, backend=backend)
    cz_count = transpiled_circuit.count_ops().get('cz', 0)
    vqe._ansatz = transpiled_circuit
    print("CZ gate count (after transpilation):", cz_count)
    transpiled_circuit.draw('mpl', idle_wires=False)
    
    timestamp = params['timestamp']
    transpiled_circuit.draw('mpl', idle_wires=False, filename=os.path.join(args.outPath, f'transpiled_circuit_{timestamp}.png'))

    params['cz_gate_count'] = cz_count

    with open(os.path.join(args.outPath, f'parameters_{timestamp}.yaml'), 'w') as f:
        yaml.safe_dump(params, f)

if __name__ == "__main__":
    args = get_parser()
    backend = FakeTorino()

    if args.proName == 'maxcut':
        vqe, params = compute_max_cut(args)
        transpile_backend(vqe, args, params)
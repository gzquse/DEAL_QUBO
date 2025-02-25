#!/usr/bin/env python3
import yaml
import os, sys, hashlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'toolbox')))
from PlotterBackbone import PlotterBackbone

# useful additional packages
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit import QuantumCircuit, transpile
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_ibm_runtime import QiskitRuntimeService
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
    parser.add_argument("-s", "--simName", choices=["qiskit.statevector_simulator", "qiskit.shot_simulator", "classic"], default='qiskit.statevector_simulator', help='simulators')
    parser.add_argument("-j", "--jobID",  default=None,help='(optional) jobID assigned during submission')
    parser.add_argument("--outPath",default='out/',help="all outputs from experiment")
    parser.add_argument("--prjName",  default='maxcut', help='problem name')
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

def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

    def compute_max_cut(self, args, MD):
        # Generating a graph of 4 nodes
        # Parameters
        n = 4  # Number of nodes
        p = 0.2  # Probability of edge creation (sparse graph)

        # Generate a sparse Erdős–Rényi graph G(n, p)
        G = nx.erdos_renyi_graph(n, p)

        # Assign random weights to edges
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.5, 2.0)  # Weights between 0.5 and 2.0

        # Compute weight matrix
        w = np.zeros((n, n))
        for i, j, data in G.edges(data=True):
            w[i, j] = data["weight"]
            w[j, i] = data["weight"]  # Since it's an undirected graph

        # Visualizing the Graph
        pos = nx.spring_layout(G)  # Layout for visualization
        edges = [(u, v) for u, v in G.edges()]
        weights = [G[u][v]['weight'] for u, v in edges]
        print(weights)

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
        vqe = SamplingVQE(sampler=AerSampler(), ansatz=ry, optimizer=optimizer)

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
        hashlib = MD['_hashlib']
        # Save the last graph
        draw_graph(G, colors, pos)
        plt.savefig(os.path.join(args.outPath, f'maxcut_graph_{hashlib}.png'))
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
            '_hashlib': MD['_hashlib'],
        }

        return vqe, params

def transpile_backend(vqe, args, params):

    service = QiskitRuntimeService(channel="ibm_quantum")
    circuit = vqe.ansatz
    coupling_map = backend.configuration().coupling_map

    transpiled_circuit = transpile(circuit, backend=backend)
    cz_count = transpiled_circuit.count_ops().get('cz', 0)
    vqe._ansatz = transpiled_circuit
    print("CZ gate count (after transpilation):", cz_count)
    print('SX gate count (after transpilation):', transpiled_circuit.count_ops().get('sx', 0))
    _hashlib = params['_hashlib']
    transpiled_circuit.draw('mpl', idle_wires=False, filename=os.path.join(args.outPath, f'transpiled_circuit_{_hashlib}.png'))

    params['cz_gate_count'] = cz_count

    with open(os.path.join(args.outPath, f'parameters_{_hashlib}.yaml'), 'w') as f:
        yaml.safe_dump(params, f)

if __name__ == "__main__":
    args = get_parser()
    MD = {}
    _hashlib = hashlib.md5(os.urandom(32)).hexdigest()[:6]
    MD['_hashlib'] = _hashlib
    plot = Plotter(args)
    # backend = FakeTorino()
    # Read the backend name from config file
    backend_name = read_config()
    # Dynamically load the backend class using the name from the config
    backend = getattr(sys.modules[__name__], backend_name)()
    if args.prjName == 'maxcut':
        vqe, params = plot.compute_max_cut(args, MD)
        transpile_backend(vqe, args, params)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ziqing Guo
@contact:
@project: QAOA

@version: 0.1

@description:
for optimier:
Gradient-free optimizers: nelder-mead, powell, cg, bfgs, tnc, cobyla, slsqp, trust-constr, dogleg, trust-ncg, trust-exact, and trust-krylov.

Gradient-based optimizers: vgd, newton, rmsprop, natural_grad_descent, spsa, l-bfgs-b, and newton-cg.

PennyLane optimizers: pennylane_adagrad, pennylane_adam, pennylane_vgd, pennylane_momentum, pennylane_nesterov_momentum, pennylane_rmsprop, pennylane_rotosolve, and pennylane_spsa.

Shot adaptive optimizers: cans, and icans
"""

# basic packages
import argparse
import os, sys, hashlib
import itertools
import numpy as np
from typing import List, Union
import datetime
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'toolbox')))
from PlotterBackbone import PlotterBackbone
from Util_problems import KP, TSP, MAXCUT, normalization
from docplex.mp.model import Model
from qiskit import qpy

# QUBO packages
from openqaoa import QAOA
import numpy as np
#import method to specify the device
from openqaoa.backends import create_device
# import the OpenQAOA Parameterisation classes manually: Manual Mode
from openqaoa.qaoa_components import (PauliOp, Hamiltonian, QAOADescriptor,
create_qaoa_variational_params, QAOAVariationalStandardParams, QAOAVariationalExtendedParams)
from openqaoa.optimizers import get_optimizer
# import the other OpenQAOA modules required for this example
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.backends.qaoa_backend import get_qaoa_backend
from openqaoa.utilities import ground_state_hamiltonian
from openqaoa.problems.converters import FromDocplex2IsingModel
import networkx as nx

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")

    parser.add_argument("-p", "--showPlots",  default='b', nargs='+',help="abc-string listing shown plots")
    parser.add_argument("-m", "--penMed", choices=["X", "X2", "hybrid"], default="hybrid",help='hamiltonian method')

    parser.add_argument("--prjName",  choices=["kp", "maxcut", "tsp"], default='kp', help='problem name')
    parser.add_argument("-s", "--simName", choices=["qiskit.statevector_simulator", "qiskit.shot_simulator", "classic"], default='qiskit.statevector_simulator', help='simulators')
    parser.add_argument("-j", "--jobID",  default=os.getenv("SLURM_JOB_ID", ""),help='(optional) jobID assigned during submission')
    parser.add_argument("-i", "--iterate", type=int, default=1, help="Number of iterations for benchmarking")

    # for backend
    parser.add_argument("--nShots", type=int, default=400, help="Number of shots for the simulator")
    parser.add_argument("--opt", default='pennylane_adam', help="optimizer")

    # IO paths
    parser.add_argument("--basePath",default=None,help="head path for set of experiments, or 'env'")
    parser.add_argument("--outPath",default='out/',help="(optional) redirect all outputs ")
    parser.add_argument("--dryRun", action='store_true', default=False, help="dry run without executing")
    args = parser.parse_args()
    # make arguments  more flexible
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    return args

##############################
#  setting for training 
##############################
penalty_strength = 1  # Penalty multiplier for all methods
connectivity = "full"  # Using a 2D lattice
idle_qubits = []  # idle qubits
# Define your terms and coefficients
terms = [PauliOp('Z', [0]), PauliOp('Z', [1]), PauliOp('ZZ', [0, 1])]
coeffs = [1.0, 1.0, 0.5]
constant = 0.0
grid_width = 2  # Required for grid topology

##############################
#  get objective hamiltonian 
##############################
# setting for penalty strength
lambda0 = 1
lambda1 = 0
lambda2 = 0

def get_model(problem: str, **kwargs) -> Model:
    """
    Get the objective Hamiltonian for a given problem.

    Parameters
    ----------
    problem : str
        Name of the problem to solve.

    Returns
    -------
    Hamiltonian
        Objective Hamiltonian for the problem.
    """
    if problem == "kp":
        # Define the QUBO terms and coefficients
        # Knapsack Problem Parameters
        MD = {
            "n": 5,  # Number of variables
            "values": [8, 47, 10, 5, 16],
            "weights": [3, 11, 14, 19, 5],
            "capacity": 26,
        }
        lambda1 = 0.9603 
        lambda2 = 0.0371
        return KP(MD["values"], MD["weights"], MD["capacity"]), MD

    elif problem == "maxcut":
        # Define the Max Cut problem
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)])
        for i, j in G.edges():
            G[i][j]['weight'] = np.random.uniform(0.5, 2.0)  # Random edge weights
        return MAXCUT(G)

    elif problem == "tsp":
        lambda0 = 38.2584  
        lambda1 = 18.2838
        lambda2 = 57.0375
        # Define the TSP problem
        MD = {
            'lx': 50, 
            'ly': 50,   # The grid size where the points will be selected randomly
            'seed': 5,
            'cities': 4
        }
        np.random.seed(MD['seed'])
        positions = {i: np.random.randint(0, [MD['lx'], MD['ly']], 2) for i in range(MD['cities'])}
        G = nx.Graph()
        G.add_nodes_from(range(MD["cities"]))
        for i in range(MD["cities"]):
            for j in range(MD["cities"]):
                if i != j:
                    rij = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
                    G.add_weighted_edges_from([[i, j, rij]])
        return TSP(G), MD
    else:
        raise ValueError(f"Invalid problem: {problem}")
    
def adaptive_xy_hybrid_mixer(
    weights: List[float], 
    capacity: float, 
    penalty_strength: float, 
    n_qubits: int, 
    idle_qubits: List[int] = None, 
    stabilizer_strength: float = 0.1, 
    connectivity: Union[List[tuple], str] = "full",
    grid_width: int = None,
    random_connectivity_prob: float = 0.3
) -> Hamiltonian:
    """
    Constructs an adaptive XY hybrid mixer Hamiltonian with idle qubit compensation and stabilizers.

    Parameters
    ----------
    weights : list
        Weights associated with each qubit (e.g., item weights in a knapsack problem).
    capacity : float
        Total capacity (constraint) of the system.
    penalty_strength : float
        Penalty multiplier for enforcing constraints.
    n_qubits : int
        Total number of qubits in the system.
    idle_qubits : list, optional
        List of qubits that are idle and need compensation.
    stabilizer_strength : float, optional
        Strength of additional stabilizers to reduce noise.
    connectivity : Union[List[tuple], str], optional
        Connectivity of the qubits, default is "full". Can be "full", "chain", "star", "ring", "grid", "tree", 
        "hypercube", "random", or a custom list of edges.
    grid_width : int, optional
        Width of the grid, required if using "grid" topology.
    random_connectivity_prob : float, optional
        Probability of an edge in "random" connectivity.

    Returns
    -------
    Hamiltonian
        The adaptive hybrid XY mixer Hamiltonian.
    """

    # Define connectivity topologies
    connectivity_topology_dict = {
        "full": list(itertools.combinations(range(n_qubits), 2)),
        "chain": [(i, i + 1) for i in range(n_qubits - 1)],
        "star": [(0, i + 1) for i in range(n_qubits - 1)],
        "ring": [(i, (i + 1) % n_qubits) for i in range(n_qubits)],  # Cyclic connection
        "grid": [(i, i + 1) for i in range(n_qubits) if (i + 1) % grid_width != 0] + 
                [(i, i + grid_width) for i in range(n_qubits - grid_width)] if grid_width else None,
        "tree": [(i, 2 * i + 1) for i in range(n_qubits // 2)] + 
                [(i, 2 * i + 2) for i in range(n_qubits // 2)],
        "hypercube": [(i, i ^ (1 << j)) for i in range(n_qubits) for j in range(int(np.log2(n_qubits))) if (i ^ (1 << j)) < n_qubits],
        "random": [(i, j) for i in range(n_qubits) for j in range(i+1, n_qubits) if np.random.rand() < random_connectivity_prob],
    }

    # Determine connectivity
    if isinstance(connectivity, str):
        if connectivity in connectivity_topology_dict:
            if connectivity == "grid" and grid_width is None:
                raise ValueError("Grid topology requires 'grid_width' parameter.")
            connectivity = connectivity_topology_dict[connectivity]
        else:
            raise ValueError(f"Invalid connectivity type. Choose from {list(connectivity_topology_dict.keys())}.")

    # Initialize Hamiltonian terms and coefficients
    pauli_terms = []
    coefficients = []

    # Hybrid penalty terms (original constraints)
    for i in range(n_qubits):
        for j in range(n_qubits):
            if i != j:
                pauli_terms.append(PauliOp('XX', [i, j]))  # Interaction term
                coefficients.append(penalty_strength * weights[i] * weights[j] / (capacity ** 2))

        # Linear constraint term
        pauli_terms.append(PauliOp('X', [i]))
        coefficients.append(penalty_strength * weights[i] * (capacity - weights[i]) / capacity)

    # XY Mixer Terms with Custom Topology
    for pair in connectivity:
        i, j = pair
        pauli_terms.append(PauliOp.X(i) @ PauliOp.X(j))  # XX terms
        pauli_terms.append(PauliOp.Y(i) @ PauliOp.Y(j))  # YY terms
        coefficients.extend([0.5, 0.5])  # Default weight for XY mixing

    # Add stabilizers for idle qubits
    if idle_qubits:
        for q in idle_qubits:
            pauli_terms.append(PauliOp('X', [q]))  # Stabilizer term
            coefficients.append(stabilizer_strength)

    # Construct and return the Hamiltonian
    return Hamiltonian(pauli_terms, coefficients, constant=0.0)

def define_qaoa_problem(args, method):
    """
    Define the QAOA Hamiltonian based on the selected penalty method.

    Returns
    -------
    qaoa : QAOA
        QAOA instance for solving the problem.
    """
    # Objective Hamiltonian (maximize value)
    method = args.penMed
    model, MD = get_model(args.prjName)
    weights = MD.get("weights", [3, 11, 14, 19, 5])
    capacity = 26
    n = MD.get("n", 5)
    # Constraint Hamiltonian
    if method == "X":
        # Add slack variables explicitly as qubits
        terms = [PauliOp('X', [i]) for i in range(n)]
        coeffs = [-penalty_strength * weights[i] for i in range(n)]
        mix_hamiltonian = Hamiltonian(terms, coeffs, constant=0.0)

    elif method == "X2":
        # Unbalanced penalty method
        terms = [PauliOp('X', [i]) for i in range(n)]
        coeffs = [
            -penalty_strength * weights[i] + penalty_strength * (weights[i] ** 2)
            for i in range(n)
        ]
        mix_hamiltonian = Hamiltonian(terms, coeffs, constant=0.0)

    elif method == "hybrid":
        # Hybrid penalty method
        mix_hamiltonian = adaptive_xy_hybrid_mixer(
                                                weights, capacity, penalty_strength, 
                                                n, idle_qubits, stabilizer_strength=0.2, 
                                                connectivity=connectivity, grid_width=grid_width
                                            )

    # Create QAOA instance
    # qaoa = QAOA()
    norm_model = normalization(FromDocplex2IsingModel(model, multipliers=lambda0, unbalanced_const=True,
                                         strength_ineq=[lambda2/lambda0, lambda1/lambda0]).ising_model) 
    objective_hamiltonian = norm_model.hamiltonian
    device_local = create_device(location='local', name=args.simName)
    qaoa_descriptor = QAOADescriptor(cost_hamiltonian = objective_hamiltonian, mixer_block = mix_hamiltonian, p=1)
    
    backend_local = get_qaoa_backend(qaoa_descriptor, device_local, n_shots=args.nShots)
    #To create a Variational Parameter Class with the Standard Parameterisation and Random Initialisation
    variate_params = create_qaoa_variational_params(qaoa_descriptor = qaoa_descriptor, params_type = 'standard', init_type = 'rand')
    optimizer = get_optimizer(backend_local, variate_params, {
        'method': 'pennylane_adam',
        'jac': "param_shift",
        'optimizer_options': dict(stepsize=0.01, beta1=0.9, beta2=0.99, eps=1e-08),
        'maxiter': 100})

    return optimizer, backend_local, objective_hamiltonian

#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

    def benchmark(self, args, MD, figId=1):
        """
        Benchmark a specific method for solving the knapsack problem.

        Returns
        -------
        result : dict
            Results of the QAOA run, including bitstring and objective value.
        """
        # Define the QAOA problem
        nrow,ncol=1,1       
        figId=self.smart_append(figId)
        # fig=self.plt.figure(figId,facecolor='white',figsize=(5.5,7))        
        ax = self.plt.subplot(nrow,ncol,1)
        # TODO plot problem 
        method = MD['method']
        _hashlib = MD['hash']
        qaoa, backend, objective_hamiltonian = define_qaoa_problem(args, method)
        if args.simName == "classic":
            energy, configuration = ground_state_hamiltonian(objective_hamiltonian)
            # print(f"Ground State energy: {energy}, Solution: {configuration}")
            method = "classic"
            result = {"energy": energy, "configuration": configuration}
        # qaoa.compile()
        # Run QAOA
        else:
            # saving circuit 
            qaoa.optimize()
            # result = qaoa.get_results()
            result = qaoa.qaoa_result.most_probable_states
            opt_results = qaoa.qaoa_result
            # print the cost history
            fig, ax = opt_results.plot_cost()
            fig.savefig(f"{args.outPath}cost_history_{method}_{_hashlib}.png")
            # saving circuits
            variational_params = qaoa.variational_params
            optimized_angles = opt_results.optimized['angles']
            variational_params.update_from_raw(optimized_angles)
            optimized_circuit = backend.qaoa_circuit(variational_params)
            if (args.verb > 1):
                print(optimized_circuit)
            with open(MD['qpyPath'], "wb") as fd:
                qpy.dump(optimized_circuit, fd)
                print("saved optimized circuit")
        # print(f"QAOA Result: {result}")   
        return {
        "method": method,
        "result": result,
        # "objective_value": objective_value,
        # "constraint_satisfied": total_weight <= capacity,
        }

if __name__ == "__main__":
    args = get_parser()
    method = args.penMed
    num_iterations = args.iterate
    _hashlib = hashlib.md5(os.urandom(32)).hexdigest()[:6]
    MD={
        'method': method,
        'hash': _hashlib,
        'qpyPath': f"circ/{method}_{_hashlib}.qpy"
    }
    results = []
    plot = Plotter(args)
    outPath = args.outPath
    for i in range(num_iterations):
        print(f"Running iteration {i+1}/{num_iterations}...")
        res = plot.benchmark(args, MD)
        res["iteration"] = i+1
        results.append(res)
    results.append({'simulator': args.simName})
    # Generate a filename based on method, simulator, and timestamp
    filename = f"benchmark_results_{method}_{args.simName}_{_hashlib}.yaml"

    # Save results to a YAML file
    with open(outPath+filename, "w") as file:
        yaml.dump(results, file)
    print("conda activate qiskit")
    print("for ziqing: switch from qaoa to latest qiskit")
    print("conda activate /pscratch/sd/g/gzquse/cudaq")
    print(f"./np_backends.py  --prjName qpy --infPath {MD['qpyPath']}")
    # plot.display_all(png=1)

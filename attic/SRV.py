#!/usr/bin/env python3
import pennylane as qml
import numpy as np
from scipy.linalg import svd

# Function to perform Schmidt decomposition
def schmidt_decomposition(state_vector, system_dims):
    """
    Perform Schmidt decomposition on a given state vector.
    
    Args:
        state_vector (array): The quantum state vector (complex amplitudes).
        system_dims (tuple): Dimensions of the subsystems (e.g., (2, 2) for 2 qubits).
    
    Returns:
        Schmidt coefficients and their count (Schmidt rank).
    """
    # Reshape the state vector into a matrix
    reshaped = state_vector.reshape(system_dims)
    
    # Perform Singular Value Decomposition (SVD)
    u, singular_values, vh = svd(reshaped)
    
    # Schmidt rank corresponds to the number of non-zero singular values
    schmidt_rank = np.sum(singular_values > 1e-10)
    
    return singular_values, schmidt_rank

# Define a quantum circuit using PennyLane
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

# Generate the quantum state
state = quantum_circuit()

# Compute Schmidt decomposition and Schmidt Rank Vector (SRV)
schmidt_coeffs, schmidt_rank = schmidt_decomposition(state, (2, 2))

# Print results
print(f"Quantum State:\n{state}")
print(f"Schmidt Coefficients: {schmidt_coeffs}")
print(f"Schmidt Rank: {schmidt_rank}")

#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Sample data: Replace with actual performance data
qubits = np.arange(5, 21)  # Number of qubits from 5 to 20
libraries = ['Library A', 'Library B', 'Library C']  # Replace with actual library names
colors = ['b', 'g', 'r']  # Colors for different libraries

# Simulated wall-clock times (in seconds) for each library, circuit type, and setup
# Replace these with actual performance measurements
def simulate_time(lib, circuit, setup, qubits):
    np.random.seed(hash(lib + circuit + setup) % 1000)
    base_time = np.exp2(qubits / 2)  # Exponential growth with qubits
    noise = np.random.normal(0, 0.1, size=qubits.shape)
    return base_time * (1 + noise)

# Circuit types and computational setups
circuit_types = ['Heisenberg Dynamics', 'Random Quantum Circuit', 'Quantum Fourier Transform']
setups = ['CPU Multithread', 'QPU', 'GPU']

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15), sharex=True, sharey=True)
fig.suptitle('Performance Comparison of Quantum Simulation Libraries', fontsize=16)

for i, circuit in enumerate(circuit_types):
    for j, setup in enumerate(setups):
        ax = axes[i, j]
        for lib, color in zip(libraries, colors):
            times = simulate_time(lib, circuit, setup, qubits)
            ax.plot(qubits, times, label=lib, color=color)
        ax.set_yscale('log')
        ax.set_title(f'{circuit} - {setup}')
        ax.set_xlabel('Number of Qubits')
        ax.set_ylabel('Wall-Clock Time (s)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a single legend for all subplots
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(libraries), fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

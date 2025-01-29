import numpy as np
import matplotlib.pyplot as plt

# Sample data
packages = ['Package A', 'Package B', 'Package C', 'Package D']
benchmarks = ['Heisenberg Dynamics', 'Random Quantum Circuit', 'QFT']

# Randomly generated data for demonstration purposes
np.random.seed(0)
e_a_over_e_asimcirq = np.random.rand(len(packages), len(benchmarks)) + 1
e_a_over_e_asimcirq_err = np.random.rand(len(packages), len(benchmarks)) * 0.1
scaling_overhead_b = np.random.rand(len(packages), len(benchmarks)) + 0.5
scaling_overhead_b_err = np.random.rand(len(packages), len(benchmarks)) * 0.05

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=len(benchmarks), figsize=(15, 10), sharey='row')

# Plot e_a / e_asimcirq
for i, benchmark in enumerate(benchmarks):
    ax = axes[0, i]
    for j, package in enumerate(packages):
        ax.errorbar(j, e_a_over_e_asimcirq[j, i], yerr=e_a_over_e_asimcirq_err[j, i], fmt='o', label=package)
    ax.axhline(y=1, color='green', linestyle='--', label='simcirq baseline')
    ax.set_title(f'{benchmark}\n$e_a / e_{{a_{{simcirq}}}}$')
    ax.set_xticks(range(len(packages)))
    ax.set_xticklabels(packages)
    if i == 0:
        ax.set_ylabel('$e_a / e_{a_{simcirq}}$')
    if i == len(benchmarks) - 1:
        ax.legend()

# Plot scaling overhead b
for i, benchmark in enumerate(benchmarks):
    ax = axes[1, i]
    for j, package in enumerate(packages):
        ax.errorbar(j, scaling_overhead_b[j, i], yerr=scaling_overhead_b_err[j, i], fmt='o', label=package)
    ax.axhline(y=0.7, color='red', linestyle='--', label='Reference line')
    ax.set_title(f'{benchmark}\nScaling Overhead $b$')
    ax.set_xticks(range(len(packages)))
    ax.set_xticklabels(packages)
    if i == 0:
        ax.set_ylabel('Scaling Overhead $b$')
    if i == len(benchmarks) - 1:
        ax.legend()

plt.tight_layout()
plt.show()

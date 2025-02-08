#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:00:56 2022

@author: Alejandro Montanez-Barrera
"""

from typing import Dict, List, Optional, Union
from docplex.mp.model import Model
from copy import copy
import multiprocessing
import numpy as np
import itertools
from qiskit_optimization.translators import to_docplex_mp, from_docplex_mp
import networkx as nx

num_cores = multiprocessing.cpu_count()


def TSP(G: nx.Graph()) -> Model:
    """
    Traveling salesman problem (TSP) docplex model from a graph. https://en.wikipedia.org/wiki/Travelling_salesman_problem
    
    Parameters
    ----------
    G : nx.Graph()
        Networx graph of the TSP.

    Returns
    -------
    Model
        Docplex model of the TSP.

    """
    mdl = Model(name="TSP")
    cities = G.number_of_nodes()
    x = {
        (i, j): mdl.binary_var(name=f"x_{i}_{j}")
        for i in range(cities)
        for j in range(cities)
        if i != j
    }

    mdl.minimize(
        mdl.sum(
            G.edges[i, j]["weight"] * x[(i, j)]
            for i in range(cities)
            for j in range(cities)
            if i != j
        )
    )
    # Only 1 edge goes out from each node
    for i in range(cities):
        mdl.add_constraint(mdl.sum(x[i, j] for j in range(cities) if i != j) == 1)
    # Only 1 edge comes into each node
    for j in range(cities):
        mdl.add_constraint(mdl.sum(x[i, j] for i in range(cities) if i != j) == 1)

    # To eliminate sub-tours
    cities_list = list(range(1, cities))
    possible_subtours = []
    for i in range(2, len(cities_list) + 1):
        for comb in itertools.combinations(cities_list, i):
            possible_subtours.append(list(comb))
    for subtour in possible_subtours:
        mdl.add_constraint(
            mdl.sum(x[(i, j)] for i in subtour for j in subtour if i != j)
            <= len(subtour) - 1
        )
    return mdl


def BPP(
    weights: List[int], max_weight: int, simplifications: Optional[bool] = False
) -> Model:
    """
    Bin Packing Problem (BPP) docplex model. https://en.wikipedia.org/wiki/Bin_packing_problem

    Parameters
    ----------
    weights : List[int]
        weights of the items.
    max_weight : int
        Maximum weight of the bins.
    simplifications : Bool, optional
        If simplifications about assigne the first item to the first bin and 
        the known minimum number of bins. The default is False.

    Returns
    -------
    mdl : Model
        Docplex model of the BPP.

    """
    num_bins = num_items = len(weights)
    mdl = Model("BinPacking")

    y = mdl.binary_var_list(
        num_bins, name="y"
    )  # list of variables that represent the bins
    x = mdl.binary_var_matrix(
        num_items, num_bins, "x"
    )  # variables that represent the items on the specific bin

    objective = mdl.sum(y)

    mdl.minimize(objective)

    for i in range(num_items):
        # First set of constraints: the items must be in any bin
        mdl.add_constraint(mdl.sum(x[i, j] for j in range(num_bins)) == 1)

    for j in range(num_bins):
        # Second set of constraints: weight constraints
        mdl.add_constraint(
            mdl.sum(weights[i] * x[i, j] for i in range(num_items)) <= max_weight * y[j]
        )

    if simplifications:
        qp = from_docplex_mp(mdl)
        l = int(np.ceil(np.sum(weights) / max_weight))
        qp = qp.substitute_variables(
            {f"y_{_}": 1 for _ in range(l)}
        )  # First simplification: we know the minimum number of bins
        qp = qp.substitute_variables(
            {"x_0_0": 1}
        )  # Assign the first item into the first bin
        qp = qp.substitute_variables(
            {f"x_0_{_}": 0 for _ in range(1, num_bins)}
        )  # because the first item is in the first bin
        mdl = to_docplex_mp(qp)
    return mdl


def KP(values: List[int], weights: List[int], max_weight: int) -> Model:
    """
    Knapsack problem (KP) docplex model. https://en.wikipedia.org/wiki/Knapsack_problem

    Parameters
    ----------
    values : List[int]
        values of the items that can be stored in the knapsack.
    weights : List[int]
        weights of the items.
    max_weight : int
        Maximum weight the knapsack can store.

    Returns
    -------
    Model
        docplex model of the KP.

    """
    mdl = Model("Knapsack")

    x = {
        i: mdl.binary_var(name=f"x_{i}") for i in range(len(values))
    }  # variables that represent the items

    mdl.maximize(
        mdl.sum(values[i] * x[i] for i in x)
    )  # indicate the objective function

    mdl.add_constraint(
        mdl.sum(weights[i] * x[i] for i in x) <= max_weight
    )  # add  the constraint for knapsack

    return mdl


def normalization(problem, normalized=-1, periodic=False):
    """
    

    Parameters
    ----------
    problem : TYPE
        DESCRIPTION.
    normalized : TYPE, optional
        DESCRIPTION. The default is -1.
    periodic : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    new_problem : TYPE
        DESCRIPTION.

    """
    abs_weights = np.unique(np.abs(problem.weights))
    arg_sort = np.argsort(abs_weights)
    max_weight = abs_weights[arg_sort[normalized]]
    new_problem = copy(problem)
    if periodic:
        new_problem.weights = [weight // max_weight for weight in new_problem.weights]
    else:
        new_problem.weights = [weight / max_weight for weight in new_problem.weights]
    new_problem.constant /= max_weight
    return new_problem

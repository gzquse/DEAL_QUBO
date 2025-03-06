#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@email: ziqinguse@gmail.com
@author: Ziqing Guo
"""

from typing import Dict, List, Optional, Union
from docplex.mp.model import Model
from copy import copy
import multiprocessing
import numpy as np
import itertools
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

def MAXCUT(G: nx.Graph()) -> Model:
    """
    Max Cut problem docplex model from a graph.
    Given an undirected graph G with edge weights (specified in the attribute "weight"),
    the objective is to partition the nodes into two sets such that the sum of the weights
    of the edges crossing the partition is maximized.

    Parameters
    ----------
    G : nx.Graph()
        Networkx graph of the Max Cut problem.

    Returns
    -------
    Model
        Docplex model for the Max Cut problem.
    """
    mdl = Model(name="MaxCut")
    
    # Create a binary variable for each node indicating its partition (0 or 1)
    x = {node: mdl.binary_var(name=f"x_{node}") for node in G.nodes()}
    
    # Create a binary variable for each edge indicating if the edge is in the cut.
    # For an edge (i, j), y_{i,j} should be 1 if x_i != x_j and 0 otherwise.
    y = {}
    for i, j in G.edges():
        y[(i, j)] = mdl.binary_var(name=f"y_{i}_{j}")
        # These constraints force y[(i,j)] to equal |x[i] - x[j]|
        mdl.add_constraint(y[(i, j)] >= x[i] - x[j])
        mdl.add_constraint(y[(i, j)] >= x[j] - x[i])
        mdl.add_constraint(y[(i, j)] <= x[i] + x[j])
        mdl.add_constraint(y[(i, j)] <= 2 - (x[i] + x[j]))
    
    # Objective: maximize the total weight of edges that are cut
    mdl.maximize(mdl.sum(G.edges[i, j]["weight"] * y[(i, j)] for i, j in G.edges()))
    
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

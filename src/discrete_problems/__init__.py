"""
Discrete Optimization Problems
"""

from .tsp import TSP, TSPSolver
from .knapsack import Knapsack, KnapsackSolver
from .graph_coloring import GraphColoring, GraphColoringSolver

__all__ = [
    'TSP', 'TSPSolver',
    'Knapsack', 'KnapsackSolver',
    'GraphColoring', 'GraphColoringSolver'
]


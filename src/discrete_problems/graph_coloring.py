"""
Graph Coloring Problem
Assign colors to vertices so that no adjacent vertices have the same color
Minimize the number of colors used
"""

import numpy as np


class GraphColoring:
    """
    Graph Coloring Problem
    
    Parameters:
    -----------
    n_vertices : int
        Number of vertices
    edges : list of tuples, optional
        List of edges (i, j)
        If None, random graph will be generated
    edge_probability : float, optional
        Probability of edge between vertices (for random graph)
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_vertices=20, edges=None, edge_probability=0.3, seed=None):
        self.n_vertices = n_vertices
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize adjacency matrix
        self.adj_matrix = np.zeros((n_vertices, n_vertices), dtype=int)
        
        if edges is not None:
            # Use provided edges
            for i, j in edges:
                self.adj_matrix[i, j] = 1
                self.adj_matrix[j, i] = 1
            self.edges = edges
        else:
            # Generate random graph
            self.edges = []
            for i in range(n_vertices):
                for j in range(i + 1, n_vertices):
                    if np.random.rand() < edge_probability:
                        self.adj_matrix[i, j] = 1
                        self.adj_matrix[j, i] = 1
                        self.edges.append((i, j))
    
    def evaluate(self, coloring):
        """
        Evaluate a graph coloring solution
        
        Parameters:
        -----------
        coloring : numpy.ndarray or list
            Array where coloring[i] is the color of vertex i
            Colors should be integers >= 0
        
        Returns:
        --------
        score : float
            Number of conflicts + number of colors used
            (Lower is better - minimization problem)
        """
        coloring = np.array(coloring, dtype=int)
        
        # Count conflicts (adjacent vertices with same color)
        conflicts = 0
        for i, j in self.edges:
            if coloring[i] == coloring[j]:
                conflicts += 1
        
        # Count number of colors used
        n_colors = len(np.unique(coloring))
        
        # Objective: minimize conflicts (high penalty) + colors used
        return conflicts * 1000 + n_colors
    
    def is_valid(self, coloring):
        """Check if coloring is valid (no conflicts)"""
        coloring = np.array(coloring, dtype=int)
        
        for i, j in self.edges:
            if coloring[i] == coloring[j]:
                return False
        
        return True
    
    def count_conflicts(self, coloring):
        """Count number of conflicts in coloring"""
        coloring = np.array(coloring, dtype=int)
        
        conflicts = 0
        for i, j in self.edges:
            if coloring[i] == coloring[j]:
                conflicts += 1
        
        return conflicts
    
    def count_colors(self, coloring):
        """Count number of colors used"""
        coloring = np.array(coloring, dtype=int)
        return len(np.unique(coloring))
    
    def chromatic_number_upper_bound(self):
        """
        Get upper bound on chromatic number (greedy coloring)
        """
        # Maximum degree + 1 is always an upper bound
        degrees = np.sum(self.adj_matrix, axis=1)
        return int(np.max(degrees)) + 1
    
    def __call__(self, coloring):
        """Allow graph to be called as a function"""
        return self.evaluate(coloring)
    
    def summary(self):
        """Print problem summary"""
        degrees = np.sum(self.adj_matrix, axis=1)
        print(f"Graph Coloring Problem Summary:")
        print(f"  Number of vertices: {self.n_vertices}")
        print(f"  Number of edges: {len(self.edges)}")
        print(f"  Edge density: {len(self.edges) / (self.n_vertices * (self.n_vertices - 1) / 2):.3f}")
        print(f"  Max degree: {int(np.max(degrees))}")
        print(f"  Average degree: {np.mean(degrees):.2f}")
        print(f"  Upper bound on chromatic number: {self.chromatic_number_upper_bound()}")


class GraphColoringSolver:
    """Collection of algorithms to solve Graph Coloring Problem"""
    
    @staticmethod
    def greedy_coloring(graph):
        """
        Greedy graph coloring algorithm
        
        Returns:
        --------
        coloring : numpy.ndarray
            Color assignment for each vertex
        n_colors : int
            Number of colors used
        """
        n = graph.n_vertices
        coloring = np.full(n, -1, dtype=int)
        
        # Color vertices one by one
        for vertex in range(n):
            # Find colors of adjacent vertices
            adjacent_colors = set()
            for neighbor in range(n):
                if graph.adj_matrix[vertex, neighbor] == 1 and coloring[neighbor] != -1:
                    adjacent_colors.add(coloring[neighbor])
            
            # Assign smallest available color
            color = 0
            while color in adjacent_colors:
                color += 1
            
            coloring[vertex] = color
        
        return coloring, len(np.unique(coloring))
    
    @staticmethod
    def greedy_dsatur(graph):
        """
        DSatur (Degree of Saturation) algorithm
        Orders vertices dynamically based on saturation degree
        
        Returns:
        --------
        coloring : numpy.ndarray
            Color assignment for each vertex
        n_colors : int
            Number of colors used
        """
        n = graph.n_vertices
        coloring = np.full(n, -1, dtype=int)
        saturation = np.zeros(n, dtype=int)  # Number of different colors in neighborhood
        degrees = np.sum(graph.adj_matrix, axis=1)
        
        # Start with vertex of highest degree
        current = np.argmax(degrees)
        coloring[current] = 0
        
        # Update saturation of neighbors
        for neighbor in range(n):
            if graph.adj_matrix[current, neighbor] == 1:
                saturation[neighbor] = 1
        
        # Color remaining vertices
        for _ in range(n - 1):
            # Choose uncolored vertex with highest saturation
            # Break ties by degree
            uncolored = np.where(coloring == -1)[0]
            
            # Sort by saturation (desc), then by degree (desc)
            scores = saturation[uncolored] * 1000 + degrees[uncolored]
            current = uncolored[np.argmax(scores)]
            
            # Find colors of adjacent vertices
            adjacent_colors = set()
            for neighbor in range(n):
                if graph.adj_matrix[current, neighbor] == 1 and coloring[neighbor] != -1:
                    adjacent_colors.add(coloring[neighbor])
            
            # Assign smallest available color
            color = 0
            while color in adjacent_colors:
                color += 1
            
            coloring[current] = color
            
            # Update saturation of neighbors
            for neighbor in range(n):
                if graph.adj_matrix[current, neighbor] == 1 and coloring[neighbor] == -1:
                    # Count distinct colors in neighborhood
                    neighbor_colors = set()
                    for v in range(n):
                        if graph.adj_matrix[neighbor, v] == 1 and coloring[v] != -1:
                            neighbor_colors.add(coloring[v])
                    saturation[neighbor] = len(neighbor_colors)
        
        return coloring, len(np.unique(coloring))
    
    @staticmethod
    def genetic_algorithm(graph, population_size=100, max_iter=200,
                         mutation_rate=0.1, crossover_rate=0.8, verbose=False):
        """
        Genetic Algorithm for Graph Coloring
        
        Parameters:
        -----------
        graph : GraphColoring
            Graph coloring problem instance
        population_size : int
            Size of population
        max_iter : int
            Maximum number of iterations
        mutation_rate : float
            Probability of mutation
        crossover_rate : float
            Probability of crossover
        verbose : bool
            Print progress
        
        Returns:
        --------
        best_coloring : numpy.ndarray
            Best coloring found
        best_score : float
            Score of best coloring (conflicts * 1000 + n_colors)
        history : dict
            Optimization history
        """
        n = graph.n_vertices
        max_colors = graph.chromatic_number_upper_bound()
        
        # Initialize population with greedy coloring + random
        population = []
        
        # Add greedy solutions
        greedy_sol, _ = GraphColoringSolver.greedy_coloring(graph)
        population.append(greedy_sol)
        
        dsatur_sol, _ = GraphColoringSolver.greedy_dsatur(graph)
        population.append(dsatur_sol)
        
        # Fill with random colorings
        for _ in range(population_size - 2):
            coloring = np.random.randint(0, max_colors, n)
            population.append(coloring)
        
        population = np.array(population)
        
        best_history = []
        mean_history = []
        conflicts_history = []
        colors_history = []
        
        best_coloring = None
        best_score = np.inf
        
        for iteration in range(max_iter):
            # Evaluate population
            scores = np.array([graph.evaluate(ind) for ind in population])
            
            # Track best
            current_best_idx = np.argmin(scores)
            current_best_score = scores[current_best_idx]
            
            if current_best_score < best_score:
                best_score = current_best_score
                best_coloring = population[current_best_idx].copy()
            
            best_history.append(best_score)
            mean_history.append(np.mean(scores))
            conflicts_history.append(graph.count_conflicts(best_coloring))
            colors_history.append(graph.count_colors(best_coloring))
            
            if verbose and iteration % 20 == 0:
                n_conflicts = graph.count_conflicts(best_coloring)
                n_colors = graph.count_colors(best_coloring)
                print(f"Iter {iteration}: Score={best_score:.2f}, "
                      f"Conflicts={n_conflicts}, Colors={n_colors}")
            
            # Selection (tournament)
            selected = []
            for _ in range(population_size):
                i1, i2 = np.random.choice(population_size, 2, replace=False)
                winner = i1 if scores[i1] < scores[i2] else i2  # Minimization
                selected.append(population[winner].copy())
            
            # Crossover
            offspring = []
            for i in range(0, population_size, 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < population_size else selected[0]
                
                if np.random.rand() < crossover_rate:
                    # Single-point crossover
                    point = np.random.randint(1, n)
                    child1 = np.concatenate([parent1[:point], parent2[point:]])
                    child2 = np.concatenate([parent2[:point], parent1[point:]])
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                
                offspring.append(child1)
                offspring.append(child2)
            
            offspring = offspring[:population_size]
            
            # Mutation
            for i in range(population_size):
                if np.random.rand() < mutation_rate:
                    # Change color of random vertex
                    vertex = np.random.randint(n)
                    offspring[i][vertex] = np.random.randint(0, max_colors)
            
            population = np.array(offspring)
            
            # Elitism: keep best solution
            population[0] = best_coloring.copy()
            
            # Early stopping if valid coloring found
            if graph.count_conflicts(best_coloring) == 0:
                if verbose:
                    print(f"Valid coloring found at iteration {iteration}!")
                break
        
        history = {
            'best_scores': best_history,
            'mean_scores': mean_history,
            'conflicts': conflicts_history,
            'n_colors': colors_history
        }
        
        return best_coloring, best_score, history
    
    @staticmethod
    def hill_climbing(graph, max_iter=1000, verbose=False):
        """
        Hill Climbing for Graph Coloring
        
        Returns:
        --------
        best_coloring : numpy.ndarray
            Best coloring found
        best_score : float
            Score of best coloring
        history : dict
            Optimization history
        """
        n = graph.n_vertices
        
        # Start with RANDOM solution instead of greedy for more diversity
        max_colors = graph.chromatic_number_upper_bound()
        current_coloring = np.random.randint(0, max_colors, n)
        current_score = graph.evaluate(current_coloring)
        
        best_coloring = current_coloring.copy()
        best_score = current_score
        
        history = []
        max_colors = graph.chromatic_number_upper_bound()
        
        for iteration in range(max_iter):
            improved = False
            
            # Try changing color of each vertex
            for vertex in range(n):
                for color in range(max_colors):
                    if color != current_coloring[vertex]:
                        neighbor = current_coloring.copy()
                        neighbor[vertex] = color
                        
                        neighbor_score = graph.evaluate(neighbor)
                        
                        if neighbor_score < current_score:
                            current_coloring = neighbor.copy()
                            current_score = neighbor_score
                            improved = True
                            
                            if current_score < best_score:
                                best_coloring = current_coloring.copy()
                                best_score = current_score
                            
                            break
                
                if improved:
                    break
            
            history.append(best_score)
            
            if not improved:
                break
            
            if verbose and iteration % 100 == 0:
                n_conflicts = graph.count_conflicts(best_coloring)
                n_colors = graph.count_colors(best_coloring)
                print(f"Iter {iteration}: Score={best_score:.2f}, "
                      f"Conflicts={n_conflicts}, Colors={n_colors}")
        
        return best_coloring, best_score, {'best_scores': history}
    
    @staticmethod
    def simulated_annealing(graph, max_iter=1000, initial_temp=100,
                           cooling_rate=0.95, verbose=False):
        """
        Simulated Annealing for Graph Coloring
        
        Returns:
        --------
        best_coloring : numpy.ndarray
            Best coloring found
        best_score : float
            Score of best coloring
        history : dict
            Optimization history
        """
        n = graph.n_vertices
        max_colors = graph.chromatic_number_upper_bound()
        
        # Start with RANDOM solution instead of greedy for more diversity
        current_coloring = np.random.randint(0, max_colors, n)
        current_score = graph.evaluate(current_coloring)
        
        best_coloring = current_coloring.copy()
        best_score = current_score
        
        temperature = initial_temp
        history = []
        
        for iteration in range(max_iter):
            # Generate neighbor by changing color of random vertex
            neighbor = current_coloring.copy()
            vertex = np.random.randint(n)
            neighbor[vertex] = np.random.randint(0, max_colors)
            
            neighbor_score = graph.evaluate(neighbor)
            delta = neighbor_score - current_score
            
            # Accept if better or with probability
            if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                current_coloring = neighbor.copy()
                current_score = neighbor_score
                
                if current_score < best_score:
                    best_coloring = current_coloring.copy()
                    best_score = current_score
            
            temperature *= cooling_rate
            history.append(best_score)
            
            if verbose and iteration % 100 == 0:
                n_conflicts = graph.count_conflicts(best_coloring)
                n_colors = graph.count_colors(best_coloring)
                print(f"Iter {iteration}: Score={best_score:.2f}, "
                      f"Conflicts={n_conflicts}, Colors={n_colors}, Temp={temperature:.2f}")
        
        return best_coloring, best_score, {'best_scores': history}


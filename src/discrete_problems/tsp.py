"""
Traveling Salesman Problem (TSP)
Find the shortest route visiting all cities exactly once and returning to start
"""

import numpy as np


class TSP:
    """
    Traveling Salesman Problem
    
    Parameters:
    -----------
    n_cities : int
        Number of cities
    distance_matrix : numpy.ndarray, optional
        Pre-defined distance matrix, shape (n_cities, n_cities)
        If None, random cities will be generated
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_cities=20, distance_matrix=None, seed=None):
        self.n_cities = n_cities
        
        if seed is not None:
            np.random.seed(seed)
        
        if distance_matrix is not None:
            self.distance_matrix = distance_matrix
            self.n_cities = len(distance_matrix)
        else:
            # Generate random cities in 2D plane
            self.cities = np.random.rand(n_cities, 2) * 100
            self.distance_matrix = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self):
        """Compute Euclidean distance matrix between all cities"""
        n = len(self.cities)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i, j] = np.sqrt(
                        np.sum((self.cities[i] - self.cities[j]) ** 2)
                    )
        
        return distances
    
    def evaluate(self, tour):
        """
        Evaluate the total distance of a tour
        
        Parameters:
        -----------
        tour : list or numpy.ndarray
            Order of cities to visit
            
        Returns:
        --------
        distance : float
            Total tour distance
        """
        total_distance = 0
        n = len(tour)
        
        for i in range(n):
            city_from = tour[i]
            city_to = tour[(i + 1) % n]
            total_distance += self.distance_matrix[city_from, city_to]
        
        return total_distance
    
    def get_cities(self):
        """Get city coordinates (if available)"""
        return getattr(self, 'cities', None)
    
    def get_distance_matrix(self):
        """Get distance matrix"""
        return self.distance_matrix


class TSPSolver:
    """
    Collection of TSP solving algorithms
    """
    
    @staticmethod
    def ant_colony_optimization(tsp, n_ants=20, max_iter=100, alpha=1.0, beta=2.0, rho=0.5, Q=100):
        """
        Ant Colony Optimization (ACO) for TSP
        Classic Ant System (AS) algorithm
        
        Parameters:
        -----------
        tsp : TSP
            TSP problem instance
        n_ants : int
            Number of ants
        max_iter : int
            Maximum number of iterations
        alpha : float
            Pheromone importance (α)
        beta : float
            Heuristic importance (β) - distance heuristic
        rho : float
            Pheromone evaporation rate (ρ), 0 < ρ < 1
        Q : float
            Pheromone deposit constant
            
        Returns:
        --------
        best_tour : list
            Best tour found
        best_distance : float
            Distance of best tour
        history : dict
            Optimization history with pheromone trails
        """
        n_cities = tsp.n_cities
        distance_matrix = tsp.distance_matrix
        
        # Initialize pheromone matrix (uniform)
        pheromone = np.ones((n_cities, n_cities)) * 0.1
        
        # Heuristic information (inverse of distance)
        # η[i][j] = 1 / distance[i][j] (closer = better)
        heuristic = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j and distance_matrix[i, j] > 0:
                    heuristic[i, j] = 1.0 / distance_matrix[i, j]
        
        # History
        best_tour = None
        best_distance = np.inf
        best_distances_history = []
        mean_distances_history = []
        pheromone_history = []
        
        for iteration in range(max_iter):
            # All ants construct tours
            all_tours = []
            all_distances = []
            
            for ant in range(n_ants):
                # Each ant starts from a random city
                current_city = np.random.randint(0, n_cities)
                tour = [current_city]
                unvisited = set(range(n_cities))
                unvisited.remove(current_city)
                
                # Construct tour
                while unvisited:
                    # Calculate probabilities for next city selection
                    probabilities = []
                    unvisited_list = list(unvisited)
                    
                    for next_city in unvisited_list:
                        # P[i][j] = (τ[i][j]^α) * (η[i][j]^β)
                        tau = pheromone[current_city, next_city] ** alpha
                        eta = heuristic[current_city, next_city] ** beta
                        probabilities.append(tau * eta)
                    
                    # Normalize probabilities
                    probabilities = np.array(probabilities)
                    sum_prob = np.sum(probabilities)
                    
                    if sum_prob > 0:
                        probabilities = probabilities / sum_prob
                    else:
                        # Fallback: uniform probability
                        probabilities = np.ones(len(unvisited_list)) / len(unvisited_list)
                    
                    # Select next city based on probabilities
                    next_city_idx = np.random.choice(len(unvisited_list), p=probabilities)
                    next_city = unvisited_list[next_city_idx]
                    
                    tour.append(next_city)
                    unvisited.remove(next_city)
                    current_city = next_city
                
                # Evaluate tour
                tour_distance = tsp.evaluate(tour)
                all_tours.append(tour)
                all_distances.append(tour_distance)
                
                # Update best solution
                if tour_distance < best_distance:
                    best_distance = tour_distance
                    best_tour = tour.copy()
            
            # Pheromone evaporation
            pheromone = (1 - rho) * pheromone
            
            # Pheromone deposit
            for ant_idx, tour in enumerate(all_tours):
                tour_distance = all_distances[ant_idx]
                # Deposit amount inversely proportional to tour length
                deposit = Q / tour_distance
                
                for i in range(n_cities):
                    city_from = tour[i]
                    city_to = tour[(i + 1) % n_cities]
                    pheromone[city_from, city_to] += deposit
                    pheromone[city_to, city_from] += deposit  # Symmetric
            
            # Record history
            best_distances_history.append(best_distance)
            mean_distances_history.append(np.mean(all_distances))
            pheromone_history.append(pheromone.copy())
        
        history = {
            'best_distances': np.array(best_distances_history),
            'mean_distances': np.array(mean_distances_history),
            'pheromone_trails': pheromone_history,
            'final_pheromone': pheromone
        }
        
        return best_tour, best_distance, history
    
    @staticmethod
    def nearest_neighbor(tsp, start_city=0):
        """
        Nearest Neighbor heuristic for TSP
        Greedy algorithm that always visits the nearest unvisited city
        
        Parameters:
        -----------
        tsp : TSP
            TSP problem instance
        start_city : int
            Starting city index
            
        Returns:
        --------
        tour : list
            Tour as list of city indices
        distance : float
            Total tour distance
        """
        n = tsp.n_cities
        unvisited = set(range(n))
        current = start_city
        tour = [current]
        unvisited.remove(current)
        
        while unvisited:
            # Find nearest unvisited city
            nearest = min(unvisited, 
                         key=lambda city: tsp.distance_matrix[current, city])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        distance = tsp.evaluate(tour)
        return tour, distance
    
    @staticmethod
    def two_opt(tsp, initial_tour=None, max_iter=1000):
        """
        2-opt local search for TSP
        Iteratively improves tour by reversing segments
        
        Parameters:
        -----------
        tsp : TSP
            TSP problem instance
        initial_tour : list, optional
            Initial tour, if None uses nearest neighbor
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        tour : list
            Improved tour
        distance : float
            Total tour distance
        """
        # Get initial tour
        if initial_tour is None:
            tour, _ = TSPSolver.nearest_neighbor(tsp)
        else:
            tour = list(initial_tour)
        
        n = len(tour)
        improved = True
        iteration = 0
        
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Try reversing segment [i:j+1]
                    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                    
                    # Check if improvement
                    if tsp.evaluate(new_tour) < tsp.evaluate(tour):
                        tour = new_tour
                        improved = True
                        break
                
                if improved:
                    break
        
        distance = tsp.evaluate(tour)
        return tour, distance
    
    @staticmethod
    def genetic_algorithm_tsp(tsp, population_size=100, max_iter=500,
                             mutation_rate=0.01, crossover_rate=0.8):
        """
        Genetic Algorithm for TSP
        
        Parameters:
        -----------
        tsp : TSP
            TSP problem instance
        population_size : int
            Number of tours in population
        max_iter : int
            Number of generations
        mutation_rate : float
            Probability of mutation
        crossover_rate : float
            Probability of crossover
            
        Returns:
        --------
        best_tour : list
            Best tour found
        best_distance : float
            Distance of best tour
        history : dict
            Optimization history
        """
        n = tsp.n_cities
        
        # Initialize population with random tours
        def random_tour():
            tour = list(range(n))
            np.random.shuffle(tour)
            return tour
        
        population = [random_tour() for _ in range(population_size)]
        
        # History
        best_distances = []
        mean_distances = []
        
        def ordered_crossover(parent1, parent2):
            """Ordered Crossover (OX)"""
            if np.random.rand() > crossover_rate:
                return parent1.copy()
            
            size = len(parent1)
            # Select random segment
            start, end = sorted(np.random.choice(size, 2, replace=False))
            
            # Copy segment from parent1
            child = [-1] * size
            child[start:end] = parent1[start:end]
            
            # Fill remaining from parent2
            pos = end
            for city in parent2[end:] + parent2[:end]:
                if city not in child:
                    if pos >= size:
                        pos = 0
                    child[pos] = city
                    pos += 1
            
            return child
        
        def swap_mutation(tour):
            """Swap two random cities"""
            if np.random.rand() > mutation_rate:
                return tour
            
            tour = tour.copy()
            i, j = np.random.choice(len(tour), 2, replace=False)
            tour[i], tour[j] = tour[j], tour[i]
            return tour
        
        # Evolution
        best_tour = None
        best_distance = np.inf
        
        for generation in range(max_iter):
            # Evaluate fitness
            fitness = np.array([tsp.evaluate(tour) for tour in population])
            
            # Update best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_distance:
                best_distance = fitness[gen_best_idx]
                best_tour = population[gen_best_idx].copy()
            
            # Record history
            best_distances.append(best_distance)
            mean_distances.append(np.mean(fitness))
            
            # Selection (tournament)
            def tournament_selection(k=3):
                tournament = np.random.choice(population_size, k, replace=False)
                winner = tournament[np.argmin(fitness[tournament])]
                return population[winner]
            
            # Create new population
            new_population = []
            
            # Elitism: keep best
            elite_count = max(1, population_size // 10)
            elite_indices = np.argsort(fitness)[:elite_count]
            new_population.extend([population[i].copy() for i in elite_indices])
            
            # Generate offspring
            while len(new_population) < population_size:
                parent1 = tournament_selection()
                parent2 = tournament_selection()
                
                child = ordered_crossover(parent1, parent2)
                child = swap_mutation(child)
                
                new_population.append(child)
            
            population = new_population
        
        history = {
            'best_distances': np.array(best_distances),
            'mean_distances': np.array(mean_distances)
        }
        
        return best_tour, best_distance, history


if __name__ == "__main__":
    # Test TSP
    print("Testing TSP Solvers")
    print("=" * 50)
    
    # Create TSP instance
    tsp = TSP(n_cities=20, seed=42)
    
    # Test Nearest Neighbor
    print("\n1. Nearest Neighbor Heuristic:")
    nn_tour, nn_distance = TSPSolver.nearest_neighbor(tsp)
    print(f"   Distance: {nn_distance:.2f}")
    print(f"   Tour: {nn_tour[:5]}... (showing first 5 cities)")
    
    # Test 2-opt
    print("\n2. 2-opt Local Search:")
    opt_tour, opt_distance = TSPSolver.two_opt(tsp)
    print(f"   Distance: {opt_distance:.2f}")
    print(f"   Improvement: {((nn_distance - opt_distance) / nn_distance * 100):.1f}%")
    
    # Test Genetic Algorithm
    print("\n3. Genetic Algorithm:")
    ga_tour, ga_distance, ga_history = TSPSolver.genetic_algorithm_tsp(
        tsp, population_size=50, max_iter=100
    )
    print(f"   Distance: {ga_distance:.2f}")
    print(f"   Improvement over NN: {((nn_distance - ga_distance) / nn_distance * 100):.1f}%")
    print(f"   Best distance found: {ga_history['best_distances'][-1]:.2f}")


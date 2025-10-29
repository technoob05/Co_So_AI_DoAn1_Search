"""
Artificial Bee Colony (ABC) Algorithm
Inspired by the foraging behavior of honey bees

Reference:
Karaboga, D. (2005). An idea based on honey bee swarm for numerical optimization.
"""

import numpy as np


class ABC:
    """
    Artificial Bee Colony Algorithm
    
    Three types of bees:
    1. Employed bees: exploit food sources
    2. Onlooker bees: select food sources based on quality
    3. Scout bees: explore new food sources
    
    Parameters:
    -----------
    n_bees : int
        Number of employed bees (total population = 2 * n_bees)
    dim : int
        Dimension of the search space
    max_iter : int
        Maximum number of iterations
    limit : int
        Abandonment limit (for scout bees)
    bounds : numpy.ndarray
        Search space bounds, shape (dim, 2)
    """
    
    def __init__(self, n_bees=30, dim=10, max_iter=100, limit=20, bounds=None):
        self.n_bees = n_bees  # Number of employed bees
        self.dim = dim
        self.max_iter = max_iter
        self.limit = limit  # Abandonment limit
        self.bounds = bounds if bounds is not None else np.array([[-100, 100]] * dim)
        
        # History
        self.best_scores_history = []
        self.mean_scores_history = []
        
    def initialize(self):
        """Initialize food sources (solutions)"""
        # Random food sources
        self.food_sources = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.n_bees, self.dim)
        )
        
        # Fitness values
        self.fitness = np.zeros(self.n_bees)
        
        # Trial counters (number of times a source hasn't improved)
        self.trials = np.zeros(self.n_bees, dtype=int)
        
        # Best solution
        self.best_solution = None
        self.best_score = np.inf
        
    def _calculate_fitness(self, score):
        """Calculate fitness from objective value"""
        if score >= 0:
            return 1.0 / (1.0 + score)
        else:
            return 1.0 + np.abs(score)
    
    def _employed_bee_phase(self, objective_function):
        """Employed bees phase: exploit current food sources"""
        for i in range(self.n_bees):
            # Select a random dimension to modify
            phi = np.random.uniform(-1, 1, self.dim)
            
            # Select a random neighbor (different from i)
            neighbor = np.random.choice([j for j in range(self.n_bees) if j != i])
            
            # Generate new solution
            new_solution = self.food_sources[i] + phi * (self.food_sources[i] - self.food_sources[neighbor])
            
            # Apply bounds
            new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
            
            # Evaluate
            new_score = objective_function(new_solution)
            new_fitness = self._calculate_fitness(new_score)
            
            # Greedy selection
            if new_fitness > self.fitness[i]:
                self.food_sources[i] = new_solution
                self.fitness[i] = new_fitness
                self.trials[i] = 0
            else:
                self.trials[i] += 1
    
    def _onlooker_bee_phase(self, objective_function):
        """Onlooker bees phase: select food sources based on probability"""
        # Calculate selection probabilities
        total_fitness = np.sum(self.fitness)
        probabilities = self.fitness / total_fitness if total_fitness > 0 else np.ones(self.n_bees) / self.n_bees
        
        # Each onlooker selects a food source
        for _ in range(self.n_bees):
            # Roulette wheel selection
            i = np.random.choice(self.n_bees, p=probabilities)
            
            # Generate new solution (same as employed bee)
            phi = np.random.uniform(-1, 1, self.dim)
            neighbor = np.random.choice([j for j in range(self.n_bees) if j != i])
            
            new_solution = self.food_sources[i] + phi * (self.food_sources[i] - self.food_sources[neighbor])
            new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
            
            new_score = objective_function(new_solution)
            new_fitness = self._calculate_fitness(new_score)
            
            if new_fitness > self.fitness[i]:
                self.food_sources[i] = new_solution
                self.fitness[i] = new_fitness
                self.trials[i] = 0
            else:
                self.trials[i] += 1
    
    def _scout_bee_phase(self):
        """Scout bees phase: abandon poor food sources and explore new ones"""
        for i in range(self.n_bees):
            if self.trials[i] >= self.limit:
                # Abandon this food source and generate a new one
                self.food_sources[i] = np.random.uniform(
                    self.bounds[:, 0],
                    self.bounds[:, 1],
                    self.dim
                )
                self.trials[i] = 0
    
    def optimize(self, objective_function, verbose=False):
        """
        Run ABC optimization
        
        Parameters:
        -----------
        objective_function : callable
            Function to minimize
        verbose : bool
            Print progress
            
        Returns:
        --------
        best_solution : numpy.ndarray
            Best solution found
        best_score : float
            Best score found
        """
        self.initialize()
        
        # Evaluate initial food sources
        scores = np.array([objective_function(source) for source in self.food_sources])
        self.fitness = np.array([self._calculate_fitness(s) for s in scores])
        
        # Initialize best
        best_idx = np.argmin(scores)
        self.best_score = scores[best_idx]
        self.best_solution = self.food_sources[best_idx].copy()
        
        for iteration in range(self.max_iter):
            # Employed bees phase
            self._employed_bee_phase(objective_function)
            
            # Onlooker bees phase
            self._onlooker_bee_phase(objective_function)
            
            # Scout bees phase
            self._scout_bee_phase()
            
            # Re-evaluate all sources
            scores = np.array([objective_function(source) for source in self.food_sources])
            self.fitness = np.array([self._calculate_fitness(s) for s in scores])
            
            # Update best
            best_idx = np.argmin(scores)
            if scores[best_idx] < self.best_score:
                self.best_score = scores[best_idx]
                self.best_solution = self.food_sources[best_idx].copy()
            
            # Record history
            self.best_scores_history.append(self.best_score)
            self.mean_scores_history.append(np.mean(scores))
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}: "
                      f"Best = {self.best_score:.6f}, "
                      f"Mean = {np.mean(scores):.6f}")
        
        return self.best_solution, self.best_score
    
    def get_history(self):
        """Get convergence history"""
        return {
            'best_scores': np.array(self.best_scores_history),
            'mean_scores': np.array(self.mean_scores_history)
        }


if __name__ == "__main__":
    # Test ABC
    from src.test_functions import get_test_function
    
    func = get_test_function('sphere', dim=10)
    abc = ABC(n_bees=30, dim=10, max_iter=100, bounds=func.bounds)
    
    best_sol, best_score = abc.optimize(func, verbose=True)
    
    print("\n" + "=" * 50)
    print(f"Best solution: {best_sol}")
    print(f"Best score: {best_score:.6f}")
    print(f"Global optimum: {func.global_optimum}")


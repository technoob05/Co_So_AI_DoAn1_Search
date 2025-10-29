"""
Cuckoo Search (CS) Algorithm
Inspired by the brood parasitism of cuckoo species

Reference:
Yang, X. S., & Deb, S. (2009). Cuckoo search via Lévy flights.
"""

import numpy as np
from math import gamma


class CuckooSearch:
    """
    Cuckoo Search Algorithm
    
    Based on three idealized rules:
    1. Each cuckoo lays one egg at a time in a randomly chosen nest
    2. Best nests with high quality eggs carry over to next generation
    3. Number of host nests is fixed, and a host can discover an alien egg
       with probability pa, leading to nest abandonment
    
    Uses Lévy flights for random walk
    
    Parameters:
    -----------
    n_nests : int
        Number of nests (solutions)
    dim : int
        Dimension of the search space
    max_iter : int
        Maximum number of iterations
    pa : float
        Discovery probability (abandon rate)
    beta : float
        Lévy exponent (1 < beta <= 3)
    bounds : numpy.ndarray
        Search space bounds, shape (dim, 2)
    """
    
    def __init__(self, n_nests=30, dim=10, max_iter=100,
                 pa=0.25, beta=1.5, bounds=None):
        self.n_nests = n_nests
        self.dim = dim
        self.max_iter = max_iter
        self.pa = pa  # Discovery probability
        self.beta = beta  # Lévy exponent
        self.bounds = bounds if bounds is not None else np.array([[-100, 100]] * dim)
        
        # History
        self.best_scores_history = []
        self.mean_scores_history = []
        
    def initialize(self):
        """Initialize nests (solutions)"""
        self.nests = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.n_nests, self.dim)
        )
        
        self.fitness = np.zeros(self.n_nests)
        
        # Best nest
        self.best_nest = None
        self.best_score = np.inf
        
    def _levy_flight(self, size):
        """
        Generate Lévy flight step
        
        Lévy flights are random walks where step lengths have a Lévy distribution
        """
        # Mantegna's algorithm for Lévy flights
        sigma_u = (gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / \
                   (gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        sigma_v = 1
        
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, sigma_v, size)
        
        step = u / np.abs(v) ** (1 / self.beta)
        
        return step
    
    def _get_cuckoo(self, iteration):
        """
        Generate new solution (cuckoo) using Lévy flights
        """
        # Randomly select a nest
        idx = np.random.randint(0, self.n_nests)
        
        # Lévy flight
        step_size = 0.01
        step = self._levy_flight(self.dim)
        
        # New solution
        new_nest = self.nests[idx] + step_size * step
        
        # Apply bounds
        new_nest = np.clip(new_nest, self.bounds[:, 0], self.bounds[:, 1])
        
        return new_nest
    
    def _abandon_nests(self):
        """
        Abandon a fraction pa of worst nests
        """
        # Find worst nests
        n_abandon = int(self.pa * self.n_nests)
        
        # Get indices of worst nests
        worst_indices = np.argsort(self.fitness)[-n_abandon:]
        
        # Generate new random nests to replace abandoned ones
        for idx in worst_indices:
            # Random walk
            step = np.random.randn(self.dim)
            K = np.random.rand()
            
            # Get two random nests
            i1, i2 = np.random.choice(self.n_nests, 2, replace=False)
            
            # New nest
            self.nests[idx] = self.nests[idx] + K * (self.nests[i1] - self.nests[i2])
            
            # Apply bounds
            self.nests[idx] = np.clip(self.nests[idx], self.bounds[:, 0], self.bounds[:, 1])
    
    def optimize(self, objective_function, verbose=False):
        """
        Run Cuckoo Search optimization
        
        Parameters:
        -----------
        objective_function : callable
            Function to minimize
        verbose : bool
            Print progress
            
        Returns:
        --------
        best_nest : numpy.ndarray
            Best solution found
        best_score : float
            Best score found
        """
        self.initialize()
        
        # Evaluate initial nests
        self.fitness = np.array([objective_function(nest) for nest in self.nests])
        
        # Initialize best
        best_idx = np.argmin(self.fitness)
        self.best_score = self.fitness[best_idx]
        self.best_nest = self.nests[best_idx].copy()
        
        for iteration in range(self.max_iter):
            # Generate new solutions via Lévy flights
            for i in range(self.n_nests):
                # Get a cuckoo randomly by Lévy flights
                new_nest = self._get_cuckoo(iteration)
                new_fitness = objective_function(new_nest)
                
                # Choose a random nest
                j = np.random.randint(0, self.n_nests)
                
                # Replace if better
                if new_fitness < self.fitness[j]:
                    self.nests[j] = new_nest
                    self.fitness[j] = new_fitness
            
            # Abandon some worse nests
            self._abandon_nests()
            
            # Re-evaluate all nests
            self.fitness = np.array([objective_function(nest) for nest in self.nests])
            
            # Update best
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_score:
                self.best_score = self.fitness[best_idx]
                self.best_nest = self.nests[best_idx].copy()
            
            # Record history
            self.best_scores_history.append(self.best_score)
            self.mean_scores_history.append(np.mean(self.fitness))
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}: "
                      f"Best = {self.best_score:.6f}, "
                      f"Mean = {np.mean(self.fitness):.6f}")
        
        return self.best_nest, self.best_score
    
    def get_history(self):
        """Get convergence history"""
        return {
            'best_scores': np.array(self.best_scores_history),
            'mean_scores': np.array(self.mean_scores_history)
        }


if __name__ == "__main__":
    # Test CS
    from src.test_functions import get_test_function
    
    func = get_test_function('sphere', dim=10)
    cs = CuckooSearch(n_nests=30, dim=10, max_iter=100, bounds=func.bounds)
    
    best_sol, best_score = cs.optimize(func, verbose=True)
    
    print("\n" + "=" * 50)
    print(f"Best solution: {best_sol}")
    print(f"Best score: {best_score:.6f}")
    print(f"Global optimum: {func.global_optimum}")


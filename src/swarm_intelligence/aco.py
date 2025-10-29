"""
Ant Colony Optimization (ACO) for continuous optimization
Adapted from discrete ACO for continuous problems

Reference:
Dorigo, M., & Stützle, T. (2004). Ant colony optimization.
Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains.
"""

import numpy as np


class ACO:
    """
    Ant Colony Optimization for Continuous Domains (ACOR)
    
    Uses Gaussian kernels to model pheromone distribution in continuous space
    
    Parameters:
    -----------
    n_ants : int
        Number of ants in the colony
    dim : int
        Dimension of the search space
    max_iter : int
        Maximum number of iterations
    archive_size : int
        Size of solution archive
    q : float
        Locality of search (smaller = more local)
    xi : float
        Speed of convergence
    bounds : numpy.ndarray
        Search space bounds, shape (dim, 2)
    """
    
    def __init__(self, n_ants=30, dim=10, max_iter=100, 
                 archive_size=50, q=0.5, xi=0.85, bounds=None):
        self.n_ants = n_ants
        self.dim = dim
        self.max_iter = max_iter
        self.archive_size = archive_size
        self.q = q  # Locality parameter
        self.xi = xi  # Convergence speed
        self.bounds = bounds if bounds is not None else np.array([[-100, 100]] * dim)
        
        # History
        self.best_scores_history = []
        self.mean_scores_history = []
        
    def initialize(self):
        """Initialize solution archive"""
        # Random initial solutions
        self.archive = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.archive_size, self.dim)
        )
        self.archive_scores = np.full(self.archive_size, np.inf)
        
        # Best solution
        self.best_solution = None
        self.best_score = np.inf
        
    def _gaussian_kernel(self, mean, std):
        """Sample from Gaussian kernel"""
        return np.random.normal(mean, std)
    
    def _construct_solution(self):
        """Construct a new solution using Gaussian kernel PDF"""
        solution = np.zeros(self.dim)
        
        # Calculate weights for archive solutions
        weights = np.zeros(self.archive_size)
        for i in range(self.archive_size):
            weights[i] = 1.0 / (self.q * self.archive_size * np.sqrt(2 * np.pi)) * \
                        np.exp(-(i ** 2) / (2 * self.q ** 2 * self.archive_size ** 2))
        weights /= np.sum(weights)
        
        # For each dimension
        for d in range(self.dim):
            # Select solution from archive based on weights
            selected_idx = np.random.choice(self.archive_size, p=weights)
            
            # Calculate standard deviation
            sum_distances = 0
            for i in range(self.archive_size):
                sum_distances += np.abs(self.archive[i, d] - self.archive[selected_idx, d])
            std = self.xi / (self.archive_size - 1) * sum_distances
            
            # Sample from Gaussian
            solution[d] = self._gaussian_kernel(self.archive[selected_idx, d], std)
            
            # Apply bounds
            solution[d] = np.clip(solution[d], self.bounds[d, 0], self.bounds[d, 1])
        
        return solution
    
    def _update_archive(self, new_solutions, new_scores):
        """Update solution archive"""
        # Combine archive and new solutions
        combined_solutions = np.vstack([self.archive, new_solutions])
        combined_scores = np.concatenate([self.archive_scores, new_scores])
        
        # Sort by score
        sorted_indices = np.argsort(combined_scores)
        
        # Keep best solutions
        self.archive = combined_solutions[sorted_indices[:self.archive_size]]
        self.archive_scores = combined_scores[sorted_indices[:self.archive_size]]
        
    def optimize(self, objective_function, verbose=False):
        """
        Run ACO optimization
        
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
        
        # Evaluate initial archive
        for i in range(self.archive_size):
            self.archive_scores[i] = objective_function(self.archive[i])
        
        # Sort archive
        sorted_indices = np.argsort(self.archive_scores)
        self.archive = self.archive[sorted_indices]
        self.archive_scores = self.archive_scores[sorted_indices]
        
        self.best_solution = self.archive[0].copy()
        self.best_score = self.archive_scores[0]
        
        for iteration in range(self.max_iter):
            # Construct new solutions
            new_solutions = np.zeros((self.n_ants, self.dim))
            new_scores = np.zeros(self.n_ants)
            
            for ant in range(self.n_ants):
                new_solutions[ant] = self._construct_solution()
                new_scores[ant] = objective_function(new_solutions[ant])
            
            # Update archive
            self._update_archive(new_solutions, new_scores)
            
            # Update best
            if self.archive_scores[0] < self.best_score:
                self.best_score = self.archive_scores[0]
                self.best_solution = self.archive[0].copy()
            
            # Record history
            self.best_scores_history.append(self.best_score)
            self.mean_scores_history.append(np.mean(new_scores))
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}: "
                      f"Best = {self.best_score:.6f}, "
                      f"Mean = {np.mean(new_scores):.6f}")
        
        return self.best_solution, self.best_score
    
    def get_history(self):
        """Get convergence history"""
        return {
            'best_scores': np.array(self.best_scores_history),
            'mean_scores': np.array(self.mean_scores_history)
        }


if __name__ == "__main__":
    # Test ACO
    from src.test_functions import get_test_function
    
    func = get_test_function('sphere', dim=10)
    aco = ACO(n_ants=30, dim=10, max_iter=100, bounds=func.bounds)
    
    best_sol, best_score = aco.optimize(func, verbose=True)
    
    print("\n" + "=" * 50)
    print(f"Best solution: {best_sol}")
    print(f"Best score: {best_score:.6f}")
    print(f"Global optimum: {func.global_optimum}")


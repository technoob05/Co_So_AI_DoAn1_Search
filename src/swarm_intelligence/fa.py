"""
Firefly Algorithm (FA)
Inspired by the flashing behavior of fireflies

Reference:
Yang, X. S. (2008). Firefly algorithm, stochastic test functions and design optimisation.
"""

import numpy as np


class FireflyAlgorithm:
    """
    Firefly Algorithm
    
    Based on three idealized rules:
    1. All fireflies are unisex (attracted to each other)
    2. Attractiveness is proportional to brightness (inversely proportional to distance)
    3. Brightness is determined by the objective function
    
    Parameters:
    -----------
    n_fireflies : int
        Number of fireflies
    dim : int
        Dimension of the search space
    max_iter : int
        Maximum number of iterations
    alpha : float
        Randomization parameter (step size)
    beta0 : float
        Attractiveness at r=0
    gamma : float
        Light absorption coefficient
    bounds : numpy.ndarray
        Search space bounds, shape (dim, 2)
    """
    
    def __init__(self, n_fireflies=30, dim=10, max_iter=100,
                 alpha=0.5, beta0=1.0, gamma=1.0, bounds=None):
        self.n_fireflies = n_fireflies
        self.dim = dim
        self.max_iter = max_iter
        self.alpha = alpha  # Randomization parameter
        self.beta0 = beta0  # Attractiveness at r=0
        self.gamma = gamma  # Light absorption coefficient
        self.bounds = bounds if bounds is not None else np.array([[-100, 100]] * dim)
        
        # History
        self.best_scores_history = []
        self.mean_scores_history = []
        
    def initialize(self):
        """Initialize firefly positions"""
        self.fireflies = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.n_fireflies, self.dim)
        )
        
        self.light_intensity = np.zeros(self.n_fireflies)
        
        # Best firefly
        self.best_firefly = None
        self.best_score = np.inf
        
    def _distance(self, firefly_i, firefly_j):
        """Calculate Euclidean distance between two fireflies"""
        return np.sqrt(np.sum((firefly_i - firefly_j) ** 2))
    
    def _attractiveness(self, distance):
        """Calculate attractiveness as a function of distance"""
        return self.beta0 * np.exp(-self.gamma * distance ** 2)
    
    def _move_firefly(self, firefly_i, firefly_j, beta):
        """Move firefly i towards firefly j"""
        # Random walk component
        epsilon = np.random.uniform(-0.5, 0.5, self.dim)
        
        # Move towards brighter firefly
        new_position = firefly_i + beta * (firefly_j - firefly_i) + \
                      self.alpha * epsilon
        
        # Apply bounds
        new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
        
        return new_position
    
    def optimize(self, objective_function, verbose=False):
        """
        Run Firefly Algorithm optimization
        
        Parameters:
        -----------
        objective_function : callable
            Function to minimize
        verbose : bool
            Print progress
            
        Returns:
        --------
        best_firefly : numpy.ndarray
            Best solution found
        best_score : float
            Best score found
        """
        self.initialize()
        
        # Evaluate initial fireflies
        self.light_intensity = np.array([objective_function(ff) for ff in self.fireflies])
        
        # Initialize best
        best_idx = np.argmin(self.light_intensity)
        self.best_score = self.light_intensity[best_idx]
        self.best_firefly = self.fireflies[best_idx].copy()
        
        for iteration in range(self.max_iter):
            # Decrease alpha over time (adaptive randomization)
            alpha_current = self.alpha * (0.95 ** iteration)
            
            # For each firefly
            for i in range(self.n_fireflies):
                # Compare with all other fireflies
                for j in range(self.n_fireflies):
                    # If firefly j is brighter (better fitness)
                    if self.light_intensity[j] < self.light_intensity[i]:
                        # Calculate distance
                        r = self._distance(self.fireflies[i], self.fireflies[j])
                        
                        # Calculate attractiveness
                        beta = self._attractiveness(r)
                        
                        # Move firefly i towards j
                        self.fireflies[i] = self._move_firefly(
                            self.fireflies[i],
                            self.fireflies[j],
                            beta
                        )
                        
                        # Evaluate new position
                        self.light_intensity[i] = objective_function(self.fireflies[i])
            
            # Update best
            best_idx = np.argmin(self.light_intensity)
            if self.light_intensity[best_idx] < self.best_score:
                self.best_score = self.light_intensity[best_idx]
                self.best_firefly = self.fireflies[best_idx].copy()
            
            # Record history
            self.best_scores_history.append(self.best_score)
            self.mean_scores_history.append(np.mean(self.light_intensity))
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}: "
                      f"Best = {self.best_score:.6f}, "
                      f"Mean = {np.mean(self.light_intensity):.6f}")
        
        return self.best_firefly, self.best_score
    
    def get_history(self):
        """Get convergence history"""
        return {
            'best_scores': np.array(self.best_scores_history),
            'mean_scores': np.array(self.mean_scores_history)
        }


if __name__ == "__main__":
    # Test FA
    from src.test_functions import get_test_function
    
    func = get_test_function('sphere', dim=10)
    fa = FireflyAlgorithm(n_fireflies=30, dim=10, max_iter=100, bounds=func.bounds)
    
    best_sol, best_score = fa.optimize(func, verbose=True)
    
    print("\n" + "=" * 50)
    print(f"Best solution: {best_sol}")
    print(f"Best score: {best_score:.6f}")
    print(f"Global optimum: {func.global_optimum}")


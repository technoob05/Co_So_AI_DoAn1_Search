"""
Hill Climbing Algorithm
A local search algorithm that continuously moves in the direction of increasing value
(or decreasing cost for minimization)
"""

import numpy as np


class HillClimbing:
    """
    Hill Climbing Algorithm (Steepest Ascent for minimization)
    
    Parameters:
    -----------
    dim : int
        Dimension of the search space
    max_iter : int
        Maximum number of iterations
    step_size : float
        Initial step size for neighbor generation
    bounds : numpy.ndarray
        Search space bounds, shape (dim, 2)
    """
    
    def __init__(self, dim=10, max_iter=100, step_size=0.1, bounds=None):
        self.dim = dim
        self.max_iter = max_iter
        self.step_size = step_size
        self.bounds = bounds if bounds is not None else np.array([[-100, 100]] * dim)
        
        # History
        self.best_scores_history = []
        
    def initialize(self):
        """Initialize random starting point"""
        self.current_solution = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            self.dim
        )
        
        self.best_solution = None
        self.best_score = np.inf
        
    def _generate_neighbors(self, solution):
        """
        Generate neighboring solutions
        Try moving in each dimension
        """
        neighbors = []
        
        # Try both directions in each dimension
        for i in range(self.dim):
            for direction in [-1, 1]:
                neighbor = solution.copy()
                neighbor[i] += direction * self.step_size
                
                # Apply bounds
                neighbor = np.clip(neighbor, self.bounds[:, 0], self.bounds[:, 1])
                
                neighbors.append(neighbor)
        
        return neighbors
    
    def optimize(self, objective_function, verbose=False):
        """
        Run Hill Climbing optimization
        
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
        
        current_score = objective_function(self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_score = current_score
        
        no_improvement_count = 0
        
        for iteration in range(self.max_iter):
            # Generate neighbors
            neighbors = self._generate_neighbors(self.current_solution)
            
            # Evaluate all neighbors
            neighbor_scores = [objective_function(n) for n in neighbors]
            
            # Find best neighbor
            best_neighbor_idx = np.argmin(neighbor_scores)
            best_neighbor_score = neighbor_scores[best_neighbor_idx]
            
            # If best neighbor is better than current, move to it
            if best_neighbor_score < current_score:
                self.current_solution = neighbors[best_neighbor_idx]
                current_score = best_neighbor_score
                no_improvement_count = 0
                
                # Update global best
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_solution = self.current_solution.copy()
            else:
                # No improvement found, reduce step size
                no_improvement_count += 1
                if no_improvement_count > 5:
                    self.step_size *= 0.9
                    no_improvement_count = 0
            
            # Record history
            self.best_scores_history.append(self.best_score)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}: "
                      f"Best = {self.best_score:.6f}, "
                      f"Step size = {self.step_size:.6f}")
            
            # Early stopping if step size too small
            if self.step_size < 1e-10:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
        
        return self.best_solution, self.best_score
    
    def get_history(self):
        """Get convergence history"""
        return {
            'best_scores': np.array(self.best_scores_history)
        }


if __name__ == "__main__":
    # Test Hill Climbing
    from src.test_functions import get_test_function
    
    func = get_test_function('sphere', dim=10)
    hc = HillClimbing(dim=10, max_iter=100, step_size=1.0, bounds=func.bounds)
    
    best_sol, best_score = hc.optimize(func, verbose=True)
    
    print("\n" + "=" * 50)
    print(f"Best solution: {best_sol}")
    print(f"Best score: {best_score:.6f}")
    print(f"Global optimum: {func.global_optimum}")


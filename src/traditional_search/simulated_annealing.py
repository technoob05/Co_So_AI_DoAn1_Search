"""
Simulated Annealing Algorithm
Inspired by the annealing process in metallurgy

Reference:
Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). 
Optimization by simulated annealing. Science, 220(4598), 671-680.
"""

import numpy as np


class SimulatedAnnealing:
    """
    Simulated Annealing Algorithm
    
    Parameters:
    -----------
    dim : int
        Dimension of the search space
    max_iter : int
        Maximum number of iterations
    initial_temp : float
        Initial temperature
    final_temp : float
        Final temperature
    alpha : float
        Cooling rate (0 < alpha < 1)
    step_size : float
        Step size for neighbor generation
    bounds : numpy.ndarray
        Search space bounds, shape (dim, 2)
    """
    
    def __init__(self, dim=10, max_iter=1000, initial_temp=100.0, 
                 final_temp=1e-3, alpha=0.95, step_size=1.0, bounds=None):
        self.dim = dim
        self.max_iter = max_iter
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha  # Cooling rate
        self.step_size = step_size
        self.bounds = bounds if bounds is not None else np.array([[-100, 100]] * dim)
        
        # History
        self.best_scores_history = []
        self.current_scores_history = []
        self.temperature_history = []
        
    def initialize(self):
        """Initialize random starting point"""
        self.current_solution = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            self.dim
        )
        
        self.best_solution = None
        self.best_score = np.inf
        self.temperature = self.initial_temp
        
    def _generate_neighbor(self, solution):
        """Generate a random neighbor"""
        # Gaussian perturbation
        neighbor = solution + np.random.randn(self.dim) * self.step_size
        
        # Apply bounds
        neighbor = np.clip(neighbor, self.bounds[:, 0], self.bounds[:, 1])
        
        return neighbor
    
    def _acceptance_probability(self, current_cost, new_cost, temperature):
        """
        Calculate acceptance probability using Metropolis criterion
        Always accept better solutions, sometimes accept worse solutions
        """
        if new_cost < current_cost:
            return 1.0
        else:
            return np.exp(-(new_cost - current_cost) / temperature)
    
    def _cooling_schedule(self, iteration):
        """
        Update temperature using exponential cooling schedule
        """
        # Exponential cooling
        self.temperature = self.initial_temp * (self.alpha ** iteration)
        
        # Alternative: Linear cooling
        # self.temperature = self.initial_temp - (self.initial_temp - self.final_temp) * iteration / self.max_iter
    
    def optimize(self, objective_function, verbose=False):
        """
        Run Simulated Annealing optimization
        
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
        
        accepted_count = 0
        
        for iteration in range(self.max_iter):
            # Generate neighbor
            neighbor = self._generate_neighbor(self.current_solution)
            neighbor_score = objective_function(neighbor)
            
            # Calculate acceptance probability
            acceptance_prob = self._acceptance_probability(
                current_score, neighbor_score, self.temperature
            )
            
            # Accept or reject
            if np.random.rand() < acceptance_prob:
                self.current_solution = neighbor
                current_score = neighbor_score
                accepted_count += 1
                
                # Update best
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_solution = self.current_solution.copy()
            
            # Cool down
            self._cooling_schedule(iteration)
            
            # Record history
            self.best_scores_history.append(self.best_score)
            self.current_scores_history.append(current_score)
            self.temperature_history.append(self.temperature)
            
            if verbose and (iteration + 1) % 100 == 0:
                acceptance_rate = accepted_count / (iteration + 1) * 100
                print(f"Iteration {iteration + 1}/{self.max_iter}: "
                      f"Best = {self.best_score:.6f}, "
                      f"Current = {current_score:.6f}, "
                      f"Temp = {self.temperature:.6f}, "
                      f"Accept rate = {acceptance_rate:.1f}%")
            
            # Early stopping if temperature too low
            if self.temperature < self.final_temp:
                if verbose:
                    print(f"Temperature reached final temp at iteration {iteration + 1}")
                break
        
        return self.best_solution, self.best_score
    
    def get_history(self):
        """Get convergence history"""
        return {
            'best_scores': np.array(self.best_scores_history),
            'current_scores': np.array(self.current_scores_history),
            'temperatures': np.array(self.temperature_history)
        }


if __name__ == "__main__":
    # Test Simulated Annealing
    from src.test_functions import get_test_function
    
    func = get_test_function('rastrigin', dim=10)
    sa = SimulatedAnnealing(
        dim=10, 
        max_iter=1000, 
        initial_temp=100.0,
        alpha=0.95,
        bounds=func.bounds
    )
    
    best_sol, best_score = sa.optimize(func, verbose=True)
    
    print("\n" + "=" * 50)
    print(f"Best solution: {best_sol}")
    print(f"Best score: {best_score:.6f}")
    print(f"Global optimum: {func.global_optimum}")


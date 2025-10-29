"""
Particle Swarm Optimization (PSO)
Inspired by bird flocking and fish schooling behaviors

Reference:
Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
"""

import numpy as np


class PSO:
    """
    Particle Swarm Optimization Algorithm
    
    Parameters:
    -----------
    n_particles : int
        Number of particles in the swarm
    dim : int
        Dimension of the search space
    max_iter : int
        Maximum number of iterations
    w : float
        Inertia weight (controls exploration vs exploitation)
    c1 : float
        Cognitive parameter (attraction to personal best)
    c2 : float
        Social parameter (attraction to global best)
    bounds : numpy.ndarray
        Search space bounds, shape (dim, 2)
    """
    
    def __init__(self, n_particles=30, dim=10, max_iter=100, 
                 w=0.7, c1=1.5, c2=1.5, bounds=None):
        self.n_particles = n_particles
        self.dim = dim
        self.max_iter = max_iter
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        self.bounds = bounds if bounds is not None else np.array([[-100, 100]] * dim)
        
        # History
        self.best_scores_history = []
        self.mean_scores_history = []
        
    def initialize(self):
        """Initialize particle positions and velocities"""
        # Random positions within bounds
        self.positions = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            (self.n_particles, self.dim)
        )
        
        # Random velocities
        velocity_range = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
        self.velocities = np.random.uniform(
            -velocity_range,
            velocity_range,
            (self.n_particles, self.dim)
        )
        
        # Personal best positions and scores
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.n_particles, np.inf)
        
        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf
        
    def optimize(self, objective_function, verbose=False):
        """
        Run PSO optimization
        
        Parameters:
        -----------
        objective_function : callable
            Function to minimize
        verbose : bool
            Print progress
            
        Returns:
        --------
        best_position : numpy.ndarray
            Best position found
        best_score : float
            Best score found
        """
        self.initialize()
        
        for iteration in range(self.max_iter):
            # Evaluate all particles
            scores = np.array([objective_function(pos) for pos in self.positions])
            
            # Update personal bests
            improved = scores < self.personal_best_scores
            self.personal_best_scores[improved] = scores[improved]
            self.personal_best_positions[improved] = self.positions[improved]
            
            # Update global best
            best_idx = np.argmin(scores)
            if scores[best_idx] < self.global_best_score:
                self.global_best_score = scores[best_idx]
                self.global_best_position = self.positions[best_idx].copy()
            
            # Update velocities and positions
            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)
            
            cognitive = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social = self.c2 * r2 * (self.global_best_position - self.positions)
            
            self.velocities = self.w * self.velocities + cognitive + social
            
            # Update positions
            self.positions = self.positions + self.velocities
            
            # Apply bounds
            self.positions = np.clip(self.positions, self.bounds[:, 0], self.bounds[:, 1])
            
            # Record history
            self.best_scores_history.append(self.global_best_score)
            self.mean_scores_history.append(np.mean(scores))
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}: "
                      f"Best = {self.global_best_score:.6f}, "
                      f"Mean = {np.mean(scores):.6f}")
        
        return self.global_best_position, self.global_best_score
    
    def get_history(self):
        """Get convergence history"""
        return {
            'best_scores': np.array(self.best_scores_history),
            'mean_scores': np.array(self.mean_scores_history)
        }


if __name__ == "__main__":
    # Test PSO
    from src.test_functions import get_test_function
    
    # Test on Sphere function
    func = get_test_function('sphere', dim=10)
    pso = PSO(n_particles=30, dim=10, max_iter=100, bounds=func.bounds)
    
    best_pos, best_score = pso.optimize(func, verbose=True)
    
    print("\n" + "=" * 50)
    print(f"Best position: {best_pos}")
    print(f"Best score: {best_score:.6f}")
    print(f"Global optimum: {func.global_optimum}")


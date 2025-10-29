"""
Genetic Algorithm (GA)
Inspired by the process of natural selection

Reference:
Holland, J. H. (1992). Genetic algorithms. Scientific American, 267(1), 66-73.
"""

import numpy as np


class GeneticAlgorithm:
    """
    Genetic Algorithm for continuous optimization
    
    Parameters:
    -----------
    population_size : int
        Number of individuals in the population
    dim : int
        Dimension of the search space
    max_iter : int
        Maximum number of generations
    crossover_rate : float
        Probability of crossover (0 to 1)
    mutation_rate : float
        Probability of mutation (0 to 1)
    elite_size : int
        Number of elite individuals to preserve
    bounds : numpy.ndarray
        Search space bounds, shape (dim, 2)
    """
    
    def __init__(self, population_size=50, dim=10, max_iter=100,
                 crossover_rate=0.8, mutation_rate=0.1, elite_size=2, bounds=None):
        self.population_size = population_size
        self.dim = dim
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.bounds = bounds if bounds is not None else np.array([[-100, 100]] * dim)
        
        # History
        self.best_scores_history = []
        self.mean_scores_history = []
        self.diversity_history = []
        
    def initialize(self):
        """Initialize random population"""
        self.population = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.population_size, self.dim)
        )
        
        self.fitness = np.zeros(self.population_size)
        
        self.best_individual = None
        self.best_score = np.inf
        
    def _calculate_diversity(self):
        """Calculate population diversity (standard deviation)"""
        return np.mean(np.std(self.population, axis=0))
    
    def _tournament_selection(self, tournament_size=3):
        """
        Tournament selection: select best individual from random subset
        """
        # Randomly select tournament_size individuals
        tournament_indices = np.random.choice(
            self.population_size, 
            tournament_size, 
            replace=False
        )
        
        # Select best individual from tournament
        tournament_fitness = self.fitness[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        
        return self.population[winner_idx]
    
    def _crossover(self, parent1, parent2):
        """
        Simulated Binary Crossover (SBX)
        """
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Blend crossover
        alpha = np.random.rand(self.dim)
        
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        
        # Apply bounds
        child1 = np.clip(child1, self.bounds[:, 0], self.bounds[:, 1])
        child2 = np.clip(child2, self.bounds[:, 0], self.bounds[:, 1])
        
        return child1, child2
    
    def _mutate(self, individual):
        """
        Gaussian mutation
        """
        if np.random.rand() > self.mutation_rate:
            return individual
        
        # Mutation strength
        mutation_strength = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
        
        # Apply Gaussian noise to some genes
        mutation_mask = np.random.rand(self.dim) < self.mutation_rate
        individual[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * \
                                      mutation_strength[mutation_mask]
        
        # Apply bounds
        individual = np.clip(individual, self.bounds[:, 0], self.bounds[:, 1])
        
        return individual
    
    def optimize(self, objective_function, verbose=False):
        """
        Run Genetic Algorithm optimization
        
        Parameters:
        -----------
        objective_function : callable
            Function to minimize
        verbose : bool
            Print progress
            
        Returns:
        --------
        best_individual : numpy.ndarray
            Best solution found
        best_score : float
            Best score found
        """
        self.initialize()
        
        # Evaluate initial population
        self.fitness = np.array([objective_function(ind) for ind in self.population])
        
        # Initialize best
        best_idx = np.argmin(self.fitness)
        self.best_score = self.fitness[best_idx]
        self.best_individual = self.population[best_idx].copy()
        
        for generation in range(self.max_iter):
            # Sort population by fitness
            sorted_indices = np.argsort(self.fitness)
            self.population = self.population[sorted_indices]
            self.fitness = self.fitness[sorted_indices]
            
            # Elitism: preserve best individuals
            new_population = [self.population[i].copy() for i in range(self.elite_size)]
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Replace population
            self.population = np.array(new_population)
            
            # Evaluate new population
            self.fitness = np.array([objective_function(ind) for ind in self.population])
            
            # Update best
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_score:
                self.best_score = self.fitness[best_idx]
                self.best_individual = self.population[best_idx].copy()
            
            # Calculate diversity
            diversity = self._calculate_diversity()
            
            # Record history
            self.best_scores_history.append(self.best_score)
            self.mean_scores_history.append(np.mean(self.fitness))
            self.diversity_history.append(diversity)
            
            if verbose and (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}/{self.max_iter}: "
                      f"Best = {self.best_score:.6f}, "
                      f"Mean = {np.mean(self.fitness):.6f}, "
                      f"Diversity = {diversity:.6f}")
        
        return self.best_individual, self.best_score
    
    def get_history(self):
        """Get convergence history"""
        return {
            'best_scores': np.array(self.best_scores_history),
            'mean_scores': np.array(self.mean_scores_history),
            'diversity': np.array(self.diversity_history)
        }


if __name__ == "__main__":
    # Test Genetic Algorithm
    from src.test_functions import get_test_function
    
    func = get_test_function('rastrigin', dim=10)
    ga = GeneticAlgorithm(
        population_size=50,
        dim=10,
        max_iter=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        bounds=func.bounds
    )
    
    best_sol, best_score = ga.optimize(func, verbose=True)
    
    print("\n" + "=" * 50)
    print(f"Best solution: {best_sol}")
    print(f"Best score: {best_score:.6f}")
    print(f"Global optimum: {func.global_optimum}")


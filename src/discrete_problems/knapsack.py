"""
0/1 Knapsack Problem
Given items with weights and values, maximize value while staying within capacity
"""

import numpy as np


class Knapsack:
    """
    0/1 Knapsack Problem
    
    Parameters:
    -----------
    n_items : int
        Number of items
    capacity : int or float
        Maximum capacity of knapsack
    weights : numpy.ndarray, optional
        Array of item weights, shape (n_items,)
        If None, random weights will be generated
    values : numpy.ndarray, optional
        Array of item values, shape (n_items,)
        If None, random values will be generated
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_items=20, capacity=None, weights=None, values=None, seed=None):
        self.n_items = n_items
        
        if seed is not None:
            np.random.seed(seed)
        
        if weights is not None and values is not None:
            self.weights = np.array(weights)
            self.values = np.array(values)
            self.n_items = len(weights)
        else:
            # Generate random weights and values
            self.weights = np.random.randint(1, 50, n_items)
            self.values = np.random.randint(10, 100, n_items)
        
        # Set capacity to ~50% of total weight if not specified
        if capacity is None:
            self.capacity = int(np.sum(self.weights) * 0.5)
        else:
            self.capacity = capacity
    
    def evaluate(self, solution):
        """
        Evaluate a knapsack solution
        
        Parameters:
        -----------
        solution : numpy.ndarray or list
            Binary array where 1 means item is selected, 0 means not selected
            Shape: (n_items,)
        
        Returns:
        --------
        value : float
            Total value (negative if constraint violated, for minimization)
        """
        solution = np.array(solution, dtype=int)
        
        total_weight = np.sum(solution * self.weights)
        total_value = np.sum(solution * self.values)
        
        # Penalty for exceeding capacity
        if total_weight > self.capacity:
            # Return negative value with penalty
            penalty = (total_weight - self.capacity) * 1000
            return -(total_value - penalty)
        
        # Return negative value for minimization (we want to maximize)
        return -total_value
    
    def is_valid(self, solution):
        """Check if solution satisfies capacity constraint"""
        solution = np.array(solution, dtype=int)
        total_weight = np.sum(solution * self.weights)
        return total_weight <= self.capacity
    
    def get_weight(self, solution):
        """Get total weight of solution"""
        solution = np.array(solution, dtype=int)
        return np.sum(solution * self.weights)
    
    def get_value(self, solution):
        """Get total value of solution"""
        solution = np.array(solution, dtype=int)
        return np.sum(solution * self.values)
    
    def __call__(self, solution):
        """Allow knapsack to be called as a function"""
        return self.evaluate(solution)
    
    def summary(self):
        """Print problem summary"""
        print(f"Knapsack Problem Summary:")
        print(f"  Number of items: {self.n_items}")
        print(f"  Capacity: {self.capacity}")
        print(f"  Total weight of all items: {np.sum(self.weights)}")
        print(f"  Total value of all items: {np.sum(self.values)}")
        print(f"  Average weight: {np.mean(self.weights):.2f}")
        print(f"  Average value: {np.mean(self.values):.2f}")


class KnapsackSolver:
    """Collection of algorithms to solve Knapsack Problem"""
    
    @staticmethod
    def greedy_value(knapsack):
        """
        Greedy algorithm: select items by value (highest first)
        
        Returns:
        --------
        solution : numpy.ndarray
            Binary array of selected items
        value : float
            Total value achieved
        """
        n = knapsack.n_items
        solution = np.zeros(n, dtype=int)
        
        # Sort by value (descending)
        indices = np.argsort(-knapsack.values)
        
        current_weight = 0
        for idx in indices:
            if current_weight + knapsack.weights[idx] <= knapsack.capacity:
                solution[idx] = 1
                current_weight += knapsack.weights[idx]
        
        return solution, knapsack.get_value(solution)
    
    @staticmethod
    def greedy_ratio(knapsack):
        """
        Greedy algorithm: select items by value/weight ratio (highest first)
        
        Returns:
        --------
        solution : numpy.ndarray
            Binary array of selected items
        value : float
            Total value achieved
        """
        n = knapsack.n_items
        solution = np.zeros(n, dtype=int)
        
        # Calculate value/weight ratio
        ratios = knapsack.values / knapsack.weights
        
        # Sort by ratio (descending)
        indices = np.argsort(-ratios)
        
        current_weight = 0
        for idx in indices:
            if current_weight + knapsack.weights[idx] <= knapsack.capacity:
                solution[idx] = 1
                current_weight += knapsack.weights[idx]
        
        return solution, knapsack.get_value(solution)
    
    @staticmethod
    def dynamic_programming(knapsack):
        """
        Dynamic Programming solution (optimal for small problems)
        
        Returns:
        --------
        solution : numpy.ndarray
            Binary array of selected items
        value : float
            Total value achieved
        """
        n = knapsack.n_items
        capacity = int(knapsack.capacity)
        weights = knapsack.weights.astype(int)
        values = knapsack.values.astype(int)
        
        # DP table
        dp = np.zeros((n + 1, capacity + 1), dtype=int)
        
        # Fill DP table
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                # Don't take item i-1
                dp[i][w] = dp[i-1][w]
                
                # Take item i-1 if possible
                if weights[i-1] <= w:
                    dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
        
        # Backtrack to find solution
        solution = np.zeros(n, dtype=int)
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                solution[i-1] = 1
                w -= weights[i-1]
        
        return solution, int(dp[n][capacity])
    
    @staticmethod
    def genetic_algorithm(knapsack, population_size=100, max_iter=200, 
                         mutation_rate=0.1, crossover_rate=0.8, verbose=False):
        """
        Genetic Algorithm for Knapsack
        
        Parameters:
        -----------
        knapsack : Knapsack
            Knapsack problem instance
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
        best_solution : numpy.ndarray
            Best solution found
        best_value : float
            Value of best solution
        history : dict
            Optimization history
        """
        n = knapsack.n_items
        
        # Initialize population randomly
        population = np.random.randint(0, 2, (population_size, n))
        
        # Repair function to fix invalid solutions
        def repair(solution):
            sol = solution.copy()
            weight = knapsack.get_weight(sol)
            
            # If overweight, remove items randomly
            while weight > knapsack.capacity:
                selected = np.where(sol == 1)[0]
                if len(selected) == 0:
                    break
                remove_idx = np.random.choice(selected)
                sol[remove_idx] = 0
                weight = knapsack.get_weight(sol)
            
            return sol
        
        # Repair initial population
        for i in range(population_size):
            population[i] = repair(population[i])
        
        best_history = []
        mean_history = []
        
        best_solution = None
        best_value = -np.inf
        
        for iteration in range(max_iter):
            # Evaluate population
            fitness = np.array([knapsack.get_value(ind) for ind in population])
            
            # Track best
            current_best_idx = np.argmax(fitness)
            current_best_value = fitness[current_best_idx]
            
            if current_best_value > best_value:
                best_value = current_best_value
                best_solution = population[current_best_idx].copy()
            
            best_history.append(best_value)
            mean_history.append(np.mean(fitness))
            
            if verbose and iteration % 20 == 0:
                print(f"Iter {iteration}: Best={best_value:.2f}, Mean={np.mean(fitness):.2f}")
            
            # Selection (tournament)
            selected = []
            for _ in range(population_size):
                i1, i2 = np.random.choice(population_size, 2, replace=False)
                winner = i1 if fitness[i1] > fitness[i2] else i2
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
                    # Flip random bit
                    bit = np.random.randint(n)
                    offspring[i][bit] = 1 - offspring[i][bit]
            
            # Repair offspring
            population = np.array([repair(ind) for ind in offspring])
            
            # Elitism: keep best solution
            population[0] = best_solution.copy()
        
        history = {
            'best_values': best_history,
            'mean_values': mean_history
        }
        
        return best_solution, best_value, history
    
    @staticmethod
    def hill_climbing(knapsack, max_iter=1000, verbose=False):
        """
        Hill Climbing for Knapsack
        
        Returns:
        --------
        best_solution : numpy.ndarray
            Best solution found
        best_value : float
            Value of best solution
        history : dict
            Optimization history
        """
        n = knapsack.n_items
        
        # Start with greedy solution
        current_solution, _ = KnapsackSolver.greedy_ratio(knapsack)
        current_value = knapsack.get_value(current_solution)
        
        best_solution = current_solution.copy()
        best_value = current_value
        
        history = []
        
        for iteration in range(max_iter):
            improved = False
            
            # Try flipping each bit
            for i in range(n):
                neighbor = current_solution.copy()
                neighbor[i] = 1 - neighbor[i]
                
                if knapsack.is_valid(neighbor):
                    neighbor_value = knapsack.get_value(neighbor)
                    
                    if neighbor_value > current_value:
                        current_solution = neighbor.copy()
                        current_value = neighbor_value
                        improved = True
                        
                        if current_value > best_value:
                            best_solution = current_solution.copy()
                            best_value = current_value
                        
                        break
            
            history.append(best_value)
            
            if not improved:
                break
            
            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration}: Best={best_value:.2f}")
        
        return best_solution, best_value, {'best_values': history}
    
    @staticmethod
    def simulated_annealing(knapsack, max_iter=1000, initial_temp=100, 
                           cooling_rate=0.95, verbose=False):
        """
        Simulated Annealing for Knapsack
        
        Returns:
        --------
        best_solution : numpy.ndarray
            Best solution found
        best_value : float
            Value of best solution
        history : dict
            Optimization history
        """
        n = knapsack.n_items
        
        # Start with greedy solution
        current_solution, _ = KnapsackSolver.greedy_ratio(knapsack)
        current_value = knapsack.get_value(current_solution)
        
        best_solution = current_solution.copy()
        best_value = current_value
        
        temperature = initial_temp
        history = []
        
        for iteration in range(max_iter):
            # Generate neighbor by flipping random bit
            neighbor = current_solution.copy()
            flip_idx = np.random.randint(n)
            neighbor[flip_idx] = 1 - neighbor[flip_idx]
            
            if knapsack.is_valid(neighbor):
                neighbor_value = knapsack.get_value(neighbor)
                delta = neighbor_value - current_value
                
                # Accept if better or with probability
                if delta > 0 or np.random.rand() < np.exp(delta / temperature):
                    current_solution = neighbor.copy()
                    current_value = neighbor_value
                    
                    if current_value > best_value:
                        best_solution = current_solution.copy()
                        best_value = current_value
            
            temperature *= cooling_rate
            history.append(best_value)
            
            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration}: Best={best_value:.2f}, Temp={temperature:.2f}")
        
        return best_solution, best_value, {'best_values': history}


"""
Test Functions for Optimization Algorithms
Includes continuous optimization benchmark functions
"""

import numpy as np


class ContinuousTestFunction:
    """Base class for continuous test functions"""
    
    def __init__(self, dim):
        self.dim = dim
        self.bounds = None
        self.global_optimum = None
    
    def __call__(self, x):
        raise NotImplementedError
    
    def get_bounds(self):
        return self.bounds


class Sphere(ContinuousTestFunction):
    """
    Sphere Function
    Global minimum: f(0, 0, ..., 0) = 0
    Search domain: [-100, 100]^d
    
    f(x) = sum(x_i^2)
    """
    
    def __init__(self, dim=10):
        super().__init__(dim)
        self.bounds = np.array([[-100, 100]] * dim)
        self.global_optimum = 0
    
    def __call__(self, x):
        return np.sum(x ** 2)


class Rastrigin(ContinuousTestFunction):
    """
    Rastrigin Function
    Global minimum: f(0, 0, ..., 0) = 0
    Search domain: [-5.12, 5.12]^d
    
    f(x) = 10d + sum(x_i^2 - 10*cos(2*pi*x_i))
    
    Highly multimodal function with many local minima
    """
    
    def __init__(self, dim=10):
        super().__init__(dim)
        self.bounds = np.array([[-5.12, 5.12]] * dim)
        self.global_optimum = 0
    
    def __call__(self, x):
        return 10 * self.dim + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


class Rosenbrock(ContinuousTestFunction):
    """
    Rosenbrock Function (Banana Function)
    Global minimum: f(1, 1, ..., 1) = 0
    Search domain: [-5, 10]^d
    
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    
    The global minimum lies in a narrow, parabolic valley
    """
    
    def __init__(self, dim=10):
        super().__init__(dim)
        self.bounds = np.array([[-5, 10]] * dim)
        self.global_optimum = 0
    
    def __call__(self, x):
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


class Ackley(ContinuousTestFunction):
    """
    Ackley Function
    Global minimum: f(0, 0, ..., 0) = 0
    Search domain: [-32.768, 32.768]^d
    
    f(x) = -20*exp(-0.2*sqrt(sum(x_i^2)/d)) - exp(sum(cos(2*pi*x_i))/d) + 20 + e
    
    Highly multimodal with many local minima
    """
    
    def __init__(self, dim=10):
        super().__init__(dim)
        self.bounds = np.array([[-32.768, 32.768]] * dim)
        self.global_optimum = 0
    
    def __call__(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        
        term1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / self.dim))
        term2 = -np.exp(np.sum(np.cos(c * x)) / self.dim)
        
        return term1 + term2 + a + np.e


# Factory function
def get_test_function(name, dim=10):
    """
    Get test function by name
    
    Parameters:
    -----------
    name : str
        Name of the function ('sphere', 'rastrigin', 'rosenbrock', 'ackley')
    dim : int
        Dimension of the problem
    
    Returns:
    --------
    function : ContinuousTestFunction
        Test function object
    """
    functions = {
        'sphere': Sphere,
        'rastrigin': Rastrigin,
        'rosenbrock': Rosenbrock,
        'ackley': Ackley
    }
    
    if name.lower() not in functions:
        raise ValueError(f"Unknown function: {name}. Available: {list(functions.keys())}")
    
    return functions[name.lower()](dim)


if __name__ == "__main__":
    # Test all functions
    dim = 10
    x_test = np.random.randn(dim)
    
    functions = ['sphere', 'rastrigin', 'rosenbrock', 'ackley']
    
    print("Testing continuous functions:")
    print("=" * 50)
    for func_name in functions:
        func = get_test_function(func_name, dim)
        value = func(x_test)
        print(f"{func_name.capitalize()}: f(x) = {value:.6f}")
        print(f"  Bounds: {func.bounds[0]}")
        print(f"  Global optimum: {func.global_optimum}")
        print()


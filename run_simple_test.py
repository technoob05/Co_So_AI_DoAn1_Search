"""
Simple test script without visualization
Use this if you have import errors with seaborn/scipy
"""

import numpy as np
import sys
sys.path.insert(0, '.')

print("="*60)
print("SIMPLE TEST - Swarm Intelligence Algorithms")
print("="*60)

# Test imports
print("\n1. Testing imports...")
try:
    from src.test_functions import get_test_function
    print("   ✓ test_functions imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    from src.swarm_intelligence.pso import PSO
    from src.swarm_intelligence.abc import ABC
    print("   ✓ swarm_intelligence imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    from src.traditional_search.hill_climbing import HillClimbing
    print("   ✓ traditional_search imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test algorithms
print("\n2. Testing algorithms on Sphere function...")
np.random.seed(42)

func = get_test_function('sphere', dim=10)
print(f"   Function: Sphere (dim=10)")
print(f"   Global optimum: {func.global_optimum}")

algorithms = {
    'PSO': PSO(n_particles=20, dim=10, max_iter=50, bounds=func.bounds),
    'ABC': ABC(n_bees=20, dim=10, max_iter=50, bounds=func.bounds),
    'Hill Climbing': HillClimbing(dim=10, max_iter=50, bounds=func.bounds)
}

results = {}

for name, algo in algorithms.items():
    print(f"\n   Running {name}...")
    best_pos, best_score = algo.optimize(func, verbose=False)
    results[name] = best_score
    print(f"   ✓ {name}: {best_score:.6f}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

sorted_results = sorted(results.items(), key=lambda x: x[1])

for rank, (name, score) in enumerate(sorted_results, 1):
    print(f"{rank}. {name:20s}: {score:.6f}")

print("\n✓ All tests passed!")
print("="*60)
print("\nIf this works, you can run: python demo.py")
print("Or use the full comparison tools in USAGE_GUIDE.md")
print("="*60)


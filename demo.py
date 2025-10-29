"""
Demo script for Swarm Intelligence Algorithms
CSC14003 - Fundamentals of Artificial Intelligence
"""

import numpy as np
import matplotlib.pyplot as plt
from src.test_functions import get_test_function
from src.swarm_intelligence import PSO, ACO, ABC, FireflyAlgorithm, CuckooSearch
from src.traditional_search import HillClimbing, SimulatedAnnealing, GeneticAlgorithm
from src.discrete_problems import TSP, TSPSolver
from src.visualization import OptimizationVisualizer
from src.comparison import AlgorithmComparison


def demo_continuous_optimization():
    """Demo continuous optimization algorithms"""
    print("\n" + "="*80)
    print("DEMO: CONTINUOUS OPTIMIZATION")
    print("="*80)
    
    # Test function
    func = get_test_function('rastrigin', dim=10)
    print(f"\nTest Function: Rastrigin (dim=10)")
    print(f"Global optimum: {func.global_optimum}")
    print(f"Search bounds: {func.bounds[0]}")
    
    # Define algorithms
    algorithms = {
        'PSO': (PSO, {
            'n_particles': 30, 
            'dim': 10, 
            'max_iter': 100, 
            'bounds': func.bounds
        }),
        'ABC': (ABC, {
            'n_bees': 30, 
            'dim': 10, 
            'max_iter': 100, 
            'bounds': func.bounds
        }),
        'FA': (FireflyAlgorithm, {
            'n_fireflies': 30, 
            'dim': 10, 
            'max_iter': 100, 
            'bounds': func.bounds
        }),
        'CS': (CuckooSearch, {
            'n_nests': 30, 
            'dim': 10, 
            'max_iter': 100, 
            'bounds': func.bounds
        }),
        'GA': (GeneticAlgorithm, {
            'population_size': 50, 
            'dim': 10, 
            'max_iter': 100, 
            'bounds': func.bounds
        }),
        'SA': (SimulatedAnnealing, {
            'dim': 10, 
            'max_iter': 1000, 
            'bounds': func.bounds
        })
    }
    
    # Run comparison
    print("\nRunning algorithms (3 trials each)...")
    results = AlgorithmComparison.compare_algorithms(
        algorithms, func, n_trials=3, verbose=False
    )
    
    # Generate report
    report = AlgorithmComparison.generate_report(
        results,
        objective_name="Rastrigin Function (dim=10)",
        target_score=0.0
    )
    print("\n" + report)
    
    # Plot convergence
    histories = []
    labels = []
    for name, algo_results in results.items():
        if algo_results[0]['history']:
            histories.append(algo_results[0]['history'])
            labels.append(name)
    
    OptimizationVisualizer.plot_convergence(
        histories, labels,
        title="Convergence Comparison - Rastrigin Function",
        log_scale=True
    )
    
    return results


def demo_tsp():
    """Demo TSP solving"""
    print("\n" + "="*80)
    print("DEMO: TRAVELING SALESMAN PROBLEM (TSP)")
    print("="*80)
    
    # Create TSP instance
    n_cities = 20
    tsp = TSP(n_cities=n_cities, seed=42)
    print(f"\nTSP with {n_cities} cities")
    
    # Solve with different methods
    print("\n1. Nearest Neighbor Heuristic:")
    nn_tour, nn_distance = TSPSolver.nearest_neighbor(tsp)
    print(f"   Distance: {nn_distance:.2f}")
    
    print("\n2. 2-opt Local Search:")
    opt_tour, opt_distance = TSPSolver.two_opt(tsp)
    print(f"   Distance: {opt_distance:.2f}")
    print(f"   Improvement: {((nn_distance - opt_distance) / nn_distance * 100):.1f}%")
    
    print("\n3. Genetic Algorithm for TSP:")
    ga_tour, ga_distance, ga_history = TSPSolver.genetic_algorithm_tsp(
        tsp, population_size=100, max_iter=200
    )
    print(f"   Distance: {ga_distance:.2f}")
    print(f"   Improvement over NN: {((nn_distance - ga_distance) / nn_distance * 100):.1f}%")
    
    # Visualize best tour
    OptimizationVisualizer.plot_tsp_tour(tsp, ga_tour, title="Best TSP Tour (GA)")
    
    # Plot GA convergence
    plt.figure(figsize=(10, 6))
    plt.plot(ga_history['best_distances'], label='Best', linewidth=2)
    plt.plot(ga_history['mean_distances'], label='Mean', linewidth=2, alpha=0.7)
    plt.xlabel('Generation')
    plt.ylabel('Tour Distance')
    plt.title('GA Convergence for TSP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def demo_visualization_2d():
    """Demo 2D function visualization"""
    print("\n" + "="*80)
    print("DEMO: 2D FUNCTION VISUALIZATION")
    print("="*80)
    
    functions = ['sphere', 'rastrigin', 'rosenbrock', 'ackley']
    
    for func_name in functions:
        print(f"\nVisualizing {func_name.upper()} function...")
        func = get_test_function(func_name, dim=2)
        
        OptimizationVisualizer.plot_3d_surface(
            func,
            x_range=(-5, 5),
            y_range=(-5, 5),
            n_points=100
        )


def main():
    """Main demo function"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "SWARM INTELLIGENCE ALGORITHMS" + " "*28 + "║")
    print("║" + " "*15 + "CSC14003 - Fundamentals of AI Project" + " "*26 + "║")
    print("╚" + "="*78 + "╝")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demos
    print("\nAvailable demos:")
    print("1. Continuous Optimization Comparison")
    print("2. Traveling Salesman Problem (TSP)")
    print("3. 2D Function Visualization")
    print("4. Run all demos")
    
    choice = input("\nSelect demo (1-4) or press Enter for all: ").strip()
    
    if choice == '1':
        demo_continuous_optimization()
    elif choice == '2':
        demo_tsp()
    elif choice == '3':
        demo_visualization_2d()
    else:
        # Run all
        demo_visualization_2d()
        results = demo_continuous_optimization()
        demo_tsp()
    
    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


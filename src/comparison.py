"""
Comparison utilities for optimization algorithms
"""

import numpy as np
import time
import pandas as pd
from tqdm import tqdm


class AlgorithmComparison:
    """Tools for comparing optimization algorithms"""
    
    @staticmethod
    def run_single_trial(algorithm, objective_function, verbose=False):
        """
        Run a single trial of an algorithm
        
        Parameters:
        -----------
        algorithm : object
            Algorithm instance with optimize() method
        objective_function : callable
            Objective function to optimize
        verbose : bool
            Print progress
            
        Returns:
        --------
        result : dict
            Trial results including score, time, and history
        """
        start_time = time.time()
        
        best_solution, best_score = algorithm.optimize(objective_function, verbose=verbose)
        
        elapsed_time = time.time() - start_time
        
        history = algorithm.get_history() if hasattr(algorithm, 'get_history') else {}
        
        return {
            'best_solution': best_solution,
            'best_score': best_score,
            'time': elapsed_time,
            'history': history
        }
    
    @staticmethod
    def run_multiple_trials(algorithm_class, algorithm_params, 
                          objective_function, n_trials=10, verbose=False):
        """
        Run multiple trials of an algorithm
        
        Parameters:
        -----------
        algorithm_class : class
            Algorithm class
        algorithm_params : dict
            Parameters for algorithm initialization
        objective_function : callable
            Objective function to optimize
        n_trials : int
            Number of trials
        verbose : bool
            Print progress
            
        Returns:
        --------
        results : list of dict
            Results from all trials
        """
        results = []
        
        iterator = tqdm(range(n_trials), desc="Running trials") if not verbose else range(n_trials)
        
        for trial in iterator:
            # Create new algorithm instance for each trial
            algorithm = algorithm_class(**algorithm_params)
            
            result = AlgorithmComparison.run_single_trial(
                algorithm, objective_function, verbose=verbose
            )
            result['trial'] = trial
            results.append(result)
        
        return results
    
    @staticmethod
    def compare_algorithms(algorithms_dict, objective_function, n_trials=10, 
                          verbose=False):
        """
        Compare multiple algorithms
        
        Parameters:
        -----------
        algorithms_dict : dict
            {name: (algorithm_class, params)}
        objective_function : callable
            Objective function to optimize
        n_trials : int
            Number of trials per algorithm
        verbose : bool
            Print progress
            
        Returns:
        --------
        comparison_results : dict
            {algorithm_name: [trial_results]}
        """
        comparison_results = {}
        
        for name, (algo_class, params) in algorithms_dict.items():
            print(f"\nRunning {name}...")
            results = AlgorithmComparison.run_multiple_trials(
                algo_class, params, objective_function, n_trials, verbose
            )
            comparison_results[name] = results
        
        return comparison_results
    
    @staticmethod
    def calculate_statistics(results):
        """
        Calculate statistics from trial results
        
        Parameters:
        -----------
        results : list of dict
            Results from multiple trials
            
        Returns:
        --------
        stats : dict
            Statistical summary
        """
        scores = [r['best_score'] for r in results]
        times = [r['time'] for r in results]
        
        stats = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'median_score': np.median(scores),
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'total_time': np.sum(times)
        }
        
        return stats
    
    @staticmethod
    def create_comparison_table(comparison_results):
        """
        Create comparison table from results
        
        Parameters:
        -----------
        comparison_results : dict
            {algorithm_name: [trial_results]}
            
        Returns:
        --------
        df : pandas.DataFrame
            Comparison table
        """
        data = []
        
        for name, results in comparison_results.items():
            stats = AlgorithmComparison.calculate_statistics(results)
            
            row = {
                'Algorithm': name,
                'Mean Score': stats['mean_score'],
                'Std Score': stats['std_score'],
                'Min Score': stats['min_score'],
                'Max Score': stats['max_score'],
                'Median Score': stats['median_score'],
                'Mean Time (s)': stats['mean_time'],
                'Std Time (s)': stats['std_time']
            }
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('Mean Score')
        
        return df
    
    @staticmethod
    def convergence_speed_metric(history, threshold=0.01):
        """
        Calculate convergence speed (iterations to reach threshold of optimum)
        
        Parameters:
        -----------
        history : dict
            History with 'best_scores' key
        threshold : float
            Threshold percentage of final best score
            
        Returns:
        --------
        iterations : int
            Number of iterations to converge
        """
        if 'best_scores' not in history:
            return None
        
        best_scores = history['best_scores']
        final_score = best_scores[-1]
        target = final_score * (1 + threshold)
        
        # Find first iteration where score is within threshold
        for i, score in enumerate(best_scores):
            if score <= target:
                return i
        
        return len(best_scores)
    
    @staticmethod
    def robustness_metric(results):
        """
        Calculate robustness (coefficient of variation of scores)
        Lower is more robust
        
        Parameters:
        -----------
        results : list of dict
            Results from multiple trials
            
        Returns:
        --------
        cv : float
            Coefficient of variation
        """
        scores = [r['best_score'] for r in results]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if mean_score == 0:
            return np.inf
        
        cv = std_score / abs(mean_score)
        return cv
    
    @staticmethod
    def success_rate(results, target_score, tolerance=1e-2):
        """
        Calculate success rate (% of trials reaching target score)
        
        Parameters:
        -----------
        results : list of dict
            Results from multiple trials
        target_score : float
            Target score to reach
        tolerance : float
            Tolerance around target score
            
        Returns:
        --------
        success_rate : float
            Percentage of successful trials
        """
        scores = [r['best_score'] for r in results]
        successes = sum(1 for s in scores if abs(s - target_score) <= tolerance)
        
        return successes / len(results) * 100
    
    @staticmethod
    def generate_report(comparison_results, objective_name="Unknown", 
                       target_score=None, save_path=None):
        """
        Generate comprehensive comparison report
        
        Parameters:
        -----------
        comparison_results : dict
            {algorithm_name: [trial_results]}
        objective_name : str
            Name of objective function
        target_score : float, optional
            Known global optimum
        save_path : str, optional
            Path to save report
            
        Returns:
        --------
        report : str
            Formatted report
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"ALGORITHM COMPARISON REPORT - {objective_name}")
        lines.append("=" * 80)
        lines.append("")
        
        # Comparison table
        df = AlgorithmComparison.create_comparison_table(comparison_results)
        lines.append("PERFORMANCE SUMMARY")
        lines.append("-" * 80)
        lines.append(df.to_string(index=False))
        lines.append("")
        
        # Detailed statistics per algorithm
        lines.append("DETAILED STATISTICS")
        lines.append("-" * 80)
        
        for name, results in comparison_results.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Number of trials: {len(results)}")
            
            stats = AlgorithmComparison.calculate_statistics(results)
            lines.append(f"  Mean score: {stats['mean_score']:.6f} ± {stats['std_score']:.6f}")
            lines.append(f"  Best score: {stats['min_score']:.6f}")
            lines.append(f"  Worst score: {stats['max_score']:.6f}")
            lines.append(f"  Median score: {stats['median_score']:.6f}")
            lines.append(f"  Mean time: {stats['mean_time']:.4f} ± {stats['std_time']:.4f} seconds")
            
            # Robustness
            cv = AlgorithmComparison.robustness_metric(results)
            lines.append(f"  Robustness (CV): {cv:.6f}")
            
            # Success rate (if target score provided)
            if target_score is not None:
                sr = AlgorithmComparison.success_rate(results, target_score)
                lines.append(f"  Success rate: {sr:.1f}%")
            
            # Convergence speed (if history available)
            if results[0]['history']:
                conv_speeds = []
                for r in results:
                    speed = AlgorithmComparison.convergence_speed_metric(r['history'])
                    if speed is not None:
                        conv_speeds.append(speed)
                
                if conv_speeds:
                    lines.append(f"  Mean convergence speed: {np.mean(conv_speeds):.1f} iterations")
        
        lines.append("")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


if __name__ == "__main__":
    # Test comparison functions
    print("Testing comparison utilities...")
    
    from src.test_functions import get_test_function
    from src.swarm_intelligence.pso import PSO
    from src.traditional_search.hill_climbing import HillClimbing
    
    # Setup
    func = get_test_function('sphere', dim=5)
    
    algorithms = {
        'PSO': (PSO, {'n_particles': 20, 'dim': 5, 'max_iter': 50, 'bounds': func.bounds}),
        'Hill Climbing': (HillClimbing, {'dim': 5, 'max_iter': 50, 'bounds': func.bounds})
    }
    
    # Run comparison
    print("\nRunning comparison (2 trials each)...")
    results = AlgorithmComparison.compare_algorithms(
        algorithms, func, n_trials=2, verbose=False
    )
    
    # Generate report
    report = AlgorithmComparison.generate_report(
        results, 
        objective_name="Sphere Function (dim=5)",
        target_score=0.0
    )
    
    print("\n" + report)


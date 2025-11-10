"""
Module tính toán Performance Metrics cho các thuật toán tối ưu

Các metrics được tính:
- Convergence speed
- Success rate
- Solution quality
- Runtime
- Memory usage
"""

import numpy as np
import time
from typing import Dict, List, Any, Callable
import psutil
import os


class PerformanceMetrics:
    """
    Class để tính toán và lưu trữ performance metrics
    """
    
    @staticmethod
    def convergence_speed(history: Dict[str, np.ndarray], threshold: float = 0.01) -> int:
        """
        Tính số iterations cần để đạt được threshold
        
        Parameters:
        -----------
        history : dict
            Dictionary chứa 'best_scores' - array các best scores qua iterations
        threshold : float
            Ngưỡng để coi là đã hội tụ (khoảng cách từ global optimum)
            
        Returns:
        --------
        iterations : int
            Số iterations cần để đạt threshold (hoặc -1 nếu không đạt)
        """
        best_scores = history.get('best_scores', [])
        if len(best_scores) == 0:
            return -1
        
        # Giả sử global optimum là 0 (đối với test functions chuẩn)
        # Hoặc có thể truyền vào như parameter
        for i, score in enumerate(best_scores):
            if abs(score) <= threshold:
                return i + 1
        
        return -1  # Không đạt threshold
    
    @staticmethod
    def success_rate(results: List[float], threshold: float = 0.01, 
                    global_optimum: float = 0.0) -> float:
        """
        Tính tỷ lệ thành công (% runs đạt được global optimum)
        
        Parameters:
        -----------
        results : list
            Danh sách các best scores từ nhiều runs
        threshold : float
            Khoảng cách chấp nhận được từ global optimum
        global_optimum : float
            Giá trị global optimum
            
        Returns:
        --------
        success_rate : float
            Tỷ lệ thành công (0-1)
        """
        if len(results) == 0:
            return 0.0
        
        successes = sum(1 for score in results if abs(score - global_optimum) <= threshold)
        return successes / len(results)
    
    @staticmethod
    def solution_quality(best_score: float, global_optimum: float = 0.0) -> Dict[str, float]:
        """
        Đánh giá chất lượng solution
        
        Parameters:
        -----------
        best_score : float
            Best score tìm được
        global_optimum : float
            Global optimum thực sự
            
        Returns:
        --------
        quality_metrics : dict
            Dictionary chứa các metrics về chất lượng
        """
        absolute_error = abs(best_score - global_optimum)
        
        # Tính relative error (tránh chia cho 0)
        if abs(global_optimum) > 1e-10:
            relative_error = absolute_error / abs(global_optimum)
        else:
            relative_error = absolute_error
        
        # Accuracy (1 = perfect, 0 = worst)
        # Sử dụng negative exponential để scale
        accuracy = np.exp(-relative_error)
        
        return {
            'best_score': best_score,
            'global_optimum': global_optimum,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'accuracy': accuracy
        }
    
    @staticmethod
    def convergence_curve_stats(history: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Phân tích đường cong hội tụ
        
        Parameters:
        -----------
        history : dict
            Dictionary chứa 'best_scores'
            
        Returns:
        --------
        stats : dict
            Thống kê về convergence curve
        """
        best_scores = history.get('best_scores', [])
        if len(best_scores) == 0:
            return {}
        
        best_scores = np.array(best_scores)
        
        # Tính improvement rate
        improvements = np.diff(best_scores)
        
        # Tính khi nào có improvement đáng kể
        significant_threshold = 0.01  # 1% improvement
        significant_improvements = np.where(
            np.abs(improvements / (best_scores[:-1] + 1e-10)) > significant_threshold
        )[0]
        
        stats = {
            'total_iterations': len(best_scores),
            'initial_score': float(best_scores[0]),
            'final_score': float(best_scores[-1]),
            'total_improvement': float(best_scores[0] - best_scores[-1]),
            'improvement_rate': float((best_scores[0] - best_scores[-1]) / len(best_scores)),
            'n_significant_improvements': len(significant_improvements),
            'plateau_iterations': len(best_scores) - len(significant_improvements),
            'converged': bool(np.std(best_scores[-10:]) < 1e-6) if len(best_scores) >= 10 else False
        }
        
        return stats
    
    @staticmethod
    def statistical_comparison(results_dict: Dict[str, List[float]]) -> pd.DataFrame:
        """
        So sánh thống kê giữa các thuật toán
        
        Parameters:
        -----------
        results_dict : dict
            {algorithm_name: [list of best scores from multiple runs]}
            
        Returns:
        --------
        comparison_df : DataFrame
            DataFrame chứa thống kê so sánh
        """
        import pandas as pd
        
        stats = []
        
        for algo_name, scores in results_dict.items():
            if len(scores) == 0:
                continue
            
            scores_array = np.array(scores)
            
            stat = {
                'Algorithm': algo_name,
                'Mean': np.mean(scores_array),
                'Std': np.std(scores_array),
                'Min': np.min(scores_array),
                'Max': np.max(scores_array),
                'Median': np.median(scores_array),
                'Q1': np.percentile(scores_array, 25),
                'Q3': np.percentile(scores_array, 75),
                'IQR': np.percentile(scores_array, 75) - np.percentile(scores_array, 25)
            }
            
            stats.append(stat)
        
        return pd.DataFrame(stats)


class BenchmarkRunner:
    """
    Class để chạy benchmark và thu thập metrics
    """
    
    def __init__(self):
        self.results = []
    
    def run_single_experiment(self, algorithm_class, algorithm_params: Dict,
                              objective_function, n_runs: int = 10,
                              verbose: bool = False) -> Dict[str, Any]:
        """
        Chạy một thuật toán nhiều lần và thu thập metrics
        
        Parameters:
        -----------
        algorithm_class : class
            Class của thuật toán
        algorithm_params : dict
            Parameters để khởi tạo thuật toán
        objective_function : callable
            Hàm mục tiêu
        n_runs : int
            Số lần chạy
        verbose : bool
            In thông tin
            
        Returns:
        --------
        results : dict
            Kết quả tổng hợp từ nhiều runs
        """
        best_scores = []
        runtimes = []
        histories = []
        memory_usages = []
        
        for run in range(n_runs):
            if verbose:
                print(f"Run {run + 1}/{n_runs}...")
            
            # Khởi tạo thuật toán
            algo = algorithm_class(**algorithm_params)
            
            # Đo memory trước khi chạy
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Chạy thuật toán và đo thời gian
            start_time = time.time()
            best_pos, best_score = algo.optimize(objective_function, verbose=False)
            runtime = time.time() - start_time
            
            # Đo memory sau khi chạy
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = mem_after - mem_before
            
            # Lưu kết quả
            best_scores.append(best_score)
            runtimes.append(runtime)
            memory_usages.append(memory_usage)
            
            # Lấy history nếu có
            if hasattr(algo, 'get_history'):
                histories.append(algo.get_history())
        
        # Tính toán metrics
        results = {
            'algorithm': algorithm_class.__name__,
            'n_runs': n_runs,
            'best_scores': best_scores,
            'mean_score': np.mean(best_scores),
            'std_score': np.std(best_scores),
            'best_score': np.min(best_scores),
            'worst_score': np.max(best_scores),
            'median_score': np.median(best_scores),
            'mean_runtime': np.mean(runtimes),
            'std_runtime': np.std(runtimes),
            'total_runtime': np.sum(runtimes),
            'mean_memory': np.mean(memory_usages),
            'histories': histories,
            'success_rate': PerformanceMetrics.success_rate(
                best_scores, 
                threshold=0.01,
                global_optimum=getattr(objective_function, 'global_optimum', 0.0)
            )
        }
        
        return results
    
    def compare_algorithms(self, algorithms: Dict[str, tuple], 
                          objective_function, n_runs: int = 10,
                          verbose: bool = False) -> Dict[str, Dict]:
        """
        So sánh nhiều thuật toán
        
        Parameters:
        -----------
        algorithms : dict
            {name: (AlgorithmClass, params_dict)}
        objective_function : callable
            Hàm mục tiêu
        n_runs : int
            Số lần chạy mỗi thuật toán
        verbose : bool
            In thông tin
            
        Returns:
        --------
        comparison : dict
            Kết quả so sánh
        """
        comparison = {}
        
        for algo_name, (algo_class, algo_params) in algorithms.items():
            if verbose:
                print(f"\n{'='*60}")
                print(f"Running {algo_name}...")
                print(f"{'='*60}")
            
            results = self.run_single_experiment(
                algo_class, algo_params, objective_function, n_runs, verbose
            )
            
            comparison[algo_name] = results
        
        return comparison
    
    def generate_report(self, comparison: Dict[str, Dict], save_path: str = None) -> str:
        """
        Tạo báo cáo văn bản từ kết quả so sánh
        
        Parameters:
        -----------
        comparison : dict
            Kết quả từ compare_algorithms
        save_path : str, optional
            Đường dẫn để lưu báo cáo
            
        Returns:
        --------
        report : str
            Báo cáo dạng text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("OPTIMIZATION ALGORITHMS COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary table
        report_lines.append("SUMMARY TABLE")
        report_lines.append("-" * 80)
        header = f"{'Algorithm':<20} {'Best':<12} {'Mean±Std':<20} {'Success%':<10} {'Time(s)':<10}"
        report_lines.append(header)
        report_lines.append("-" * 80)
        
        for algo_name, results in comparison.items():
            best = f"{results['best_score']:.6f}"
            mean_std = f"{results['mean_score']:.6f}±{results['std_score']:.6f}"
            success = f"{results['success_rate']*100:.1f}%"
            time_val = f"{results['mean_runtime']:.3f}"
            
            line = f"{algo_name:<20} {best:<12} {mean_std:<20} {success:<10} {time_val:<10}"
            report_lines.append(line)
        
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Detailed results for each algorithm
        for algo_name, results in comparison.items():
            report_lines.append(f"\n{'='*80}")
            report_lines.append(f"DETAILED RESULTS: {algo_name}")
            report_lines.append(f"{'='*80}")
            report_lines.append(f"Number of runs: {results['n_runs']}")
            report_lines.append(f"Best score: {results['best_score']:.8f}")
            report_lines.append(f"Worst score: {results['worst_score']:.8f}")
            report_lines.append(f"Mean score: {results['mean_score']:.8f}")
            report_lines.append(f"Std score: {results['std_score']:.8f}")
            report_lines.append(f"Median score: {results['median_score']:.8f}")
            report_lines.append(f"Success rate: {results['success_rate']*100:.2f}%")
            report_lines.append(f"Mean runtime: {results['mean_runtime']:.4f} seconds")
            report_lines.append(f"Total runtime: {results['total_runtime']:.4f} seconds")
            report_lines.append(f"Mean memory usage: {results['mean_memory']:.2f} MB")
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Save if path provided
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


if __name__ == "__main__":
    # Test metrics
    from src.test_functions import get_test_function
    from src.swarm_intelligence.pso import PSO
    from src.traditional_search.genetic_algorithm import GeneticAlgorithm
    
    print("Testing Performance Metrics...")
    print("=" * 60)
    
    # Create test function
    func = get_test_function('sphere', dim=10)
    
    # Test benchmark runner
    runner = BenchmarkRunner()
    
    # Define algorithms to compare
    algorithms = {
        'PSO': (PSO, {
            'n_particles': 30,
            'dim': 10,
            'max_iter': 50,
            'bounds': func.bounds
        }),
        'GA': (GeneticAlgorithm, {
            'population_size': 30,
            'dim': 10,
            'max_iter': 50,
            'bounds': func.bounds
        })
    }
    
    # Run comparison
    print("\nRunning comparison (5 runs each)...")
    comparison = runner.compare_algorithms(algorithms, func, n_runs=5, verbose=True)
    
    # Generate report
    print("\n" + "=" * 60)
    print("GENERATING REPORT")
    print("=" * 60)
    report = runner.generate_report(comparison)
    print(report)


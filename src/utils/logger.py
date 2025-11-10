"""
Module Logging và Export Results

Chức năng:
- Log tất cả experiments
- Export results dạng CSV, JSON
- Export charts dạng PNG, PDF với high DPI
- Generate LaTeX tables cho report
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ExperimentLogger:
    """
    Class để log experiments và export results
    """
    
    def __init__(self, log_dir: str = "logs", results_dir: str = "results"):
        """
        Khởi tạo logger
        
        Parameters:
        -----------
        log_dir : str
            Thư mục lưu log files
        results_dir : str
            Thư mục lưu results
        """
        self.log_dir = log_dir
        self.results_dir = results_dir
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "reports"), exist_ok=True)
        
        self.experiments = []
    
    def log_experiment(self, algorithm: str, problem: str, parameters: Dict,
                      results: Dict, metadata: Dict = None) -> str:
        """
        Log một experiment
        
        Parameters:
        -----------
        algorithm : str
            Tên thuật toán
        problem : str
            Tên bài toán
        parameters : dict
            Parameters của thuật toán
        results : dict
            Kết quả (best_score, runtime, etc.)
        metadata : dict, optional
            Thông tin khác
            
        Returns:
        --------
        exp_id : str
            ID của experiment
        """
        timestamp = datetime.now()
        exp_id = f"{algorithm}_{problem}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        experiment = {
            'exp_id': exp_id,
            'timestamp': timestamp.isoformat(),
            'algorithm': algorithm,
            'problem': problem,
            'parameters': parameters,
            'results': self._convert_to_serializable(results),
            'metadata': metadata or {}
        }
        
        self.experiments.append(experiment)
        
        # Lưu vào file JSON
        log_file = os.path.join(self.log_dir, f"{exp_id}.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(experiment, f, indent=2, ensure_ascii=False)
        
        return exp_id
    
    def _convert_to_serializable(self, obj):
        """Chuyển đổi numpy arrays và các object khác thành serializable"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def export_csv(self, experiments: List[Dict] = None, filename: str = None) -> str:
        """
        Export experiments ra CSV
        
        Parameters:
        -----------
        experiments : list, optional
            Danh sách experiments (nếu None thì dùng self.experiments)
        filename : str, optional
            Tên file (nếu None thì tự động generate)
            
        Returns:
        --------
        filepath : str
            Đường dẫn file đã lưu
        """
        if experiments is None:
            experiments = self.experiments
        
        if len(experiments) == 0:
            raise ValueError("No experiments to export")
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"experiments_{timestamp}.csv"
        
        filepath = os.path.join(self.results_dir, "data", filename)
        
        # Chuẩn bị data
        rows = []
        for exp in experiments:
            row = {
                'exp_id': exp['exp_id'],
                'timestamp': exp['timestamp'],
                'algorithm': exp['algorithm'],
                'problem': exp['problem'],
            }
            
            # Thêm parameters (flatten)
            for k, v in exp.get('parameters', {}).items():
                row[f'param_{k}'] = v
            
            # Thêm results (flatten)
            for k, v in exp.get('results', {}).items():
                if isinstance(v, (int, float, str, bool)):
                    row[f'result_{k}'] = v
            
            rows.append(row)
        
        # Ghi file CSV
        if rows:
            keys = rows[0].keys()
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        
        return filepath
    
    def export_json(self, experiments: List[Dict] = None, filename: str = None) -> str:
        """
        Export experiments ra JSON
        
        Parameters:
        -----------
        experiments : list, optional
            Danh sách experiments
        filename : str, optional
            Tên file
            
        Returns:
        --------
        filepath : str
            Đường dẫn file đã lưu
        """
        if experiments is None:
            experiments = self.experiments
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"experiments_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, "data", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(experiments, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def export_chart(self, fig, filename: str, dpi: int = 300, 
                    formats: List[str] = ['png', 'pdf']) -> List[str]:
        """
        Export matplotlib figure với high DPI
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure để export
        filename : str
            Tên file (không có extension)
        dpi : int
            DPI (dots per inch)
        formats : list
            Danh sách format cần export
            
        Returns:
        --------
        filepaths : list
            Danh sách đường dẫn files đã lưu
        """
        filepaths = []
        
        for fmt in formats:
            filepath = os.path.join(self.results_dir, "plots", f"{filename}.{fmt}")
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format=fmt)
            filepaths.append(filepath)
        
        return filepaths
    
    def generate_latex_table(self, comparison_df, caption: str = "", 
                            label: str = "tab:comparison") -> str:
        """
        Generate LaTeX table từ pandas DataFrame
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            DataFrame chứa kết quả so sánh
        caption : str
            Caption cho table
        label : str
            Label cho reference
            
        Returns:
        --------
        latex_code : str
            LaTeX code
        """
        # Convert DataFrame to LaTeX
        latex = comparison_df.to_latex(
            index=False,
            float_format="%.6f",
            column_format='l' + 'r' * (len(comparison_df.columns) - 1),
            escape=False
        )
        
        # Thêm caption và label
        lines = latex.split('\n')
        
        # Insert after \begin{tabular}
        for i, line in enumerate(lines):
            if '\\begin{tabular}' in line:
                lines.insert(i, '\\centering')
                if caption:
                    lines.insert(i, f'\\caption{{{caption}}}')
                if label:
                    lines.insert(i, f'\\label{{{label}}}')
                break
        
        latex_code = '\n'.join(lines)
        
        # Lưu vào file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.results_dir, "reports", 
                               f"table_{timestamp}.tex")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(latex_code)
        
        return latex_code
    
    def create_comparison_report(self, comparison_results: Dict[str, Dict],
                                 problem_name: str, output_prefix: str = None) -> Dict[str, str]:
        """
        Tạo báo cáo so sánh hoàn chỉnh với charts và tables
        
        Parameters:
        -----------
        comparison_results : dict
            Kết quả từ BenchmarkRunner.compare_algorithms
        problem_name : str
            Tên bài toán
        output_prefix : str, optional
            Prefix cho tên files
            
        Returns:
        --------
        outputs : dict
            Dictionary chứa đường dẫn các files đã tạo
        """
        if output_prefix is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_prefix = f"{problem_name}_{timestamp}"
        
        outputs = {}
        
        # 1. Convergence Plot
        fig_conv, ax = plt.subplots(figsize=(12, 6))
        
        for algo_name, results in comparison_results.items():
            if 'histories' in results and len(results['histories']) > 0:
                # Plot mean convergence với std
                all_histories = []
                for hist in results['histories']:
                    if 'best_scores' in hist:
                        all_histories.append(hist['best_scores'])
                
                if all_histories:
                    # Pad to same length
                    max_len = max(len(h) for h in all_histories)
                    padded = []
                    for h in all_histories:
                        if len(h) < max_len:
                            padded_h = np.full(max_len, h[-1])
                            padded_h[:len(h)] = h
                            padded.append(padded_h)
                        else:
                            padded.append(h)
                    
                    histories_array = np.array(padded)
                    mean_history = np.mean(histories_array, axis=0)
                    std_history = np.std(histories_array, axis=0)
                    
                    iterations = range(len(mean_history))
                    ax.plot(iterations, mean_history, linewidth=2, label=algo_name)
                    ax.fill_between(iterations, 
                                   mean_history - std_history,
                                   mean_history + std_history,
                                   alpha=0.2)
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Best Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Convergence Comparison - {problem_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        conv_files = self.export_chart(fig_conv, f"{output_prefix}_convergence")
        outputs['convergence_plot'] = conv_files
        plt.close(fig_conv)
        
        # 2. Box Plot
        fig_box, ax = plt.subplots(figsize=(12, 6))
        
        data = []
        labels = []
        for algo_name, results in comparison_results.items():
            if 'best_scores' in results:
                data.append(results['best_scores'])
                labels.append(algo_name)
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
            
            # Màu sắc
            colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Best Score', fontsize=12, fontweight='bold')
            ax.set_title(f'Performance Distribution - {problem_name}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            box_files = self.export_chart(fig_box, f"{output_prefix}_boxplot")
            outputs['boxplot'] = box_files
            plt.close(fig_box)
        
        # 3. Bar Chart - Mean Scores
        fig_bar, ax = plt.subplots(figsize=(12, 6))
        
        algo_names = list(comparison_results.keys())
        mean_scores = [results['mean_score'] for results in comparison_results.values()]
        std_scores = [results['std_score'] for results in comparison_results.values()]
        
        x = np.arange(len(algo_names))
        bars = ax.bar(x, mean_scores, yerr=std_scores, capsize=5, 
                     color=plt.cm.Set3(np.linspace(0, 1, len(algo_names))),
                     edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Best Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Mean Performance - {problem_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        bar_files = self.export_chart(fig_bar, f"{output_prefix}_barplot")
        outputs['barplot'] = bar_files
        plt.close(fig_bar)
        
        # 4. Export data
        csv_file = self.export_csv(
            [self.log_experiment(algo, problem_name, {}, results) 
             for algo, results in comparison_results.items()],
            filename=f"{output_prefix}_results.csv"
        )
        outputs['csv'] = csv_file
        
        json_file = self.export_json(
            [{'algorithm': algo, 'results': results} 
             for algo, results in comparison_results.items()],
            filename=f"{output_prefix}_results.json"
        )
        outputs['json'] = json_file
        
        return outputs
    
    def load_experiments(self, exp_ids: List[str] = None) -> List[Dict]:
        """
        Load experiments từ log files
        
        Parameters:
        -----------
        exp_ids : list, optional
            Danh sách experiment IDs (nếu None thì load tất cả)
            
        Returns:
        --------
        experiments : list
            Danh sách experiments
        """
        experiments = []
        
        if exp_ids is None:
            # Load tất cả files trong log_dir
            for filename in os.listdir(self.log_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.log_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        experiments.append(json.load(f))
        else:
            # Load specific experiments
            for exp_id in exp_ids:
                filepath = os.path.join(self.log_dir, f"{exp_id}.json")
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        experiments.append(json.load(f))
        
        return experiments


if __name__ == "__main__":
    # Test logger
    print("Testing ExperimentLogger...")
    print("=" * 60)
    
    # Khởi tạo logger
    logger = ExperimentLogger(log_dir="test_logs", results_dir="test_results")
    
    # Log một experiment
    exp_id = logger.log_experiment(
        algorithm="PSO",
        problem="Sphere",
        parameters={'n_particles': 30, 'w': 0.7, 'c1': 1.5, 'c2': 1.5},
        results={'best_score': 0.00123, 'runtime': 2.5, 'iterations': 100},
        metadata={'dim': 10, 'bounds': [-100, 100]}
    )
    
    print(f"Logged experiment: {exp_id}")
    
    # Export CSV
    csv_file = logger.export_csv()
    print(f"Exported CSV: {csv_file}")
    
    # Export JSON
    json_file = logger.export_json()
    print(f"Exported JSON: {json_file}")
    
    # Create a test chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([1, 2, 3, 4], [10, 5, 2, 1], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Test Convergence')
    
    chart_files = logger.export_chart(fig, "test_convergence")
    print(f"Exported charts: {chart_files}")
    plt.close(fig)
    
    print("\nTest completed successfully!")


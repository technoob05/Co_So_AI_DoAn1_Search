"""
Visualization utilities for optimization algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    # Use matplotlib style instead
    plt.style.use('default')

plt.rcParams['figure.figsize'] = (12, 8)


class OptimizationVisualizer:
    """Visualization tools for optimization algorithms"""
    
    @staticmethod
    def plot_convergence(histories, labels, title="Convergence Comparison", 
                        save_path=None, log_scale=False):
        """
        Plot convergence curves for multiple algorithms
        
        Parameters:
        -----------
        histories : list of dict
            List of history dictionaries from algorithms
        labels : list of str
            Algorithm names
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        log_scale : bool
            Use log scale for y-axis
        """
        plt.figure(figsize=(12, 6))
        
        for history, label in zip(histories, labels):
            if 'best_scores' in history:
                iterations = range(len(history['best_scores']))
                plt.plot(iterations, history['best_scores'], 
                        label=label, linewidth=2, alpha=0.8)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Score', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if log_scale:
            plt.yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_convergence_with_std(histories, labels, title="Convergence with Std Dev",
                                  save_path=None):
        """
        Plot convergence with mean and standard deviation
        Useful when running multiple trials
        
        Parameters:
        -----------
        histories : list of list of dict
            histories[i][j] = history of algorithm i, trial j
        labels : list of str
            Algorithm names
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        plt.figure(figsize=(12, 6))
        
        for algo_histories, label in zip(histories, labels):
            # Stack all trials
            all_scores = []
            for h in algo_histories:
                if 'best_scores' in h:
                    all_scores.append(h['best_scores'])
            
            if all_scores:
                # Pad sequences to same length
                max_len = max(len(s) for s in all_scores)
                padded = []
                for s in all_scores:
                    padded_s = np.full(max_len, s[-1])
                    padded_s[:len(s)] = s
                    padded.append(padded_s)
                
                scores_array = np.array(padded)
                mean_scores = np.mean(scores_array, axis=0)
                std_scores = np.std(scores_array, axis=0)
                iterations = range(len(mean_scores))
                
                plt.plot(iterations, mean_scores, label=label, linewidth=2)
                plt.fill_between(iterations, 
                               mean_scores - std_scores,
                               mean_scores + std_scores,
                               alpha=0.2)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Score', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_3d_surface(test_function, x_range=(-5, 5), y_range=(-5, 5),
                       n_points=100, save_path=None):
        """
        Plot 3D surface of a 2D test function
        
        Parameters:
        -----------
        test_function : callable
            Function to plot (must accept 2D input)
        x_range : tuple
            Range for x axis
        y_range : tuple
            Range for y axis
        n_points : int
            Number of points per dimension
        save_path : str, optional
            Path to save figure
        """
        # Create mesh grid
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate function
        Z = np.zeros_like(X)
        for i in range(n_points):
            for j in range(n_points):
                Z[i, j] = test_function(np.array([X[i, j], Y[i, j]]))
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 6))
        
        # Surface plot
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('X', fontsize=10)
        ax1.set_ylabel('Y', fontsize=10)
        ax1.set_zlabel('f(X, Y)', fontsize=10)
        ax1.set_title('3D Surface', fontsize=12, fontweight='bold')
        fig.colorbar(surf, ax=ax1, shrink=0.5)
        
        # Contour plot
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
        ax2.contour(X, Y, Z, levels=20, colors='black', alpha=0.2, linewidths=0.5)
        ax2.set_xlabel('X', fontsize=10)
        ax2.set_ylabel('Y', fontsize=10)
        ax2.set_title('Contour Plot', fontsize=12, fontweight='bold')
        fig.colorbar(contour, ax=ax2, shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_tsp_tour(tsp, tour, title="TSP Tour", save_path=None):
        """
        Visualize TSP tour
        
        Parameters:
        -----------
        tsp : TSP
            TSP problem instance
        tour : list
            Tour as list of city indices
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        cities = tsp.get_cities()
        
        if cities is None:
            print("No city coordinates available for visualization")
            return
        
        plt.figure(figsize=(10, 10))
        
        # Plot cities
        plt.scatter(cities[:, 0], cities[:, 1], c='red', s=200, 
                   zorder=3, edgecolors='black', linewidth=2)
        
        # Add city labels
        for i, (x, y) in enumerate(cities):
            plt.annotate(str(i), (x, y), fontsize=10, ha='center', va='center',
                        color='white', fontweight='bold')
        
        # Plot tour
        for i in range(len(tour)):
            start = tour[i]
            end = tour[(i + 1) % len(tour)]
            
            plt.plot([cities[start, 0], cities[end, 0]],
                    [cities[start, 1], cities[end, 1]],
                    'b-', linewidth=2, alpha=0.6, zorder=1)
            
            # Add arrow
            dx = cities[end, 0] - cities[start, 0]
            dy = cities[end, 1] - cities[start, 1]
            plt.arrow(cities[start, 0], cities[start, 1], dx * 0.8, dy * 0.8,
                     head_width=1.5, head_length=1.5, fc='blue', ec='blue',
                     alpha=0.3, zorder=2)
        
        distance = tsp.evaluate(tour)
        plt.title(f"{title}\nTotal Distance: {distance:.2f}", 
                 fontsize=14, fontweight='bold')
        plt.xlabel('X coordinate', fontsize=12)
        plt.ylabel('Y coordinate', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_parameter_sensitivity(param_values, results, param_name,
                                   metric_name="Best Score", title=None,
                                   save_path=None):
        """
        Plot parameter sensitivity analysis
        
        Parameters:
        -----------
        param_values : list
            Parameter values tested
        results : list
            Results for each parameter value (can be list of lists for multiple trials)
        param_name : str
            Name of parameter
        metric_name : str
            Name of metric being measured
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure
        """
        plt.figure(figsize=(10, 6))
        
        # Check if results is 2D (multiple trials)
        if isinstance(results[0], (list, np.ndarray)):
            means = [np.mean(r) for r in results]
            stds = [np.std(r) for r in results]
            
            plt.errorbar(param_values, means, yerr=stds, 
                        marker='o', markersize=8, linewidth=2,
                        capsize=5, capthick=2)
        else:
            plt.plot(param_values, results, marker='o', markersize=8, linewidth=2)
        
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        
        if title is None:
            title = f"Parameter Sensitivity: {param_name}"
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_box_comparison(results_dict, title="Algorithm Comparison",
                           ylabel="Best Score", save_path=None):
        """
        Box plot comparison of algorithms
        
        Parameters:
        -----------
        results_dict : dict
            {algorithm_name: [results from multiple runs]}
        title : str
            Plot title
        ylabel : str
            Y-axis label
        save_path : str, optional
            Path to save figure
        """
        plt.figure(figsize=(12, 6))
        
        data = list(results_dict.values())
        labels = list(results_dict.keys())
        
        bp = plt.boxplot(data, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)
        
        # Customize colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization functions...")
    
    # Test convergence plot
    np.random.seed(42)
    history1 = {'best_scores': np.exp(-np.linspace(0, 5, 100)) + np.random.rand(100) * 0.1}
    history2 = {'best_scores': np.exp(-np.linspace(0, 4, 100)) + np.random.rand(100) * 0.15}
    
    OptimizationVisualizer.plot_convergence(
        [history1, history2],
        ['Algorithm 1', 'Algorithm 2'],
        title="Test Convergence"
    )


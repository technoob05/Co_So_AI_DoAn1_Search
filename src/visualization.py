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
    
    @staticmethod
    def plot_knapsack_solution(knapsack, solution, title="Knapsack Solution",
                               save_path=None):
        """
        Visualize knapsack solution
        
        Parameters:
        -----------
        knapsack : Knapsack
            Knapsack problem instance
        solution : numpy.ndarray
            Binary solution array
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Items selected
        selected = np.where(solution == 1)[0]
        not_selected = np.where(solution == 0)[0]
        
        axes[0].scatter(knapsack.weights[not_selected], knapsack.values[not_selected],
                       c='lightgray', s=100, alpha=0.5, label='Not selected')
        axes[0].scatter(knapsack.weights[selected], knapsack.values[selected],
                       c='green', s=150, alpha=0.7, label='Selected')
        
        axes[0].set_xlabel('Weight', fontsize=12)
        axes[0].set_ylabel('Value', fontsize=12)
        axes[0].set_title('Items: Weight vs Value', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Right: Summary statistics
        total_weight = knapsack.get_weight(solution)
        total_value = knapsack.get_value(solution)
        n_selected = np.sum(solution)
        
        summary_text = (
            f"Selected Items: {n_selected}/{knapsack.n_items}\n"
            f"Total Weight: {total_weight}/{knapsack.capacity}\n"
            f"Total Value: {total_value}\n"
            f"Capacity Used: {total_weight/knapsack.capacity*100:.1f}%\n"
            f"Valid: {'Yes' if knapsack.is_valid(solution) else 'No'}"
        )
        
        axes[1].text(0.1, 0.5, summary_text, fontsize=14, 
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].axis('off')
        
        # Bar chart for capacity usage
        ax_bar = axes[1].inset_axes([0.1, 0.1, 0.8, 0.25])
        ax_bar.barh(['Capacity'], [total_weight], color='green', alpha=0.7)
        ax_bar.barh(['Capacity'], [knapsack.capacity - total_weight], 
                   left=[total_weight], color='lightgray', alpha=0.5)
        ax_bar.set_xlim(0, knapsack.capacity)
        ax_bar.set_xlabel('Weight')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_graph_coloring(graph, coloring, title="Graph Coloring Solution",
                           save_path=None):
        """
        Visualize graph coloring solution
        
        Parameters:
        -----------
        graph : GraphColoring
            Graph coloring problem instance
        coloring : numpy.ndarray
            Color assignment for each vertex
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Graph visualization (circular layout)
        n = graph.n_vertices
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        pos = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Draw edges
        for i, j in graph.edges:
            x = [pos[i, 0], pos[j, 0]]
            y = [pos[i, 1], pos[j, 1]]
            # Red if conflict, gray otherwise
            color = 'red' if coloring[i] == coloring[j] else 'gray'
            alpha = 0.8 if coloring[i] == coloring[j] else 0.3
            axes[0].plot(x, y, color=color, alpha=alpha, linewidth=1.5)
        
        # Draw vertices
        colors = plt.cm.Set3(np.linspace(0, 1, max(coloring) + 1))
        vertex_colors = [colors[c] for c in coloring]
        
        axes[0].scatter(pos[:, 0], pos[:, 1], c=vertex_colors, 
                       s=300, edgecolors='black', linewidth=2, zorder=10)
        
        # Add vertex labels
        for i in range(n):
            axes[0].text(pos[i, 0], pos[i, 1], str(i), 
                        ha='center', va='center', fontsize=8, fontweight='bold')
        
        axes[0].set_xlim(-1.2, 1.2)
        axes[0].set_ylim(-1.2, 1.2)
        axes[0].set_aspect('equal')
        axes[0].axis('off')
        axes[0].set_title('Graph Coloring', fontsize=13, fontweight='bold')
        
        # Right: Summary statistics
        n_conflicts = graph.count_conflicts(coloring)
        n_colors = graph.count_colors(coloring)
        is_valid = graph.is_valid(coloring)
        
        summary_text = (
            f"Vertices: {graph.n_vertices}\n"
            f"Edges: {len(graph.edges)}\n"
            f"Colors Used: {n_colors}\n"
            f"Conflicts: {n_conflicts}\n"
            f"Valid: {'Yes' if is_valid else 'No'}"
        )
        
        axes[1].text(0.1, 0.5, summary_text, fontsize=14,
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[1].axis('off')
        
        # Color distribution
        color_counts = np.bincount(coloring)
        ax_bar = axes[1].inset_axes([0.1, 0.1, 0.8, 0.3])
        ax_bar.bar(range(len(color_counts)), color_counts, 
                  color=[colors[i] for i in range(len(color_counts))],
                  edgecolor='black')
        ax_bar.set_xlabel('Color')
        ax_bar.set_ylabel('Count')
        ax_bar.set_title('Color Distribution', fontsize=10)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_grid_path(grid_world, path, title="Path Finding Solution",
                      save_path=None):
        """
        Visualize path finding solution on grid
        
        Parameters:
        -----------
        grid_world : GridWorld
            Grid world instance
        path : list of tuples
            Path from start to goal
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create grid visualization
        grid_display = np.zeros((grid_world.rows, grid_world.cols, 3))
        
        # Color obstacles (dark gray)
        grid_display[grid_world.grid == 1] = [0.3, 0.3, 0.3]
        
        # Color free spaces (white)
        grid_display[grid_world.grid == 0] = [1, 1, 1]
        
        # Color path (light blue)
        if path:
            for r, c in path:
                grid_display[r, c] = [0.5, 0.8, 1.0]
        
        # Color start (green)
        grid_display[grid_world.start] = [0, 1, 0]
        
        # Color goal (red)
        grid_display[grid_world.goal] = [1, 0, 0]
        
        ax.imshow(grid_display, interpolation='nearest')
        
        # Draw grid lines
        for i in range(grid_world.rows + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        for j in range(grid_world.cols + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        
        # Draw path with arrows
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                r1, c1 = path[i]
                r2, c2 = path[i + 1]
                ax.arrow(c1, r1, c2 - c1, r2 - r1,
                        head_width=0.3, head_length=0.2,
                        fc='blue', ec='blue', alpha=0.5, length_includes_head=True)
        
        # Add labels
        ax.text(grid_world.start[1], grid_world.start[0], 'S',
               ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        ax.text(grid_world.goal[1], grid_world.goal[0], 'G',
               ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[0, 1, 0], label='Start'),
            Patch(facecolor=[1, 0, 0], label='Goal'),
            Patch(facecolor=[0.5, 0.8, 1.0], label='Path'),
            Patch(facecolor=[0.3, 0.3, 0.3], label='Obstacle'),
            Patch(facecolor=[1, 1, 1], edgecolor='gray', label='Free')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
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


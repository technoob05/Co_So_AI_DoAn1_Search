"""
Visualization utilities cho Discrete Optimization Problems
Bao gồm TSP, Knapsack, Graph Coloring với animation chi tiết
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import cm
import seaborn as sns

# Try to import networkx for graph visualization
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class TSPVisualizer:
    """Visualization cho Traveling Salesman Problem"""
    
    @staticmethod
    def create_tsp_animation_frame(tsp, current_tour, best_tour, iteration, 
                                   algorithm_name, population_tours=None,
                                   pheromone_matrix=None, current_city=None,
                                   tour_history=None):
        """
        Tạo frame animation cho TSP với multi-panel visualization
        
        Parameters:
        -----------
        tsp : TSP
            TSP problem instance
        current_tour : list
            Tour hiện tại
        best_tour : list
            Tour tốt nhất tìm được
        iteration : int
            Iteration hiện tại
        algorithm_name : str
            Tên thuật toán
        population_tours : list of list, optional
            Danh sách các tours trong population (cho swarm algorithms)
        pheromone_matrix : numpy.ndarray, optional
            Ma trận pheromone (cho ACO)
        current_city : int, optional
            City đang được xử lý (highlight bằng vòng đỏ)
        tour_history : list of float, optional
            Lịch sử tour lengths qua các iterations
        """
        # Tạo figure với multiple subplots
        if pheromone_matrix is not None:
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            ax_main = fig.add_subplot(gs[:, :2])  # Main tour visualization
            ax_pheromone = fig.add_subplot(gs[0, 2])  # Pheromone heatmap
            ax_stats = fig.add_subplot(gs[1, 2])  # Statistics
        else:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            ax_main = fig.add_subplot(gs[:, 0])  # Main tour visualization
            ax_stats = fig.add_subplot(gs[0, 1])  # Statistics
            ax_pop = fig.add_subplot(gs[1, 1])  # Population info
        
        # ========== MAIN PANEL: Tour Visualization ==========
        cities = tsp.cities
        
        # Vẽ population tours với transparency (nếu có)
        if population_tours is not None and len(population_tours) > 0:
            for tour in population_tours[:10]:  # Giới hạn 10 tours để tránh clutter
                tour_extended = list(tour) + [tour[0]]
                tour_coords = cities[tour_extended]
                ax_main.plot(tour_coords[:, 0], tour_coords[:, 1],
                           'gray', alpha=0.2, linewidth=1, linestyle='--')
        
        # Vẽ best tour (đường đỏ đậm)
        if best_tour is not None and len(best_tour) > 0:
            best_tour_extended = list(best_tour) + [best_tour[0]]
            best_tour_coords = cities[best_tour_extended]
            
            # Vẽ đường với gradient color
            for i in range(len(best_tour)):
                start = cities[best_tour[i]]
                end = cities[best_tour[(i + 1) % len(best_tour)]]
                
                # Màu gradient từ xanh lá sang đỏ
                color = plt.cm.RdYlGn_r(i / len(best_tour))
                ax_main.plot([start[0], end[0]], [start[1], end[1]],
                           color=color, linewidth=3, alpha=0.8, zorder=2)
                
                # Vẽ arrow để chỉ hướng
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                ax_main.arrow(start[0] + dx * 0.3, start[1] + dy * 0.3,
                            dx * 0.2, dy * 0.2,
                            head_width=2, head_length=1.5, fc=color, ec=color,
                            alpha=0.6, zorder=3)
        
        # Vẽ cities (nodes)
        ax_main.scatter(cities[:, 0], cities[:, 1], 
                       c='blue', s=200, marker='o', 
                       edgecolors='darkblue', linewidths=2,
                       alpha=0.8, zorder=5, label='Cities')
        
        # Vẽ city numbers
        for i, (x, y) in enumerate(cities):
            ax_main.text(x, y, str(i), fontsize=9, ha='center', va='center',
                       color='white', weight='bold', zorder=6)
        
        # Highlight current city với HIỆU ỨNG CỰC NỔI BẬT (đang xử lý)
        if current_city is not None and 0 <= current_city < len(cities):
            cx, cy = cities[current_city]
            
            # Layer 1: Outer glow (red circle lớn nhất)
            ax_main.scatter(cx, cy, c='red', s=1200, marker='o',
                           alpha=0.3, zorder=7, edgecolors='none')
            
            # Layer 2: Middle ring (orange)
            ax_main.scatter(cx, cy, c='orange', s=800, marker='o',
                           alpha=0.5, zorder=8, edgecolors='red', linewidths=4)
            
            # Layer 3: Inner circle (yellow - main highlight)
            ax_main.scatter(cx, cy, c='yellow', s=500, marker='o',
                           edgecolors='red', linewidths=6,
                           alpha=1.0, zorder=9, label=f'⚡ Processing: City {current_city}')
            
            # Add LARGE arrow pointing to city
            arrow_start_x = cx + 15
            arrow_start_y = cy + 15
            ax_main.annotate('', xy=(cx, cy), xytext=(arrow_start_x, arrow_start_y),
                           arrowprops=dict(arrowstyle='->', lw=4, color='red',
                                         mutation_scale=30, alpha=0.9),
                           zorder=10)
            
            # Add LARGE text label
            ax_main.text(arrow_start_x + 2, arrow_start_y + 2,
                        f'◄ PROCESSING\nCITY {current_city}',
                        fontsize=14, fontweight='bold',
                        color='red', zorder=10,
                        bbox=dict(boxstyle='round,pad=0.8', 
                                facecolor='yellow', 
                                edgecolor='red', 
                                linewidth=3,
                                alpha=0.95))
        
        # Highlight start city (city 0)
        ax_main.scatter(cities[0, 0], cities[0, 1],
                       c='lime', s=400, marker='*',
                       edgecolors='darkgreen', linewidths=3,
                       alpha=0.9, zorder=7, label='Start')
        
        # Calculate tour length
        best_length = tsp.evaluate(best_tour) if best_tour is not None else float('inf')
        
        ax_main.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
        ax_main.set_title(f'{algorithm_name} - TSP Solution\nIteration {iteration} | Best Length: {best_length:.2f}',
                         fontsize=14, fontweight='bold')
        ax_main.legend(loc='upper right', fontsize=10)
        ax_main.grid(True, alpha=0.3)
        ax_main.set_aspect('equal', adjustable='box')
        
        # ========== PHEROMONE HEATMAP (cho ACO) ==========
        if pheromone_matrix is not None:
            im = ax_pheromone.imshow(pheromone_matrix, cmap='YlOrRd', 
                                    aspect='auto', interpolation='nearest')
            ax_pheromone.set_xlabel('City', fontsize=10, fontweight='bold')
            ax_pheromone.set_ylabel('City', fontsize=10, fontweight='bold')
            ax_pheromone.set_title('Pheromone Intensity', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax_pheromone, fraction=0.046, pad=0.04)
            
            # Highlight best tour edges
            if best_tour is not None:
                for i in range(len(best_tour)):
                    city1 = best_tour[i]
                    city2 = best_tour[(i + 1) % len(best_tour)]
                    ax_pheromone.plot([city2, city2], [city1, city1], 
                                    'b*', markersize=8, alpha=0.7)
        
        # ========== STATISTICS PANEL ==========
        if pheromone_matrix is not None:
            ax_to_use = ax_stats
        else:
            ax_to_use = ax_stats
            
        ax_to_use.axis('off')
        
        # Create statistics text
        stats_text = f"""
        📊 STATISTICS
        ─────────────────────────
        Algorithm: {algorithm_name}
        Iteration: {iteration}
        
        🎯 Best Tour Length: {best_length:.2f}
        
        🏙️  Cities: {tsp.n_cities}
        """
        
        if population_tours is not None:
            pop_lengths = [tsp.evaluate(tour) for tour in population_tours]
            avg_length = np.mean(pop_lengths)
            std_length = np.std(pop_lengths)
            stats_text += f"""
        👥 Population Size: {len(population_tours)}
        📈 Avg Tour Length: {avg_length:.2f}
        📊 Std Deviation: {std_length:.2f}
        """
        
        ax_to_use.text(0.1, 0.95, stats_text, transform=ax_to_use.transAxes,
                      fontsize=11, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                      family='monospace')
        
        # ========== TOUR QUALITY VISUALIZATION (đẹp hơn histogram) ==========
        if pheromone_matrix is None:
            if tour_history is not None and len(tour_history) > 1:
                # TOUR IMPROVEMENT TIMELINE (trực quan hơn)
                iterations = range(len(tour_history))
                
                # Best tour line
                ax_pop.plot(iterations, tour_history, 'b-', linewidth=3, 
                           label='Best Tour Length', alpha=0.8)
                ax_pop.fill_between(iterations, tour_history, 
                                   alpha=0.3, color='blue')
                
                # Mark improvements
                improvements = []
                for i in range(1, len(tour_history)):
                    if tour_history[i] < tour_history[i-1]:
                        improvements.append(i)
                
                if improvements:
                    ax_pop.scatter([improvements], [tour_history[i] for i in improvements],
                                  c='red', s=100, marker='*', zorder=5,
                                  label='Improvements', edgecolors='darkred', linewidths=2)
                
                # Current position
                ax_pop.scatter([len(tour_history)-1], [tour_history[-1]],
                              c='gold', s=200, marker='D', zorder=6,
                              label='Current', edgecolors='darkgoldenrod', linewidths=2)
                
                ax_pop.set_xlabel('Iteration', fontsize=10, fontweight='bold')
                ax_pop.set_ylabel('Tour Length', fontsize=10, fontweight='bold')
                ax_pop.set_title('🚀 Tour Improvement Progress', fontsize=12, fontweight='bold')
                ax_pop.legend(fontsize=9, loc='upper right')
                ax_pop.grid(True, alpha=0.3)
                
                # Annotate best
                ax_pop.text(0.02, 0.98, f'Best: {best_length:.2f}',
                           transform=ax_pop.transAxes, fontsize=11, fontweight='bold',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                
                # Annotate improvement count
                ax_pop.text(0.02, 0.88, f'Improvements: {len(improvements)}',
                           transform=ax_pop.transAxes, fontsize=10,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                
            elif population_tours is not None and len(population_tours) > 0:
                # POPULATION QUALITY BARS (cho GA) - đẹp hơn histogram
                pop_lengths = sorted([tsp.evaluate(tour) for tour in population_tours])
                n_show = min(20, len(pop_lengths))
                indices = np.linspace(0, len(pop_lengths)-1, n_show, dtype=int)
                
                # Color gradient from best (green) to worst (red)
                colors = plt.cm.RdYlGn_r(np.linspace(0, 1, n_show))
                
                bars = ax_pop.bar(range(n_show), [pop_lengths[i] for i in indices],
                                 color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
                
                # Highlight best
                bars[0].set_edgecolor('darkgreen')
                bars[0].set_linewidth(3)
                
                ax_pop.axhline(best_length, color='red', linestyle='--',
                              linewidth=2, alpha=0.7, label=f'Best: {best_length:.2f}')
                ax_pop.axhline(np.mean(pop_lengths), color='blue', linestyle=':',
                              linewidth=2, alpha=0.7, label=f'Avg: {np.mean(pop_lengths):.2f}')
                
                ax_pop.set_xlabel('Solution Rank (Best → Worst)', fontsize=10, fontweight='bold')
                ax_pop.set_ylabel('Tour Length', fontsize=10, fontweight='bold')
                ax_pop.set_title('📊 Population Quality Spectrum', fontsize=12, fontweight='bold')
                ax_pop.legend(fontsize=9, loc='upper left')
                ax_pop.grid(True, alpha=0.3, axis='y')
                
                # Stats box
                stats_text = f'Range: {pop_lengths[0]:.1f} - {pop_lengths[-1]:.1f}\nStd: {np.std(pop_lengths):.2f}'
                ax_pop.text(0.98, 0.98, stats_text,
                           transform=ax_pop.transAxes, fontsize=9,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            else:
                # No data - show message
                ax_pop.axis('off')
                ax_pop.text(0.5, 0.5, '📈 Tour Progress\n(Data will appear during optimization)',
                           ha='center', va='center', fontsize=12, fontweight='bold',
                           transform=ax_pop.transAxes,
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_convergence_comparison(histories, labels, title="TSP Convergence"):
        """
        So sánh convergence của nhiều thuật toán
        
        Parameters:
        -----------
        histories : list of dict
            Mỗi dict chứa 'tour_lengths': list of float
        labels : list of str
            Tên các thuật toán
        title : str
            Tiêu đề
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # ===== Left: Convergence curves =====
        for history, label in zip(histories, labels):
            if 'tour_lengths' in history:
                iterations = range(len(history['tour_lengths']))
                ax1.plot(iterations, history['tour_lengths'],
                        linewidth=2.5, alpha=0.8, label=label, marker='o',
                        markersize=4, markevery=max(1, len(iterations)//20))
        
        ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Best Tour Length', fontsize=12, fontweight='bold')
        ax1.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # ===== Right: Final comparison bar chart =====
        final_lengths = [history['tour_lengths'][-1] for history in histories 
                        if 'tour_lengths' in history]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
        bars = ax2.bar(labels, final_lengths, color=colors, 
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Highlight best
        best_idx = np.argmin(final_lengths)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('darkgoldenrod')
        bars[best_idx].set_linewidth(3)
        
        ax2.set_ylabel('Final Tour Length', fontsize=12, fontweight='bold')
        ax2.set_title('Final Solution Quality', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add values on top of bars
        for bar, length in zip(bars, final_lengths):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{length:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


class KnapsackVisualizer:
    """Visualization cho Knapsack Problem"""
    
    @staticmethod
    def create_knapsack_animation_frame(knapsack, solution, best_solution,
                                       iteration, algorithm_name,
                                       population_solutions=None):
        """
        Tạo frame animation cho Knapsack Problem
        
        Parameters:
        -----------
        knapsack : Knapsack
            Knapsack problem instance
        solution : numpy.ndarray
            Solution hiện tại (binary array)
        best_solution : numpy.ndarray
            Best solution tìm được
        iteration : int
            Iteration hiện tại
        algorithm_name : str
            Tên thuật toán
        population_solutions : list of numpy.ndarray, optional
            Population solutions (cho swarm algorithms)
        """
        # Layout giống hình: 2 rows x 2 columns + 1 stats panel
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.4, 
                             width_ratios=[1.2, 1.2, 0.8], height_ratios=[1, 0.6])
        
        # Calculate metrics
        n_items = len(knapsack.weights)
        current_weight = knapsack.get_weight(best_solution)
        current_value = knapsack.get_value(best_solution)
        capacity = knapsack.capacity
        n_selected = int(np.sum(best_solution))
        
        # Calculate value/weight ratios
        ratios = knapsack.values / (knapsack.weights + 1e-10)
        
        # ========== PANEL 1: Item Selection Bar Chart (Top Left) ==========
        ax_items = fig.add_subplot(gs[0, 0])
        
        items = np.arange(n_items)
        
        # Colors: green for selected, gray for not selected
        colors = ['limegreen' if best_solution[i] == 1 else 'lightgray' 
                  for i in range(n_items)]
        
        # Horizontal bar chart
        bars = ax_items.barh(items, ratios, color=colors,
                            edgecolor='black', linewidth=1, alpha=0.85)
        
        # Add labels for selected items with high ratio
        for i, (ratio, weight, value) in enumerate(zip(ratios, knapsack.weights, knapsack.values)):
            if best_solution[i] == 1 and ratio > np.percentile(ratios, 70):
                ax_items.text(ratio + 0.5, i, f'V:{int(value)} W:{int(weight)}',
                            va='center', fontsize=8, fontweight='bold')
        
        ax_items.set_xlabel('Value/Weight Ratio', fontsize=11, fontweight='bold')
        ax_items.set_ylabel('Item Index', fontsize=11, fontweight='bold')
        ax_items.set_title(f'Item Selection - Iteration {iteration}\nSelected: {n_selected}/{n_items} items',
                          fontsize=12, fontweight='bold')
        ax_items.grid(True, alpha=0.3, axis='x')
        ax_items.invert_yaxis()  # Để item 0 ở trên cùng
        
        # ========== PANEL 2: Item Distribution Scatter (Top Right) ==========
        ax_scatter = fig.add_subplot(gs[0, 1])
        
        # Scatter plot: Weight vs Value
        for i in range(n_items):
            if best_solution[i] == 1:
                ax_scatter.scatter(knapsack.weights[i], knapsack.values[i],
                                 s=150, c='limegreen', marker='s',
                                 edgecolors='darkgreen', linewidths=2,
                                 alpha=0.8, zorder=3)
            else:
                ax_scatter.scatter(knapsack.weights[i], knapsack.values[i],
                                 s=100, c='lightgray', marker='o',
                                 edgecolors='gray', linewidths=1,
                                 alpha=0.6, zorder=2)
        
        ax_scatter.set_xlabel('Weight', fontsize=11, fontweight='bold')
        ax_scatter.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax_scatter.set_title('Item Distribution', fontsize=12, fontweight='bold')
        ax_scatter.grid(True, alpha=0.3)
        
        # Legend - sử dụng Line2D để có marker
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='limegreen',
                   markeredgecolor='darkgreen', markersize=10, label='Selected'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                   markeredgecolor='gray', markersize=8, label='Not Selected')
        ]
        ax_scatter.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        # ========== PANEL 3: Capacity Usage Bar (Bottom Left) ==========
        ax_capacity = fig.add_subplot(gs[1, 0])
        
        usage_pct = (current_weight / capacity) * 100
        
        # Determine color and status based on usage
        if usage_pct < 80:
            color = 'limegreen'
            status = 'Good'
        elif usage_pct < 100:
            color = 'gold'
            status = 'Good'
        else:
            color = 'orangered'
            status = 'Over Capacity!'
        
        # Draw horizontal progress bar
        ax_capacity.barh([0], [capacity], height=0.5,
                        color='lightgray', edgecolor='black', linewidth=2)
        ax_capacity.barh([0], [current_weight], height=0.5,
                        color=color, edgecolor='darkgreen', linewidth=2, alpha=0.85)
        
        # Add text label on bar
        ax_capacity.text(current_weight / 2, 0, f'{int(current_weight)}kg',
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        color='white' if usage_pct > 50 else 'black')
        
        # Add capacity markers
        ax_capacity.text(0, -0.8, '0', ha='center', va='top', fontsize=10)
        ax_capacity.text(capacity, -0.8, str(int(capacity)), ha='center', va='top', fontsize=10)
        
        # Title with status
        ax_capacity.set_title(f'Capacity Usage\nCapacity: {int(capacity)}kg | Used: {usage_pct:.1f}% | Status: {status}',
                            fontsize=11, fontweight='bold', pad=10)
        
        ax_capacity.set_xlim(0, capacity * 1.1)
        ax_capacity.set_ylim(-1, 1)
        ax_capacity.axis('off')
        
        # ========== PANEL 4: Statistics Panel (Right Side - SPANS 2 ROWS) ==========
        ax_stats = fig.add_subplot(gs[:, 2])
        ax_stats.axis('off')
        
        # Calculate metrics
        value_density = current_value / (current_weight + 1e-10)
        not_selected = n_items - n_selected
        is_valid = knapsack.is_valid(best_solution)
        
        # Create styled text box - giống hình user gửi
        stats_text = f"""
╔══════════════════════════╗
║   KNAPSACK STATISTICS    ║
╚══════════════════════════╝

▣ Total Items: {n_items}
▣ Selected: {n_selected}
▣ Not Selected: {not_selected}

▣ Total Value: {int(current_value)}
▣ Total Weight: {int(current_weight)} / {int(capacity)}
▣ Capacity Used: {usage_pct:.1f}%

▣ Value Density: {value_density:.2f}
  (value per kg)

╔══════════════════════════╗
║      VALIDATION          ║
╚══════════════════════════╝

Valid Solution: ☑ {'YES' if is_valid else 'NO'}
Status: {status}

╔══════════════════════════╗
║      ALGORITHM           ║
╚══════════════════════════╝

Name: {algorithm_name}
Iteration: {iteration}
"""
        
        # Display text with monospace font
        ax_stats.text(0.5, 0.95, stats_text,
                     ha='center', va='top',
                     fontsize=10, family='monospace',
                     transform=ax_stats.transAxes,
                     bbox=dict(boxstyle='round,pad=1.5', 
                              facecolor='lightcyan',
                              edgecolor='darkblue',
                              linewidth=2, alpha=0.9))
        
        # Title at top
        fig.suptitle(f'Knapsack Problem - {algorithm_name}', 
                    fontsize=15, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    
    @staticmethod
    def plot_comparison_table(results, algorithms):
        """
        Tạo bảng so sánh kết quả
        
        Parameters:
        -----------
        results : list of dict
            Mỗi dict chứa: 'value', 'weight', 'valid', 'time'
        algorithms : list of str
            Tên các thuật toán
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = [['Algorithm', 'Best Value', 'Weight Used', 'Valid?', 'Time (s)']]
        
        for algo, result in zip(algorithms, results):
            row = [
                algo,
                f"{result.get('value', 0):.1f}",
                f"{result.get('weight', 0):.1f}",
                "✅" if result.get('valid', False) else "❌",
                f"{result.get('time', 0):.3f}"
            ]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.25, 0.2, 0.2, 0.15, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style header row
        for i in range(len(table_data[0])):
            cell = table[(0, i)]
            cell.set_facecolor('#1f77b4')
            cell.set_text_props(weight='bold', color='white')
        
        # Style data rows (alternating colors)
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                else:
                    cell.set_facecolor('white')
        
        # Highlight best value
        values = [result.get('value', 0) for result in results]
        if values:
            best_idx = np.argmax(values) + 1
            for j in range(len(table_data[0])):
                table[(best_idx, j)].set_facecolor('#90EE90')
        
        plt.title('Knapsack Algorithm Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig


class GraphColoringVisualizer:
    """Visualization cho Graph Coloring Problem"""
    
    @staticmethod
    def create_graph_coloring_frame(graph_coloring, coloring, best_coloring,
                                   iteration, algorithm_name, current_node=None):
        """
        Tạo frame animation cho Graph Coloring
        
        Parameters:
        -----------
        graph_coloring : GraphColoring
            Graph coloring problem instance
        coloring : numpy.ndarray
            Current coloring
        best_coloring : numpy.ndarray
            Best coloring found
        iteration : int
            Current iteration
        algorithm_name : str
            Algorithm name
        current_node : int, optional
            Node đang được xử lý (sẽ highlight bằng vòng đỏ)
        """
        if not HAS_NETWORKX:
            # Fallback visualization without networkx
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'NetworkX not installed\nCannot display graph visualization',
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # ========== MAIN PANEL: Graph Visualization ==========
        ax_graph = fig.add_subplot(gs[:, :2])
        
        # Create networkx graph
        G = nx.Graph()
        n_nodes = len(best_coloring)
        G.add_nodes_from(range(n_nodes))
        
        # Add edges from adjacency matrix
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if graph_coloring.adj_matrix[i, j] == 1:
                    G.add_edge(i, j)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Determine colors for nodes
        # Handle -1 (uncolored nodes) với màu xám nhạt
        colored_nodes = best_coloring[best_coloring >= 0]
        n_colors = int(np.max(colored_nodes)) + 1 if len(colored_nodes) > 0 else 1
        color_palette = plt.cm.Set3(np.linspace(0, 1, max(n_colors, 3)))
        
        node_colors = []
        for i in range(n_nodes):
            if best_coloring[i] == -1:
                # Uncolored node: light gray
                node_colors.append('#E0E0E0')
            else:
                node_colors.append(color_palette[int(best_coloring[i])])
        
        # Draw edges (màu đỏ nếu conflict, xám nếu valid, light gray nếu chưa color)
        edge_colors = []
        edge_widths = []
        for (i, j) in G.edges():
            # Nếu một trong hai node chưa được tô màu
            if best_coloring[i] == -1 or best_coloring[j] == -1:
                edge_colors.append('#CCCCCC')  # Light gray for uncolored
                edge_widths.append(1)
            elif best_coloring[i] == best_coloring[j]:
                edge_colors.append('red')
                edge_widths.append(3)
            else:
                edge_colors.append('gray')
                edge_widths.append(1)
        
        nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color=edge_colors,
                              width=edge_widths, alpha=0.6)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color=node_colors,
                              node_size=500, edgecolors='black', linewidths=2)
        
        # Highlight current node với HIỆU ỨNG CỰC NỔI BẬT
        if current_node is not None and current_node < n_nodes:
            # Layer 1: Outer glow (red - lớn nhất)
            nx.draw_networkx_nodes(G, pos, nodelist=[current_node], ax=ax_graph,
                                  node_color='red', node_size=1500,
                                  edgecolors='none', alpha=0.3, node_shape='o')
            
            # Layer 2: Middle ring (orange)
            nx.draw_networkx_nodes(G, pos, nodelist=[current_node], ax=ax_graph,
                                  node_color='orange', node_size=1000,
                                  edgecolors='red', linewidths=4, alpha=0.6, node_shape='o')
            
            # Layer 3: Inner node (current color với red border)
            nx.draw_networkx_nodes(G, pos, nodelist=[current_node], ax=ax_graph,
                                  node_color=node_colors[current_node:current_node+1],
                                  node_size=700, edgecolors='red', linewidths=7,
                                  node_shape='o', alpha=1.0)
            
            # Add LARGE arrow and label
            node_pos = pos[current_node]
            arrow_offset = 0.15
            arrow_start = [node_pos[0] + arrow_offset, node_pos[1] + arrow_offset]
            
            ax_graph.annotate('', xy=node_pos, xytext=arrow_start,
                            arrowprops=dict(arrowstyle='->', lw=5, color='red',
                                          mutation_scale=35, alpha=0.9),
                            zorder=100)
            
            # Add LARGE text label
            ax_graph.text(arrow_start[0] + 0.05, arrow_start[1] + 0.05,
                         f'◄ PROCESSING\nNODE {current_node}',
                         fontsize=13, fontweight='bold',
                         color='red', zorder=100,
                         bbox=dict(boxstyle='round,pad=0.8',
                                 facecolor='yellow',
                                 edgecolor='red',
                                 linewidth=3,
                                 alpha=0.95))
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=10,
                               font_weight='bold', font_color='black')
        
        # Count conflicts
        n_conflicts = graph_coloring.count_conflicts(best_coloring)
        
        ax_graph.set_title(f'{algorithm_name} - Graph Coloring\n'
                          f'Iteration {iteration} | Colors Used: {n_colors} | '
                          f'Conflicts: {n_conflicts}',
                          fontsize=14, fontweight='bold')
        ax_graph.axis('off')
        
        # ========== CONFLICT HEATMAP ==========
        ax_conflict = fig.add_subplot(gs[0, 2])
        
        # Create conflict matrix
        conflict_matrix = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if graph_coloring.adj_matrix[i, j] == 1:
                    if best_coloring[i] == best_coloring[j]:
                        conflict_matrix[i, j] = 1
        
        im = ax_conflict.imshow(conflict_matrix, cmap='RdYlGn_r',
                               aspect='auto', interpolation='nearest')
        ax_conflict.set_xlabel('Node', fontsize=10, fontweight='bold')
        ax_conflict.set_ylabel('Node', fontsize=10, fontweight='bold')
        ax_conflict.set_title('Conflict Matrix', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax_conflict, fraction=0.046, pad=0.04)
        
        # ========== COLOR DISTRIBUTION ==========
        ax_dist = fig.add_subplot(gs[1, 2])
        
        # Count nodes per color (ignore -1 = uncolored)
        colored_only = best_coloring[best_coloring >= 0]
        if len(colored_only) > 0:
            color_counts = np.bincount(colored_only.astype(int))
            colors_used = np.arange(len(color_counts))
        else:
            color_counts = np.array([])
            colors_used = np.array([])
        
        if len(colors_used) > 0:
            bars = ax_dist.bar(colors_used, color_counts,
                              color=[color_palette[i] for i in colors_used],
                              edgecolor='black', linewidth=1.5, alpha=0.8)
        else:
            ax_dist.text(0.5, 0.5, 'No nodes colored yet', 
                        ha='center', va='center', fontsize=10)
        
        ax_dist.set_xlabel('Color ID', fontsize=11, fontweight='bold')
        ax_dist.set_ylabel('Number of Nodes', fontsize=11, fontweight='bold')
        ax_dist.set_title('Color Distribution', fontsize=12, fontweight='bold')
        ax_dist.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, color_counts):
            height = bar.get_height()
            ax_dist.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(count)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return fig


class DiscreteComparisonVisualizer:
    """Visualization cho so sánh nhiều thuật toán"""
    
    @staticmethod
    def create_multi_algorithm_grid(problem, algorithms, solutions, 
                                   problem_type='tsp'):
        """
        Tạo grid view so sánh nhiều thuật toán
        
        Parameters:
        -----------
        problem : TSP, Knapsack, or GraphColoring
            Problem instance
        algorithms : list of str
            Algorithm names
        solutions : list
            Solutions from each algorithm
        problem_type : str
            'tsp', 'knapsack', or 'graph_coloring'
        """
        n_algos = len(algorithms)
        n_rows = (n_algos + 2) // 3
        n_cols = min(3, n_algos)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if n_algos == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each algorithm's solution
        for idx, (algo, sol, ax) in enumerate(zip(algorithms, solutions, axes)):
            if problem_type == 'tsp':
                # TSP visualization
                cities = problem.cities
                if sol is not None and len(sol) > 0:
                    tour_extended = list(sol) + [sol[0]]
                    tour_coords = cities[tour_extended]
                    ax.plot(tour_coords[:, 0], tour_coords[:, 1],
                           'b-', linewidth=2, alpha=0.8)
                    ax.scatter(cities[:, 0], cities[:, 1],
                             c='red', s=100, zorder=5)
                    length = problem.evaluate(sol)
                    ax.set_title(f'{algo}\nLength: {length:.2f}',
                               fontsize=12, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'No Solution', ha='center', va='center',
                           transform=ax.transAxes)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
            
            # Disable unused axes
            for idx in range(len(algorithms), len(axes)):
                axes[idx].axis('off')
        
        plt.suptitle(f'{problem_type.upper()} Algorithm Comparison',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_convergence_race(histories, labels, metric_name='Best Score'):
        """
        Tạo convergence race plot
        
        Parameters:
        -----------
        histories : list of dict
            History từ mỗi thuật toán
        labels : list of str
            Algorithm names
        metric_name : str
            Tên metric để plot
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        
        for history, label, color in zip(histories, labels, colors):
            if 'best_scores' in history:
                scores = history['best_scores']
            elif 'tour_lengths' in history:
                scores = history['tour_lengths']
            elif 'best_values' in history:
                scores = history['best_values']
            else:
                continue
            
            iterations = range(len(scores))
            ax.plot(iterations, scores, linewidth=3, alpha=0.8,
                   label=label, color=color, marker='o',
                   markersize=5, markevery=max(1, len(iterations)//20))
        
        ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')
        ax.set_title(f'Convergence Race - {metric_name}',
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig


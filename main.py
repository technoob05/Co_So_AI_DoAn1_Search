import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import time
from matplotlib import cm
import json
from datetime import datetime
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import os

from src.swarm_intelligence.pso import PSO
from src.swarm_intelligence.aco import ACO
from src.swarm_intelligence.abc import ABC
from src.swarm_intelligence.fa import FireflyAlgorithm
from src.swarm_intelligence.cs import CuckooSearch

from src.traditional_search.hill_climbing import HillClimbing
from src.traditional_search.simulated_annealing import SimulatedAnnealing
from src.traditional_search.genetic_algorithm import GeneticAlgorithm
from src.traditional_search.graph_search import BreadthFirstSearch, DepthFirstSearch, AStarSearch, GridWorld

from src.test_functions import get_test_function
from src.discrete_problems.tsp import TSP, TSPSolver
from src.discrete_problems.knapsack import Knapsack, KnapsackSolver
from src.discrete_problems.graph_coloring import GraphColoring, GraphColoringSolver

from src.visualization import OptimizationVisualizer
from src.visualization_discrete import TSPVisualizer, KnapsackVisualizer, GraphColoringVisualizer, DiscreteComparisonVisualizer

st.set_page_config(
    page_title="Optimization Algorithms Comparison",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Optimization Algorithms Visualization\nSo sánh Swarm Intelligence và Traditional Search"
    }
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# UTILS

@st.cache_data
def load_swarm_image(algorithm_name):
    """Load swarm algorithm images"""
    image_map = {
        "PSO": "fish.png",
        "ACO": "ant.png",
        "ABC": "bee.png",
        "Firefly": "firefly.png",
        "Cuckoo": "cuckoo.png"
    }
    
    if algorithm_name not in image_map:
        return None
    
    image_path = os.path.join("assets", image_map[algorithm_name])
    
    if not os.path.exists(image_path):
        return None
    
    try:
        img = Image.open(image_path)
        size = (55, 55) if algorithm_name == "Firefly" else (70, 70)
        img = img.resize(size, Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image for {algorithm_name}: {e}")
        return None

@st.cache_data
def get_algorithm_info():
    """Algorithm information"""
    return {
        "PSO": {
            "name": "Particle Swarm Optimization",
            "type": "swarm",
            "description": "Mô phỏng hành vi bầy đàn của chim và cá. Các hạt (particles) di chuyển trong không gian tìm kiếm, học từ kinh nghiệm cá nhân và tập thể.",
            "params": ["n_particles", "w", "c1", "c2"],
            "best_for": "Tối ưu liên tục, hội tụ nhanh"
        },
        "ACO": {
            "name": "Ant Colony Optimization",
            "type": "swarm",
            "description": "Lấy cảm hứng từ kiến tìm đường. Sử dụng pheromone để chia sẻ thông tin và tìm giải pháp tốt.",
            "params": ["n_ants", "archive_size", "q", "xi"],
            "best_for": "Bài toán tổ hợp, TSP"
        },
        "ABC": {
            "name": "Artificial Bee Colony",
            "type": "swarm",
            "description": "Mô phỏng ong mật tìm nguồn thức ăn. Ba loại ong: employed, onlooker, scout với vai trò khác nhau.",
            "params": ["n_bees", "limit"],
            "best_for": "Cân bằng exploration và exploitation"
        },
        "Firefly": {
            "name": "Firefly Algorithm",
            "type": "swarm",
            "description": "Dựa trên ánh sáng của đom đóm. Các cá thể bị thu hút bởi những cá thể sáng hơn (fitness tốt hơn).",
            "params": ["n_fireflies", "alpha", "beta0", "gamma"],
            "best_for": "Multimodal optimization"
        },
        "Cuckoo": {
            "name": "Cuckoo Search",
            "type": "swarm",
            "description": "Dựa trên hành vi đẻ trứng ký sinh của chim cu-cu. Sử dụng Lévy flights để탐 hiểm không gian.",
            "params": ["n_nests", "pa", "beta"],
            "best_for": "Global optimization, Lévy flights hiệu quả"
        },
        
        # Traditional Search
        "Hill Climbing": {
            "name": "Hill Climbing",
            "type": "traditional",
            "description": "Thuật toán tìm kiếm cục bộ đơn giản. Leo dần lên đỉnh bằng cách chọn láng giềng tốt nhất.",
            "params": ["step_size"],
            "best_for": "Nhanh, đơn giản, dễ bị stuck ở local optimum"
        },
        "Simulated Annealing": {
            "name": "Simulated Annealing",
            "type": "traditional",
            "description": "Mô phỏng quá trình ủ kim loại. Chấp nhận giải pháp xấu hơn với xác suất giảm dần theo nhiệt độ.",
            "params": ["initial_temp", "alpha", "step_size"],
            "best_for": "Thoát khỏi local optimum"
        },
        "Genetic Algorithm": {
            "name": "Genetic Algorithm",
            "type": "traditional",
            "description": "Mô phỏng quá trình tiến hóa tự nhiên. Sử dụng selection, crossover, mutation.",
            "params": ["population_size", "crossover_rate", "mutation_rate", "elite_size"],
            "best_for": "Robust, versatile"
        },
        "BFS": {
            "name": "Breadth-First Search",
            "type": "graph",
            "description": "Tìm kiếm theo chiều rộng. Đảm bảo tìm được đường đi ngắn nhất trên đồ thị không trọng số.",
            "params": [],
            "best_for": "Đường đi ngắn nhất, hoàn chỉnh"
        },
        "DFS": {
            "name": "Depth-First Search",
            "type": "graph",
            "description": "Tìm kiếm theo chiều sâu. Đi sâu vào một nhánh trước khi quay lại.",
            "params": ["max_depth"],
            "best_for": "Tiết kiệm bộ nhớ, không đảm bảo optimal"
        },
        "A*": {
            "name": "A* Search",
            "type": "graph",
            "description": "Tìm kiếm có thông tin với heuristic. Đảm bảo tìm được đường đi tối ưu nếu heuristic admissible.",
            "params": ["heuristic"],
            "best_for": "Tối ưu và hiệu quả với heuristic tốt"
        }
    }

def create_animated_3d_plot(func, particles, best_pos, best_score, iteration, algorithm_name, func_name):
    """Create 3D plot with 2D contour view"""
    fig = plt.figure(figsize=(16, 7))
    
    x = np.linspace(func.bounds[0, 0], func.bounds[0, 1], 50)
    y = np.linspace(func.bounds[1, 0], func.bounds[1, 1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = func(np.array([X[j, i], Y[j, i]]))
    
    ax1 = fig.add_subplot(121, projection='3d')
    
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.6,
                           antialiased=True, linewidth=0, zorder=1)
    
    ax1.contour(X, Y, Z, levels=15, cmap=cm.coolwarm, alpha=0.3,
                offset=np.min(Z), zorder=0)
    
    swarm_img = load_swarm_image(algorithm_name)
    
    if particles is not None and len(particles) > 0:
        particle_z = np.array([func(p) for p in particles])
        
        if swarm_img is not None:
            particle_size = 150 if algorithm_name == "Firefly" else 180
            edge_width = 2.2 if algorithm_name == "Firefly" else 2.5
            colors = plt.cm.RdYlGn_r((particle_z - particle_z.min()) / 
                                     (particle_z.max() - particle_z.min() + 1e-10))
            ax1.scatter(particles[:, 0], particles[:, 1], particle_z,
                       c=colors, s=particle_size, marker='o', alpha=0.9,
                       edgecolors='white', linewidths=edge_width,
                       label=f'{algorithm_name} Agents', zorder=5)
        else:
            colors = plt.cm.RdYlGn_r((particle_z - particle_z.min()) / 
                                     (particle_z.max() - particle_z.min() + 1e-10))
            ax1.scatter(particles[:, 0], particles[:, 1], particle_z,
                       c=colors, s=100, marker='o', alpha=0.8,
                       edgecolors='black', linewidths=1.5,
                       label='Population', zorder=5)
    
    if best_pos is not None:
        ax1.scatter([best_pos[0]], [best_pos[1]], [best_score],
                   c='red', marker='*', s=600,
                   edgecolors='darkred', linewidths=3,
                   label=f'Best (f={best_score:.4f})', zorder=10)
    
    ax1.scatter([0], [0], [func.global_optimum],
               c='lime', marker='*', s=700,
               edgecolors='darkgreen', linewidths=3,
               label=f'Global Optimum (f={func.global_optimum})', zorder=10)
    
    ax1.set_xlabel('X₁', fontsize=11, fontweight='bold')
    ax1.set_ylabel('X₂', fontsize=11, fontweight='bold')
    ax1.set_zlabel('f(X)', fontsize=11, fontweight='bold')
    ax1.set_title(f'3D View - {algorithm_name}\nIteration {iteration}',
                 fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.view_init(elev=25, azim=45 + iteration * 2)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(122)
    
    contourf = ax2.contourf(X, Y, Z, levels=30, cmap=cm.coolwarm, alpha=0.8)
    contour = ax2.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    
    if particles is not None and len(particles) > 0:
        particle_z = np.array([func(p) for p in particles])
        
        if swarm_img is not None:
            # Glow effect
            glow_size = 250 if algorithm_name == "Firefly" else 300
            colors_for_glow = plt.cm.RdYlGn_r((particle_z - particle_z.min()) / 
                                              (particle_z.max() - particle_z.min() + 1e-10))
            ax2.scatter(particles[:, 0], particles[:, 1],
                       c=colors_for_glow, s=glow_size, marker='o', alpha=0.3,
                       edgecolors='none', zorder=4)
            
            # Display images
            for i, (px, py) in enumerate(particles):
                zoom_level = 0.32 if algorithm_name == "Firefly" else 0.4
                imagebox = OffsetImage(swarm_img, zoom=zoom_level)
                ab = AnnotationBbox(imagebox, (px, py), frameon=False, pad=0, zorder=5)
                ax2.add_artist(ab)
            
            ax2.scatter([], [], c='blue', s=200, marker='o', alpha=0.8,
                       edgecolors='white', linewidths=2,
                       label=f'{algorithm_name} Agents', zorder=5)
        else:
            colors = plt.cm.RdYlGn_r((particle_z - particle_z.min()) / 
                                     (particle_z.max() - particle_z.min() + 1e-10))
            
            ax2.scatter(particles[:, 0], particles[:, 1],
                       c=colors, s=150, marker='o', alpha=0.9,
                       edgecolors='black', linewidths=2,
                       label='Population', zorder=5)
            
            if len(particles) > 1:
                for i in range(len(particles)):
                    ax2.plot([particles[i, 0]], [particles[i, 1]], 
                            'o', markersize=3, alpha=0.3, color=colors[i])
    
    if best_pos is not None:
        ax2.scatter([best_pos[0]], [best_pos[1]],
                   c='red', marker='*', s=800,
                   edgecolors='darkred', linewidths=4,
                   label=f'Best', zorder=10)
        
        circle = plt.Circle((best_pos[0], best_pos[1]), 
                           (func.bounds[0, 1] - func.bounds[0, 0]) * 0.05,
                           color='red', fill=False, linewidth=2, 
                           linestyle='--', alpha=0.5, zorder=9)
        ax2.add_patch(circle)
    
    ax2.scatter([0], [0],
               c='lime', marker='*', s=900,
               edgecolors='darkgreen', linewidths=4,
               label='Global Optimum', zorder=10)
    
    ax2.plot([0], [0], 'x', color='darkgreen', markersize=15, 
            markeredgewidth=3, zorder=11)
    
    ax2.set_xlabel('X₁', fontsize=11, fontweight='bold')
    ax2.set_ylabel('X₂', fontsize=11, fontweight='bold')
    ax2.set_title(f'Top-Down View (Contour)\n{func_name.capitalize()} Function',
                 fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.set_aspect('equal', adjustable='box')
    
    cbar = fig.colorbar(contourf, ax=ax2, shrink=0.8, aspect=20)
    cbar.set_label('f(X₁, X₂)', fontsize=10, fontweight='bold')
    
    fig.suptitle(f'{algorithm_name} on {func_name.capitalize()} - Iteration {iteration}/{iteration}',
                fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig

# SIDEBAR

st.sidebar.markdown("# ⚙️ Cấu Hình")

if 'main_tab' not in st.session_state:
    st.session_state.main_tab = "🎬 Visualization & Demo"

st.sidebar.markdown("### Chọn chức năng:")

tab_options = [
    {"name": "🎬 Visualization & Demo", "icon": "🎬", "desc": "Real-time visualization"},
    {"name": "📊 Comparison Dashboard", "icon": "📊", "desc": "So sánh thuật toán"},
    {"name": "ℹ️ Algorithm Info", "icon": "ℹ️", "desc": "Thông tin chi tiết"}
]

for option in tab_options:
    is_selected = st.session_state.main_tab == option["name"]
    
    if is_selected:
        box_style = """
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border: 2px solid #764ba2;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        cursor: pointer;
        """
    else:
        box_style = """
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: all 0.3s;
        """
    
    if st.sidebar.button(
        f"{option['icon']} **{option['name'].replace(option['icon'], '').strip()}**",
        key=f"tab_{option['name']}",
        use_container_width=True,
        type="primary" if is_selected else "secondary"
    ):
        st.session_state.main_tab = option["name"]
        st.rerun()

main_tab = st.session_state.main_tab

st.sidebar.markdown("---")

if main_tab == "🎬 Visualization & Demo":
    st.sidebar.markdown("### 🎯 Chọn Bài Toán")
    
    # Cấp 1: Chọn loại bài toán chính
    problem_category = st.sidebar.radio(
        "Loại bài toán:",
        ["Continuous Optimization", "Discrete Optimization"],
        help="Chọn loại bài toán tối ưu"
    )
    
    # Cấp 2: Chọn bài toán cụ thể (nếu là Discrete)
    if problem_category == "Continuous Optimization":
        problem_type = "Continuous Optimization"
        
        function_name = st.sidebar.selectbox(
            "Test Function:",
            ["Sphere", "Rastrigin", "Rosenbrock", "Ackley"]
        )
        
        dimensions = st.sidebar.slider("Số chiều (dimension):", 2, 10, 2)
        
    else:  # Discrete Optimization
        discrete_problem = st.sidebar.selectbox(
            "Chọn bài toán cụ thể:",
            ["TSP (Traveling Salesman)", "Knapsack", "Graph Coloring", "Path Finding (Grid World)"],
            help="Chọn bài toán rời rạc cụ thể"
        )
        
        # Map discrete_problem to problem_type for backward compatibility
        if discrete_problem == "TSP (Traveling Salesman)":
            problem_type = "Discrete Optimization (TSP)"
            n_cities = st.sidebar.slider("Số thành phố:", 5, 30, 15)
            
        elif discrete_problem == "Knapsack":
            problem_type = "Discrete Optimization (Knapsack)"
            n_items = st.sidebar.slider("Số items:", 10, 50, 20)
            
        elif discrete_problem == "Graph Coloring":
            problem_type = "Discrete Optimization (Graph Coloring)"
            n_nodes = st.sidebar.slider("Số nodes:", 5, 30, 15)
            edge_prob = st.sidebar.slider("Xác suất edge:", 0.1, 0.8, 0.3)
            viz_update_freq = st.sidebar.slider("Visualization Update (mỗi N iterations):", 1, 10, 2, 
                                               help="1 = update mỗi iteration (chậm), 5-10 = update ít hơn (nhanh hơn)")
            
        else:  # Path Finding
            problem_type = "Path Finding (Grid World)"
            grid_size = st.sidebar.slider("Kích thước grid:", 10, 30, 20)
            obstacle_prob = st.sidebar.slider("Xác suất vật cản:", 0.0, 0.4, 0.2)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Chọn Thuật Toán")
    
    if problem_type == "Continuous Optimization":
        algo_category = st.sidebar.radio("Loại thuật toán:", ["Swarm Intelligence", "Traditional Search"])
        
        if algo_category == "Swarm Intelligence":
            algorithm = st.sidebar.selectbox(
                "Thuật toán:",
                ["PSO", "ACO", "ABC", "Firefly", "Cuckoo"]
            )
        else:
            algorithm = st.sidebar.selectbox(
                "Thuật toán:",
                ["Hill Climbing", "Simulated Annealing", "Genetic Algorithm"]
            )
    else:
        if problem_type == "Path Finding (Grid World)":
            algorithm = st.sidebar.selectbox(
                "Thuật toán:",
                ["BFS", "DFS", "A*"]
            )
        elif problem_type == "Discrete Optimization (Graph Coloring)":
            algorithm = st.sidebar.selectbox(
                "Thuật toán:",
                ["Genetic Algorithm", "Hill Climbing", "Simulated Annealing"]
            )
        else:
            algorithm = st.sidebar.selectbox(
                "Thuật toán:",
                ["Genetic Algorithm", "Hill Climbing", "Simulated Annealing"]
            )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Tham Số Thuật Toán")
    
    # Tham số chung
    if problem_type == "Continuous Optimization":
        if algorithm in ["PSO", "ACO", "ABC", "Firefly", "Cuckoo"]:
            population_size = st.sidebar.slider("Population/Swarm Size:", 10, 100, 30, 5)
        elif algorithm == "Genetic Algorithm":
            population_size = st.sidebar.slider("Population Size:", 10, 100, 50, 5)
        else:
            population_size = None
        
        max_iter = st.sidebar.slider("Max Iterations:", 10, 200, 50, 10)
    
    elif problem_type in ["Discrete Optimization (TSP)", "Discrete Optimization (Knapsack)", 
                          "Discrete Optimization (Graph Coloring)"]:
        # Parameters for discrete problems
        max_iter = st.sidebar.slider("Max Iterations:", 10, 200, 100, 10)
        
        if algorithm == "Genetic Algorithm":
            # GA parameters already defined below
            pass
    
    else:
        # Path finding - no additional parameters needed
        pass
    
    # Tham số đặc trưng từng thuật toán
    if algorithm == "PSO":
        w = st.sidebar.slider("Inertia weight (w):", 0.1, 1.0, 0.7, 0.1, 
                             help="Cân bằng exploration vs exploitation")
        c1 = st.sidebar.slider("Cognitive param (c1):", 0.5, 3.0, 1.5, 0.1,
                              help="Thu hút về personal best")
        c2 = st.sidebar.slider("Social param (c2):", 0.5, 3.0, 1.5, 0.1,
                              help="Thu hút về global best")
    
    elif algorithm == "ACO":
        archive_size = st.sidebar.slider("Archive Size:", 20, 100, 50, 10)
        q = st.sidebar.slider("Locality (q):", 0.1, 1.0, 0.5, 0.1)
        xi = st.sidebar.slider("Convergence speed (xi):", 0.5, 1.0, 0.85, 0.05)
    
    elif algorithm == "ABC":
        limit = st.sidebar.slider("Abandonment limit:", 10, 50, 20, 5,
                                 help="Số lần không cải thiện trước khi bỏ food source")
    
    elif algorithm == "Firefly":
        alpha = st.sidebar.slider("Randomization (alpha):", 0.1, 1.0, 0.5, 0.1)
        beta0 = st.sidebar.slider("Attractiveness (beta0):", 0.5, 2.0, 1.0, 0.1)
        gamma = st.sidebar.slider("Absorption (gamma):", 0.1, 2.0, 1.0, 0.1)
    
    elif algorithm == "Cuckoo":
        pa = st.sidebar.slider("Discovery probability (pa):", 0.1, 0.5, 0.25, 0.05)
        beta = st.sidebar.slider("Lévy exponent (beta):", 1.0, 3.0, 1.5, 0.1)
    
    elif algorithm == "Simulated Annealing":
        initial_temp = st.sidebar.slider("Initial temperature:", 50.0, 200.0, 100.0, 10.0)
        cooling_rate = st.sidebar.slider("Cooling rate (alpha):", 0.8, 0.99, 0.95, 0.01)
        step_size = st.sidebar.slider("Step size:", 0.1, 2.0, 1.0, 0.1)
    
    elif algorithm == "Hill Climbing":
        step_size = st.sidebar.slider("Step size:", 0.1, 2.0, 0.5, 0.1)
    
    elif algorithm == "Genetic Algorithm":
        crossover_rate = st.sidebar.slider("Crossover rate:", 0.5, 1.0, 0.8, 0.05)
        mutation_rate = st.sidebar.slider("Mutation rate:", 0.01, 0.3, 0.1, 0.01)
        elite_size = st.sidebar.slider("Elite size:", 1, 10, 2, 1)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Animation")
    
    animation_speed = st.sidebar.slider(
        "Animation speed (delay):",
        min_value=0.01,
        max_value=1.0,
        value=0.2,
        step=0.01,
        help="Độ trễ giữa các frame (giây)"
    )
    
    st.sidebar.markdown("---")
    
    run_button = st.sidebar.button("▶️ Run Animation", type="primary", use_container_width=True)

# MAIN CONTENT

# Header đẹp hơn
st.markdown("""
<div style='text-align: center; padding: 2rem 0 1rem 0;'>
    <h1 style='color: #667eea; font-size: 3rem; font-weight: 800; margin: 0;'>
        🔬 Optimization Algorithms Visualization
    </h1>
    <p style='color: #666; font-size: 1.3rem; margin-top: 0.5rem;'>
        So sánh Swarm Intelligence với Traditional Search Algorithms
    </p>
</div>
<hr style='border: none; height: 3px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); margin: 1.5rem 0;'>
""", unsafe_allow_html=True)

if main_tab == "🎬 Visualization & Demo":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; border-left: 5px solid #667eea;'>
        <h2 style='margin: 0; color: #667eea;'>🎬 Visualization & Demo</h2>
        <p style='margin: 0.5rem 0 0 0; color: #666;'>Real-time visualization của thuật toán optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    if run_button:
        if problem_type == "Continuous Optimization":
            func = get_test_function(function_name, dim=dimensions)
            
            if algorithm == "PSO":
                algo = PSO(n_particles=population_size, dim=dimensions, max_iter=max_iter,
                          w=w, c1=c1, c2=c2, bounds=func.bounds)
            elif algorithm == "ACO":
                algo = ACO(n_ants=population_size, dim=dimensions, max_iter=max_iter,
                          archive_size=archive_size, q=q, xi=xi, bounds=func.bounds)
            elif algorithm == "ABC":
                algo = ABC(n_bees=population_size, dim=dimensions, max_iter=max_iter,
                          limit=limit, bounds=func.bounds)
            elif algorithm == "Firefly":
                algo = FireflyAlgorithm(n_fireflies=population_size, dim=dimensions, max_iter=max_iter,
                                       alpha=alpha, beta0=beta0, gamma=gamma, bounds=func.bounds)
            elif algorithm == "Cuckoo":
                algo = CuckooSearch(n_nests=population_size, dim=dimensions, max_iter=max_iter,
                                   pa=pa, beta=beta, bounds=func.bounds)
            elif algorithm == "Hill Climbing":
                algo = HillClimbing(dim=dimensions, max_iter=max_iter, step_size=step_size,
                                   bounds=func.bounds)
            elif algorithm == "Simulated Annealing":
                algo = SimulatedAnnealing(dim=dimensions, max_iter=max_iter,
                                         initial_temp=initial_temp, alpha=cooling_rate,
                                         step_size=step_size, bounds=func.bounds)
            elif algorithm == "Genetic Algorithm":
                algo = GeneticAlgorithm(population_size=population_size, dim=dimensions,
                                       max_iter=max_iter, crossover_rate=crossover_rate,
                                       mutation_rate=mutation_rate, elite_size=elite_size,
                                       bounds=func.bounds)
            
            if dimensions == 2:
                st.success(f"✅ Đang chạy {algorithm} trên {function_name.upper()}...")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    plot_placeholder = st.empty()
                
                with col2:
                    st.markdown("### 📊 Metrics")
                    iteration_metric = st.empty()
                    score_metric = st.empty()
                    gap_metric = st.empty()
                    
                    st.markdown("### 📈 Convergence")
                    convergence_placeholder = st.empty()
                
                progress_bar = st.progress(0)
                
                algo.initialize()
                
                if algorithm == "Hill Climbing":
                    algo.best_score = func(algo.current_solution)
                    algo.best_solution = algo.current_solution.copy()
                elif algorithm == "Simulated Annealing":
                    algo.current_score = func(algo.current_solution)
                    algo.best_score = algo.current_score
                    algo.best_solution = algo.current_solution.copy()
                elif algorithm == "Genetic Algorithm":
                    algo.population = np.random.uniform(
                        algo.bounds[:, 0], algo.bounds[:, 1], 
                        (algo.population_size, algo.dim)
                    )
                    fitness = np.array([func(ind) for ind in algo.population])
                    best_idx = np.argmin(fitness)
                    algo.best_solution = algo.population[best_idx].copy()
                    algo.best_score = fitness[best_idx]
                
                best_scores_history = []
                
                for iteration in range(max_iter):
                    if algorithm in ["PSO", "ACO", "ABC", "Firefly", "Cuckoo"]:
                        if algorithm == "PSO":
                            scores = np.array([func(pos) for pos in algo.positions])
                            improved = scores < algo.personal_best_scores
                            algo.personal_best_scores[improved] = scores[improved]
                            algo.personal_best_positions[improved] = algo.positions[improved]
                            
                            best_idx = np.argmin(scores)
                            if scores[best_idx] < algo.global_best_score:
                                algo.global_best_score = scores[best_idx]
                                algo.global_best_position = algo.positions[best_idx].copy()
                            
                            r1 = np.random.rand(algo.n_particles, algo.dim)
                            r2 = np.random.rand(algo.n_particles, algo.dim)
                            
                            cognitive = algo.c1 * r1 * (algo.personal_best_positions - algo.positions)
                            social = algo.c2 * r2 * (algo.global_best_position - algo.positions)
                            
                            algo.velocities = algo.w * algo.velocities + cognitive + social
                            algo.positions = algo.positions + algo.velocities
                            algo.positions = np.clip(algo.positions, algo.bounds[:, 0], algo.bounds[:, 1])
                            
                            particles = algo.positions.copy()
                            best_pos = algo.global_best_position.copy()
                            best_score = algo.global_best_score
                        
                        elif algorithm == "ACO":
                            new_solutions = np.zeros((algo.n_ants, algo.dim))
                            new_scores = np.zeros(algo.n_ants)
                            
                            for ant in range(algo.n_ants):
                                solution = np.zeros(algo.dim)
                                weights = np.zeros(algo.archive_size)
                                for i in range(algo.archive_size):
                                    weights[i] = 1.0 / (algo.q * algo.archive_size * np.sqrt(2 * np.pi)) * \
                                                np.exp(-(i ** 2) / (2 * algo.q ** 2 * algo.archive_size ** 2))
                                weights /= np.sum(weights)
                                
                                for d in range(algo.dim):
                                    selected_idx = np.random.choice(algo.archive_size, p=weights)
                                    sum_distances = np.sum(np.abs(algo.archive[:, d] - algo.archive[selected_idx, d]))
                                    std = algo.xi / (algo.archive_size - 1) * sum_distances if algo.archive_size > 1 else 0.1
                                    solution[d] = np.random.normal(algo.archive[selected_idx, d], std + 1e-10)
                                    solution[d] = np.clip(solution[d], algo.bounds[d, 0], algo.bounds[d, 1])
                                
                                new_solutions[ant] = solution
                                new_scores[ant] = func(solution)
                            
                            combined_solutions = np.vstack([algo.archive, new_solutions])
                            combined_scores = np.concatenate([algo.archive_scores, new_scores])
                            sorted_indices = np.argsort(combined_scores)
                            algo.archive = combined_solutions[sorted_indices[:algo.archive_size]]
                            algo.archive_scores = combined_scores[sorted_indices[:algo.archive_size]]
                            
                            if algo.archive_scores[0] < algo.best_score:
                                algo.best_score = algo.archive_scores[0]
                                algo.best_solution = algo.archive[0].copy()
                            
                            particles = algo.archive[:min(30, algo.archive_size)]
                            best_pos = algo.best_solution.copy()
                            best_score = algo.best_score
                        
                        elif algorithm == "ABC":
                            for i in range(algo.n_bees):
                                phi = np.random.uniform(-1, 1, algo.dim)
                                neighbor = np.random.choice([j for j in range(algo.n_bees) if j != i])
                                new_solution = algo.food_sources[i] + phi * (algo.food_sources[i] - algo.food_sources[neighbor])
                                new_solution = np.clip(new_solution, algo.bounds[:, 0], algo.bounds[:, 1])
                                
                                new_score = func(new_solution)
                                new_fitness = algo._calculate_fitness(new_score)
                                
                                if new_fitness > algo.fitness[i]:
                                    algo.food_sources[i] = new_solution
                                    algo.fitness[i] = new_fitness
                                    algo.trials[i] = 0
                                else:
                                    algo.trials[i] += 1
                            
                            total_fitness = np.sum(algo.fitness)
                            probabilities = algo.fitness / total_fitness if total_fitness > 0 else np.ones(algo.n_bees) / algo.n_bees
                            
                            for _ in range(algo.n_bees):
                                i = np.random.choice(algo.n_bees, p=probabilities)
                                phi = np.random.uniform(-1, 1, algo.dim)
                                neighbor = np.random.choice([j for j in range(algo.n_bees) if j != i])
                                new_solution = algo.food_sources[i] + phi * (algo.food_sources[i] - algo.food_sources[neighbor])
                                new_solution = np.clip(new_solution, algo.bounds[:, 0], algo.bounds[:, 1])
                                
                                new_score = func(new_solution)
                                new_fitness = algo._calculate_fitness(new_score)
                                
                                if new_fitness > algo.fitness[i]:
                                    algo.food_sources[i] = new_solution
                                    algo.fitness[i] = new_fitness
                                    algo.trials[i] = 0
                            
                            for i in range(algo.n_bees):
                                if algo.trials[i] >= algo.limit:
                                    algo.food_sources[i] = np.random.uniform(algo.bounds[:, 0], algo.bounds[:, 1], algo.dim)
                                    algo.trials[i] = 0
                            
                            scores = np.array([func(source) for source in algo.food_sources])
                            best_idx = np.argmin(scores)
                            if scores[best_idx] < algo.best_score:
                                algo.best_score = scores[best_idx]
                                algo.best_solution = algo.food_sources[best_idx].copy()
                            
                            particles = algo.food_sources.copy()
                            best_pos = algo.best_solution.copy()
                            best_score = algo.best_score
                        
                        elif algorithm == "Firefly":
                            alpha_current = algo.alpha * (0.95 ** iteration)
                            
                            original_intensity = algo.light_intensity.copy()
                            
                            for i in range(algo.n_fireflies):
                                for j in range(algo.n_fireflies):
                                    if original_intensity[j] < original_intensity[i]:
                                        r = np.sqrt(np.sum((algo.fireflies[i] - algo.fireflies[j]) ** 2))
                                        beta = algo.beta0 * np.exp(-algo.gamma * r ** 2)
                                        epsilon = np.random.uniform(-0.5, 0.5, algo.dim)
                                        
                                        algo.fireflies[i] = algo.fireflies[i] + beta * (algo.fireflies[j] - algo.fireflies[i]) + \
                                                           alpha_current * epsilon
                                        algo.fireflies[i] = np.clip(algo.fireflies[i], algo.bounds[:, 0], algo.bounds[:, 1])
                            
                            algo.light_intensity = np.array([func(ff) for ff in algo.fireflies])
                            
                            best_idx = np.argmin(algo.light_intensity)
                            if algo.light_intensity[best_idx] < algo.best_score:
                                algo.best_score = algo.light_intensity[best_idx]
                                algo.best_firefly = algo.fireflies[best_idx].copy()
                            
                            particles = algo.fireflies.copy()
                            best_pos = algo.best_firefly.copy()
                            best_score = algo.best_score
                        
                        elif algorithm == "Cuckoo":
                            for i in range(algo.n_nests):
                                idx = np.random.randint(0, algo.n_nests)
                                
                                sigma_u = (np.math.gamma(1 + algo.beta) * np.sin(np.pi * algo.beta / 2) / \
                                          (np.math.gamma((1 + algo.beta) / 2) * algo.beta * 2 ** ((algo.beta - 1) / 2))) ** (1 / algo.beta)
                                u = np.random.normal(0, sigma_u, algo.dim)
                                v = np.random.normal(0, 1, algo.dim)
                                step = u / np.abs(v) ** (1 / algo.beta)
                                
                                new_nest = algo.nests[idx] + 0.01 * step
                                new_nest = np.clip(new_nest, algo.bounds[:, 0], algo.bounds[:, 1])
                                new_fitness = func(new_nest)
                                
                                j = np.random.randint(0, algo.n_nests)
                                if new_fitness < algo.fitness[j]:
                                    algo.nests[j] = new_nest
                                    algo.fitness[j] = new_fitness
                            
                            n_abandon = int(algo.pa * algo.n_nests)
                            worst_indices = np.argsort(algo.fitness)[-n_abandon:]
                            
                            for idx in worst_indices:
                                K = np.random.rand()
                                i1, i2 = np.random.choice(algo.n_nests, 2, replace=False)
                                algo.nests[idx] = algo.nests[idx] + K * (algo.nests[i1] - algo.nests[i2])
                                algo.nests[idx] = np.clip(algo.nests[idx], algo.bounds[:, 0], algo.bounds[:, 1])
                            
                            algo.fitness = np.array([func(nest) for nest in algo.nests])
                            best_idx = np.argmin(algo.fitness)
                            if algo.fitness[best_idx] < algo.best_score:
                                algo.best_score = algo.fitness[best_idx]
                                algo.best_nest = algo.nests[best_idx].copy()
                            
                            particles = algo.nests.copy()
                            best_pos = algo.best_nest.copy()
                            best_score = algo.best_score
                            
                    elif algorithm == "Hill Climbing":
                        # Manual step-by-step for animation
                        neighbor = algo.current_solution + np.random.uniform(-algo.step_size, algo.step_size, algo.dim)
                        neighbor = np.clip(neighbor, algo.bounds[:, 0], algo.bounds[:, 1])
                        neighbor_score = func(neighbor)
                        
                        if neighbor_score < algo.best_score:
                            algo.current_solution = neighbor.copy()
                            algo.best_score = neighbor_score
                            algo.best_solution = neighbor.copy()
                        
                        particles = np.array([algo.current_solution])
                        best_pos = algo.best_solution.copy()
                        best_score = algo.best_score
                    
                    elif algorithm == "Simulated Annealing":
                        # Manual step-by-step for animation
                        neighbor = algo.current_solution + np.random.uniform(-algo.step_size, algo.step_size, algo.dim)
                        neighbor = np.clip(neighbor, algo.bounds[:, 0], algo.bounds[:, 1])
                        neighbor_score = func(neighbor)
                        
                        # Acceptance criterion
                        delta = neighbor_score - algo.current_score
                        if delta < 0 or np.random.rand() < np.exp(-delta / algo.temperature):
                            algo.current_solution = neighbor.copy()
                            algo.current_score = neighbor_score
                            
                            if neighbor_score < algo.best_score:
                                algo.best_score = neighbor_score
                                algo.best_solution = neighbor.copy()
                        
                        # Cool down
                        algo.temperature *= algo.alpha
                        
                        particles = np.array([algo.current_solution])
                        best_pos = algo.best_solution.copy()
                        best_score = algo.best_score
                        
                    elif algorithm == "Genetic Algorithm":
                        # Manual step-by-step for animation
                        # Evaluate fitness
                        fitness = np.array([func(ind) for ind in algo.population])
                        
                        # Selection (tournament)
                        new_population = []
                        for _ in range(algo.population_size):
                            tournament = np.random.choice(algo.population_size, 3, replace=False)
                            winner = algo.population[tournament[np.argmin(fitness[tournament])]]
                            new_population.append(winner.copy())
                        
                        # Crossover
                        for i in range(0, algo.population_size - 1, 2):
                            if np.random.rand() < algo.crossover_rate:
                                alpha = np.random.rand(algo.dim)
                                new_population[i] = alpha * new_population[i] + (1 - alpha) * new_population[i + 1]
                                new_population[i + 1] = alpha * new_population[i + 1] + (1 - alpha) * new_population[i]
                        
                        # Mutation
                        for i in range(algo.population_size):
                            if np.random.rand() < algo.mutation_rate:
                                mutation = np.random.uniform(-1, 1, algo.dim)
                                new_population[i] = new_population[i] + mutation
                                new_population[i] = np.clip(new_population[i], algo.bounds[:, 0], algo.bounds[:, 1])
                        
                        # Elitism
                        elite_indices = np.argsort(fitness)[:algo.elite_size]
                        for idx, elite_idx in enumerate(elite_indices):
                            new_population[idx] = algo.population[elite_idx].copy()
                        
                        algo.population = np.array(new_population)
                        
                        # Update best
                        current_fitness = np.array([func(ind) for ind in algo.population])
                        best_idx = np.argmin(current_fitness)
                        if current_fitness[best_idx] < algo.best_score:
                            algo.best_score = current_fitness[best_idx]
                            algo.best_solution = algo.population[best_idx].copy()
                        
                        particles = algo.population[:20] if len(algo.population) > 20 else algo.population
                        best_pos = algo.best_solution.copy()
                        best_score = algo.best_score
                    
                    best_scores_history.append(best_score)
                    
                    iteration_metric.metric("Iteration", f"{iteration + 1}/{max_iter}")
                    score_metric.metric("Best Score", f"{best_score:.8f}")
                    gap = abs(best_score - func.global_optimum)
                    gap_metric.metric("Gap to Optimum", f"{gap:.8f}")
                    
                    if dimensions == 2:
                        fig = create_animated_3d_plot(func, particles, best_pos, best_score,
                                                     iteration + 1, algorithm, function_name)
                        plot_placeholder.pyplot(fig)
                        plt.close(fig)
                    
                    if len(best_scores_history) > 1:
                        fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
                        ax_conv.plot(best_scores_history, 'b-', linewidth=2.5, label='Best Score')
                        ax_conv.axhline(y=func.global_optimum, color='r', linestyle='--',
                                       linewidth=2, label='Global Optimum')
                        ax_conv.set_xlabel('Iteration', fontsize=11, fontweight='bold')
                        ax_conv.set_ylabel('Best Score', fontsize=11, fontweight='bold')
                        ax_conv.set_yscale('log')
                        ax_conv.legend()
                        ax_conv.grid(True, alpha=0.3)
                        plt.tight_layout()
                        convergence_placeholder.pyplot(fig_conv)
                        plt.close(fig_conv)
                    
                    progress_bar.progress((iteration + 1) / max_iter)
                    
                    time.sleep(animation_speed)
                
                st.markdown("---")
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("## 🎉 Kết Quả Cuối Cùng")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Score", f"{best_score:.8f}")
                with col2:
                    st.metric("Global Optimum", f"{func.global_optimum:.8f}")
                with col3:
                    gap = abs(best_score - func.global_optimum)
                    st.metric("Gap", f"{gap:.8f}")
                with col4:
                    if gap < 0.01:
                        st.metric("Status", "✅ SUCCESS")
                    else:
                        st.metric("Status", "⚠️ SUBOPTIMAL")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                with st.expander("🎯 Chi tiết Best Position"):
                    st.write(f"**Best Position:** {best_pos}")
                    st.write(f"**Dimensions:** {dimensions}")
                    st.write(f"**Function:** {function_name}")
                    st.write(f"**Algorithm:** {algorithm}")
            
            else:
                st.info(f"Đang chạy {algorithm} trên {function_name.upper()} với {dimensions} chiều...")
                
                with st.spinner("Optimizing..."):
                    best_pos, best_score = algo.optimize(func, verbose=False)
                    history = algo.get_history()
                
                st.success("✅ Hoàn thành!")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(9, 5))
                    ax.plot(history['best_scores'], 'b-', linewidth=2.5, label='Best Score')
                    ax.axhline(y=func.global_optimum, color='r', linestyle='--',
                              linewidth=2, label='Global Optimum')
                    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Best Score', fontsize=12, fontweight='bold')
                    ax.set_yscale('log')
                    ax.set_title(f'Convergence: {algorithm} on {function_name}',
                                fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("### 📊 Results")
                    st.metric("Best Score", f"{best_score:.8f}")
                    st.metric("Global Optimum", f"{func.global_optimum:.8f}")
                    gap = abs(best_score - func.global_optimum)
                    st.metric("Gap", f"{gap:.8f}")
                    
                    st.markdown("### 📈 Statistics")
                    st.write(f"**Iterations:** {len(history['best_scores'])}")
                    st.write(f"**Final Score:** {history['best_scores'][-1]:.8f}")
                    st.write(f"**Improvement:** {history['best_scores'][0] - history['best_scores'][-1]:.8f}")
                
                with st.expander("🎯 Best Position"):
                    st.write(best_pos)
        
        elif problem_type == "Discrete Optimization (TSP)":
            tsp = TSP(n_cities=n_cities, seed=42)
            
            st.success(f"✅ Đang chạy {algorithm} trên TSP với {n_cities} thành phố...")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                plot_placeholder = st.empty()
            
            with col2:
                st.markdown("### 📊 Metrics")
                iteration_metric = st.empty()
                distance_metric = st.empty()
                improvement_metric = st.empty()
                
                st.markdown("### 📈 Progress")
                convergence_placeholder = st.empty()
            
            progress_bar = st.progress(0)
            
            tour_history = []
            distance_history = []
            
            if algorithm == "Genetic Algorithm":
                population_size_tsp = 50
                
                population = [np.random.permutation(n_cities).tolist() 
                             for _ in range(population_size_tsp)]
                
                best_tour = min(population, key=lambda t: tsp.evaluate(t))
                best_distance = tsp.evaluate(best_tour)
                
                if 'max_iter' not in locals():
                    max_iter = 100
                
                for iteration in range(max_iter):
                    new_population = []
                    for _ in range(population_size_tsp):
                        tournament = np.random.choice(population_size_tsp, 3, replace=False)
                        winner = min([population[i] for i in tournament],
                                   key=lambda t: tsp.evaluate(t))
                        new_population.append(winner[:])
                    
                    for i in range(0, population_size_tsp-1, 2):
                        if np.random.rand() < crossover_rate:
                            parent1, parent2 = new_population[i], new_population[i+1]
                            start, end = sorted(np.random.choice(n_cities, 2, replace=False))
                            
                            child1 = [-1] * n_cities
                            child1[start:end] = parent1[start:end]
                            remaining = [x for x in parent2 if x not in child1]
                            child1 = [remaining.pop(0) if x == -1 else x for x in child1]
                            
                            new_population[i] = child1
                    
                    last_mutated_city = None
                    for i in range(population_size_tsp):
                        if np.random.rand() < mutation_rate:
                            idx1, idx2 = np.random.choice(n_cities, 2, replace=False)
                            new_population[i][idx1], new_population[i][idx2] = \
                                new_population[i][idx2], new_population[i][idx1]
                            last_mutated_city = new_population[i][idx1]
                    
                    population = new_population
                    
                    current_best = min(population, key=lambda t: tsp.evaluate(t))
                    current_distance = tsp.evaluate(current_best)
                    
                    if current_distance < best_distance:
                        best_distance = current_distance
                        best_tour = current_best[:]
                    
                    tour_history.append(best_tour[:])
                    distance_history.append(best_distance)
                    
                    iteration_metric.metric("Iteration", f"{iteration + 1}/{max_iter}")
                    distance_metric.metric("Best Distance", f"{best_distance:.2f}")
                    
                    improvement = distance_history[0] - best_distance if len(distance_history) > 0 else 0
                    improvement_metric.metric("Improvement", f"{improvement:.2f}")
                    
                    viz_frequency = max(1, max_iter // 40)
                    if iteration % viz_frequency == 0 or iteration == max_iter - 1 or iteration < 5:
                        population_sample = population[:10] if len(population) > 10 else population
                        
                        fig = TSPVisualizer.create_tsp_animation_frame(
                            tsp, 
                            current_tour=current_best,
                            best_tour=best_tour,
                            iteration=iteration + 1,
                            algorithm_name=algorithm,
                            population_tours=population_sample,
                            current_city=last_mutated_city,
                            tour_history=distance_history
                        )
                        plot_placeholder.pyplot(fig)
                        plt.close(fig)
                    
                    if len(distance_history) > 1:
                        fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
                        ax_conv.plot(distance_history, 'b-', linewidth=2.5, label='Best Distance')
                        ax_conv.set_xlabel('Iteration', fontsize=11, fontweight='bold')
                        ax_conv.set_ylabel('Tour Distance', fontsize=11, fontweight='bold')
                        ax_conv.legend()
                        ax_conv.grid(True, alpha=0.3)
                        ax_conv.fill_between(range(len(distance_history)), distance_history, alpha=0.3)
                        plt.tight_layout()
                        convergence_placeholder.pyplot(fig_conv)
                        plt.close(fig_conv)
                    
                    progress_bar.progress((iteration + 1) / max_iter)
                    time.sleep(animation_speed)
            
            else:
                if 'max_iter' not in locals():
                    max_iter = 100
                
                best_tour, _ = TSPSolver.nearest_neighbor(tsp)
                best_distance = tsp.evaluate(best_tour)
                distance_history = [best_distance]
                
                if algorithm == "Simulated Annealing":
                    temperature = initial_temp if 'initial_temp' in locals() else 100.0
                    alpha_cooling = cooling_rate if 'cooling_rate' in locals() else 0.95
                
                no_improvement_count = 0
                max_no_improvement = 10
                
                for iteration in range(max_iter):
                    improved = False
                    
                    swap_city_i = None
                    swap_city_j = None
                    
                    n_swap_tries = 8 if algorithm == "Hill Climbing" else 5
                    
                    for try_idx in range(n_swap_tries):
                        i = np.random.randint(0, n_cities - 2)
                        j = np.random.randint(i + 2, min(n_cities, i + 10))
                        
                        current_swap_cities = [best_tour[i], best_tour[j-1] if j > 0 else best_tour[-1]]
                        
                        new_tour = best_tour[:]
                        new_tour[i:j] = list(reversed(new_tour[i:j]))
                        new_distance = tsp.evaluate(new_tour)
                        
                        show_this_try = (try_idx % 2 == 0) or (new_distance < best_distance)
                        
                        if algorithm in ["Hill Climbing", "Simulated Annealing"] and show_this_try:
                            fig = TSPVisualizer.create_tsp_animation_frame(
                                tsp, 
                                current_tour=new_tour,
                                best_tour=best_tour,
                                iteration=iteration + 1,
                                algorithm_name=f"{algorithm} - Swap City {current_swap_cities[0]} ↔ {current_swap_cities[1]}",
                                current_city=current_swap_cities[0],
                                tour_history=distance_history if len(distance_history) > 0 else [best_distance]
                            )
                            plot_placeholder.pyplot(fig)
                            plt.close(fig)
                            time.sleep(animation_speed * 0.4)
                        
                        if algorithm == "Hill Climbing":
                            if new_distance < best_distance:
                                best_tour = new_tour
                                best_distance = new_distance
                                improved = True
                                swap_city_i, swap_city_j = current_swap_cities[0], current_swap_cities[1]
                                no_improvement_count = 0
                                break
                        
                        else:  # Simulated Annealing
                            delta = new_distance - best_distance
                            if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                                best_tour = new_tour
                                best_distance = new_distance
                                improved = True
                                swap_city_i, swap_city_j = current_swap_cities[0], current_swap_cities[1]
                                break
                    
                    if algorithm == "Hill Climbing" and not improved:
                        no_improvement_count += 1
                    
                    if algorithm == "Simulated Annealing":
                        temperature *= alpha_cooling
                    
                    distance_history.append(best_distance)
                    
                    iteration_metric.metric("Iteration", f"{iteration + 1}/{max_iter}")
                    distance_metric.metric("Best Distance", f"{best_distance:.2f}")
                    
                    improvement = distance_history[0] - best_distance
                    improvement_metric.metric("Improvement", f"{improvement:.2f}")
                    
                    fig = TSPVisualizer.create_tsp_animation_frame(
                        tsp, best_tour, best_tour,
                        iteration=iteration + 1,
                        algorithm_name=algorithm,
                        current_city=swap_city_i if swap_city_i is not None else None,
                        tour_history=distance_history
                    )
                    plot_placeholder.pyplot(fig)
                    plt.close(fig)
                    
                    if len(distance_history) > 1:
                        fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
                        ax_conv.plot(distance_history, 'b-', linewidth=2.5, label='Best Distance')
                        ax_conv.set_xlabel('Iteration', fontsize=11, fontweight='bold')
                        ax_conv.set_ylabel('Tour Distance', fontsize=11, fontweight='bold')
                        ax_conv.legend()
                        ax_conv.grid(True, alpha=0.3)
                        ax_conv.fill_between(range(len(distance_history)), distance_history, alpha=0.3)
                        plt.tight_layout()
                        convergence_placeholder.pyplot(fig_conv)
                        plt.close(fig_conv)
                    
                    progress_bar.progress((iteration + 1) / max_iter)
                    time.sleep(animation_speed)
                    
                    if algorithm == "Hill Climbing" and no_improvement_count >= max_no_improvement:
                        st.info(f"🛑 Hill Climbing converged after {iteration + 1} iterations (no improvement for {max_no_improvement} iterations)")
                        break
            
            st.markdown("---")
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("## 🎉 TSP Solved!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tour Distance", f"{best_distance:.2f}")
            with col2:
                st.metric("Cities", n_cities)
            with col3:
                st.metric("Algorithm", algorithm)
            with col4:
                if distance_history:
                    improvement_pct = ((distance_history[0] - best_distance) / distance_history[0]) * 100
                    st.metric("Improvement", f"{improvement_pct:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # === HIỂN THỊ BEST SOLUTION RÕ RÀNG ===
            st.markdown("### 🏆 Best Solution Found:")
            
            col_viz, col_info = st.columns([2, 1])
            
            with col_viz:
                fig_final = TSPVisualizer.create_tsp_animation_frame(
                    tsp, 
                    current_tour=best_tour,
                    best_tour=best_tour,
                    iteration=len(distance_history) if distance_history else max_iter,
                    algorithm_name=f"✅ BEST TOUR - {algorithm}",
                    tour_history=distance_history
                )
                st.pyplot(fig_final)
                plt.close(fig_final)
            
            with col_info:
                st.markdown("#### 📋 Tour Details")
                st.write(f"**🎯 Best Distance:** `{best_distance:.2f}`")
                
                if distance_history:
                    st.write(f"**📍 Initial Distance:** `{distance_history[0]:.2f}`")
                    st.write(f"**📈 Improvement:** `{distance_history[0] - best_distance:.2f}`")
                    st.write(f"**📊 Iterations:** `{len(distance_history)}`")
                
                st.write(f"**🏙️ Cities:** `{n_cities}`")
                
                st.markdown("---")
                st.markdown("**🗺️ Tour Sequence:**")
                
                tour_str = ""
                for i, city in enumerate(best_tour):
                    if i % 5 == 0 and i > 0:
                        tour_str += "\n"
                    tour_str += f"`{city}` → "
                tour_str += f"`{best_tour[0]}`"
                
                st.markdown(tour_str)
            
            with st.expander("📊 Detailed Statistics"):
                st.write(f"**Full Tour Sequence:** {' → '.join(map(str, best_tour))} → {best_tour[0]}")
                st.write(f"**Total Distance:** {best_distance:.2f}")
                st.write(f"**Cities Visited:** {len(best_tour)}")
                if distance_history:
                    st.write(f"**Initial Distance:** {distance_history[0]:.2f}")
                    st.write(f"**Final Distance:** {best_distance:.2f}")
                    st.write(f"**Total Improvement:** {distance_history[0] - best_distance:.2f}")
                    st.write(f"**Improvement Percentage:** {improvement_pct:.2f}%")
                    st.write(f"**Total Iterations:** {len(distance_history)}")
                    st.write(f"**Average Distance per Iteration:** {np.mean(distance_history):.2f}")
        
        elif problem_type == "Discrete Optimization (Knapsack)":
            knapsack = Knapsack(n_items=n_items, seed=42)
            
            st.success(f"✅ Đang chạy {algorithm} trên Knapsack với {n_items} items...")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                plot_placeholder = st.empty()
            
            with col2:
                st.markdown("### 📊 Metrics")
                iteration_metric = st.empty()
                value_metric = st.empty()
                weight_metric = st.empty()
                
                st.markdown("### 📈 Progress")
                convergence_placeholder = st.empty()
            
            progress_bar = st.progress(0)
            
            value_history = []
            
            if 'max_iter' not in locals():
                max_iter = 100
            
            if algorithm == "Genetic Algorithm":
                population_size_kp = 50
                
                population = [np.random.randint(0, 2, n_items) for _ in range(population_size_kp)]
                
                for i in range(population_size_kp):
                    while not knapsack.is_valid(population[i]):
                        # Remove random item
                        ones = np.where(population[i] == 1)[0]
                        if len(ones) > 0:
                            population[i][np.random.choice(ones)] = 0
                        else:
                            break
                
                best_solution = max(population, key=lambda s: knapsack.get_value(s) if knapsack.is_valid(s) else 0)
                best_value = knapsack.get_value(best_solution)
                
                for iteration in range(max_iter):
                    fitness = [knapsack.get_value(s) if knapsack.is_valid(s) else 0 for s in population]
                    new_population = []
                    
                    for _ in range(population_size_kp):
                        tournament = np.random.choice(population_size_kp, 3, replace=False)
                        winner = tournament[np.argmax([fitness[i] for i in tournament])]
                        new_population.append(population[winner].copy())
                    
                    for i in range(0, population_size_kp-1, 2):
                        if np.random.rand() < crossover_rate:
                            point = np.random.randint(1, n_items-1)
                            new_population[i][point:], new_population[i+1][point:] = \
                                new_population[i+1][point:].copy(), new_population[i][point:].copy()
                    
                    for i in range(population_size_kp):
                        if np.random.rand() < mutation_rate:
                            idx = np.random.randint(0, n_items)
                            new_population[i][idx] = 1 - new_population[i][idx]
                    
                    for i in range(population_size_kp):
                        while not knapsack.is_valid(new_population[i]):
                            ones = np.where(new_population[i] == 1)[0]
                            if len(ones) > 0:
                                new_population[i][np.random.choice(ones)] = 0
                            else:
                                break
                    
                    population = new_population
                    
                    fitness = [knapsack.get_value(s) if knapsack.is_valid(s) else 0 for s in population]
                    current_best_idx = np.argmax(fitness)
                    current_value = fitness[current_best_idx]
                    
                    if current_value > best_value:
                        best_value = current_value
                        best_solution = population[current_best_idx].copy()
                    
                    value_history.append(best_value)
                    
                    # Update metrics
                    iteration_metric.metric("Iteration", f"{iteration + 1}/{max_iter}")
                    value_metric.metric("Best Value", f"{best_value:.0f}")
                    weight_metric.metric("Weight", f"{knapsack.get_weight(best_solution):.0f}/{knapsack.capacity:.0f}")
                    
                    if iteration % max(1, max_iter // 50) == 0 or iteration == max_iter - 1:
                        fig = KnapsackVisualizer.create_knapsack_animation_frame(
                            knapsack,
                            solution=best_solution,
                            best_solution=best_solution,
                            iteration=iteration + 1,
                            algorithm_name=algorithm,
                            population_solutions=population[:20]
                        )
                        plot_placeholder.pyplot(fig)
                        plt.close(fig)
                    
                    # Update convergence plot
                    if len(value_history) > 1:
                        fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
                        ax_conv.plot(value_history, 'g-', linewidth=2.5, label='Best Value')
                        ax_conv.set_xlabel('Iteration', fontsize=11, fontweight='bold')
                        ax_conv.set_ylabel('Total Value', fontsize=11, fontweight='bold')
                        ax_conv.legend()
                        ax_conv.grid(True, alpha=0.3)
                        ax_conv.fill_between(range(len(value_history)), value_history, alpha=0.3, color='green')
                        plt.tight_layout()
                        convergence_placeholder.pyplot(fig_conv)
                        plt.close(fig_conv)
                    
                    progress_bar.progress((iteration + 1) / max_iter)
                    time.sleep(animation_speed)
            
            elif algorithm in ["Hill Climbing", "Simulated Annealing"]:
                best_solution, _ = KnapsackSolver.greedy_ratio(knapsack)
                best_value = knapsack.get_value(best_solution)
                value_history = [best_value]
                
                if algorithm == "Simulated Annealing":
                    temperature = initial_temp if 'initial_temp' in locals() else 100.0
                    alpha_cooling = cooling_rate if 'cooling_rate' in locals() else 0.95
                
                for iteration in range(max_iter):
                    neighbor = best_solution.copy()
                    flip_idx = np.random.randint(0, n_items)
                    neighbor[flip_idx] = 1 - neighbor[flip_idx]
                    
                    while not knapsack.is_valid(neighbor):
                        ones = np.where(neighbor == 1)[0]
                        if len(ones) > 0:
                            neighbor[np.random.choice(ones)] = 0
                        else:
                            break
                    
                    neighbor_value = knapsack.get_value(neighbor)
                    
                    if algorithm == "Hill Climbing":
                        if neighbor_value > best_value:
                            best_solution = neighbor
                            best_value = neighbor_value
                    
                    else:
                        delta = best_value - neighbor_value
                        if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                            best_solution = neighbor
                            best_value = neighbor_value
                        temperature *= alpha_cooling
                    
                    value_history.append(best_value)
                    
                    # Update metrics
                    iteration_metric.metric("Iteration", f"{iteration + 1}/{max_iter}")
                    value_metric.metric("Best Value", f"{best_value:.0f}")
                    weight_metric.metric("Weight", f"{knapsack.get_weight(best_solution):.0f}/{knapsack.capacity:.0f}")
                    
                    if iteration % max(1, max_iter // 20) == 0 or iteration == max_iter - 1:
                        fig = KnapsackVisualizer.create_knapsack_animation_frame(
                            knapsack,
                            solution=best_solution,
                            best_solution=best_solution,
                            iteration=iteration + 1,
                            algorithm_name=algorithm
                        )
                        plot_placeholder.pyplot(fig)
                        plt.close(fig)
                    
                    # Update convergence plot
                    if len(value_history) > 1:
                        fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
                        ax_conv.plot(value_history, 'g-', linewidth=2.5, label='Best Value')
                        ax_conv.set_xlabel('Iteration', fontsize=11, fontweight='bold')
                        ax_conv.set_ylabel('Total Value', fontsize=11, fontweight='bold')
                        ax_conv.legend()
                        ax_conv.grid(True, alpha=0.3)
                        ax_conv.fill_between(range(len(value_history)), value_history, alpha=0.3, color='green')
                        plt.tight_layout()
                        convergence_placeholder.pyplot(fig_conv)
                        plt.close(fig_conv)
                    
                    progress_bar.progress((iteration + 1) / max_iter)
                    time.sleep(animation_speed)
            
            else:
                with st.spinner(f"Running {algorithm}..."):
                    best_solution, best_value = KnapsackSolver.greedy_ratio(knapsack)
                    value_history = [best_value]
                
                fig = KnapsackVisualizer.create_knapsack_animation_frame(
                    knapsack, best_solution, best_solution,
                    iteration=1,
                    algorithm_name=algorithm
                )
                plot_placeholder.pyplot(fig)
                plt.close(fig)
            
            st.markdown("---")
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("## 🎉 Knapsack Solved!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Value", f"{best_value:.0f}")
            with col2:
                st.metric("Weight Used", f"{knapsack.get_weight(best_solution):.0f}/{knapsack.capacity:.0f}")
            with col3:
                st.metric("Items Selected", f"{np.sum(best_solution)}/{n_items}")
            with col4:
                valid = "✅ Valid" if knapsack.is_valid(best_solution) else "❌ Invalid"
                st.metric("Status", valid)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # === HIỂN THỊ BEST SOLUTION RÕ RÀNG ===
            st.markdown("### 🏆 Best Solution Found:")
            
            col_viz, col_info = st.columns([2, 1])
            
            with col_viz:
                # Visualization FINAL của best solution
                fig_final = KnapsackVisualizer.create_knapsack_animation_frame(
                    knapsack,
                    solution=best_solution,
                    best_solution=best_solution,
                    iteration=len(value_history) if 'value_history' in locals() and value_history else 1,
                    algorithm_name=f"✅ BEST SOLUTION - {algorithm}"
                )
                st.pyplot(fig_final)
                plt.close(fig_final)
            
            with col_info:
                st.markdown("#### 📋 Solution Details")
                st.write(f"**💰 Total Value:** `{best_value:.0f}`")
                st.write(f"**⚖️ Total Weight:** `{knapsack.get_weight(best_solution):.0f}` / `{knapsack.capacity:.0f}`")
                st.write(f"**📦 Items Selected:** `{np.sum(best_solution)}` / `{n_items}`")
                utilization = (knapsack.get_weight(best_solution)/knapsack.capacity)*100
                st.write(f"**📊 Utilization:** `{utilization:.1f}%`")
                
                is_valid = knapsack.is_valid(best_solution)
                if is_valid:
                    st.success("✅ Valid solution!")
                else:
                    st.error("❌ Invalid solution (over capacity)")
                
                st.markdown("---")
                st.markdown("**📦 Selected Items:**")
                
                selected_items = np.where(best_solution == 1)[0]
                items_str = ""
                for i, item_id in enumerate(selected_items):
                    if i > 0 and i % 10 == 0:
                        items_str += "\n"
                    items_str += f"`{item_id}` "
                
                st.markdown(items_str if items_str else "*No items selected*")
            
            with st.expander("📊 Detailed Statistics"):
                selected_items = np.where(best_solution == 1)[0]
                st.write(f"**Selected Items (IDs):** {', '.join(map(str, selected_items))}")
                st.write(f"**Number of Items Selected:** {len(selected_items)}")
                st.write(f"**Total Value:** {best_value:.0f}")
                st.write(f"**Total Weight:** {knapsack.get_weight(best_solution):.0f}")
                st.write(f"**Capacity:** {knapsack.capacity:.0f}")
                st.write(f"**Remaining Capacity:** {knapsack.capacity - knapsack.get_weight(best_solution):.0f}")
                st.write(f"**Capacity Utilization:** {utilization:.2f}%")
                st.write(f"**Valid Solution:** {'Yes ✅' if is_valid else 'No ❌'}")
                
                if 'value_history' in locals() and value_history and len(value_history) > 1:
                    st.write(f"**Initial Value:** {value_history[0]:.0f}")
                    st.write(f"**Final Value:** {best_value:.0f}")
                    st.write(f"**Total Improvement:** {best_value - value_history[0]:.0f}")
                    st.write(f"**Iterations:** {len(value_history)}")
        
        elif problem_type == "Discrete Optimization (Graph Coloring)":
            graph_coloring = GraphColoring(n_vertices=n_nodes, edge_probability=edge_prob, seed=42)
            
            st.success(f"✅ Đang chạy {algorithm} trên Graph Coloring với {n_nodes} nodes...")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                plot_placeholder = st.empty()
            
            with col2:
                st.markdown("### 📊 Metrics")
                iteration_metric = st.empty()
                colors_metric = st.empty()
                conflicts_metric = st.empty()
                
                st.markdown("### 📈 Progress")
                convergence_placeholder = st.empty()
            
            progress_bar = st.progress(0)
            
            if 'max_iter' not in locals():
                max_iter = 100
            
            st.info(f"🎨 Phase 1: Tô màu từng node với {algorithm}...")
            
            n = n_nodes
            coloring = np.full(n, -1, dtype=int)
            
            for step in range(n):
                if algorithm == "Genetic Algorithm":
                    uncolored = [i for i in range(n) if coloring[i] == -1]
                    if not uncolored:
                        break
                    vertex = np.random.choice(uncolored)
                else:
                    uncolored = [i for i in range(n) if coloring[i] == -1]
                    if not uncolored:
                        break
                    vertex = uncolored[0]
                
                adjacent_colors = set()
                for neighbor in range(n):
                    if graph_coloring.adj_matrix[vertex, neighbor] == 1 and coloring[neighbor] != -1:
                        adjacent_colors.add(coloring[neighbor])
                
                if algorithm == "Genetic Algorithm" and np.random.rand() < 0.3:
                    max_color = int(np.max(coloring[coloring >= 0])) if len(coloring[coloring >= 0]) > 0 else 0
                    color = np.random.randint(0, max_color + 2)
                else:
                    color = 0
                    while color in adjacent_colors:
                        color += 1
                
                coloring[vertex] = color
                
                iteration_metric.metric("Phase", "1: Constructive")
                iteration_metric.metric("Step", f"{step + 1}/{n}")
                colors_metric.metric("Colors Used", len(np.unique(coloring[coloring >= 0])))
                conflicts_metric.metric("Nodes Colored", f"{step + 1}/{n}")
                
                fig = GraphColoringVisualizer.create_graph_coloring_frame(
                    graph_coloring,
                    coloring=coloring,
                    best_coloring=coloring,
                    iteration=step + 1,
                    algorithm_name=f"{algorithm} - Constructive: Node {vertex}",
                    current_node=vertex
                )
                plot_placeholder.pyplot(fig)
                plt.close(fig)
                
                progress_bar.progress((step + 1) / (n + max_iter))
                time.sleep(animation_speed)
            
            best_coloring = coloring.copy()
            best_conflicts = graph_coloring.count_conflicts(best_coloring)
            best_n_colors = graph_coloring.count_colors(best_coloring)
            
            conflicts_history = [best_conflicts]
            colors_history = [best_n_colors]
            
            st.info(f"✅ Phase 1 Done! Colors: {best_n_colors}, Conflicts: {best_conflicts}")
            st.info(f"🔧 Phase 2: Optimization với {algorithm}...")
            
            if algorithm == "Simulated Annealing":
                temperature = initial_temp if 'initial_temp' in locals() else 100.0
                alpha_cooling = cooling_rate if 'cooling_rate' in locals() else 0.95
            
            for iteration in range(max_iter):
                current_node = None
                
                if algorithm == "Genetic Algorithm":
                    neighbor = coloring.copy()
                    current_node = np.random.randint(0, n_nodes)
                    max_color = int(np.max(neighbor))
                    new_color = np.random.randint(0, max_color + 2)
                    neighbor[current_node] = new_color
                else:
                    neighbor = coloring.copy()
                    if np.random.rand() < 0.5:
                        node1, node2 = np.random.choice(n_nodes, 2, replace=False)
                        neighbor[node1], neighbor[node2] = neighbor[node2], neighbor[node1]
                        current_node = node1
                    else:
                        current_node = np.random.randint(0, n_nodes)
                        max_color = int(np.max(neighbor))
                        neighbor[current_node] = np.random.randint(0, max_color + 1)
                    
                    neighbor_conflicts = graph_coloring.count_conflicts(neighbor)
                    neighbor_n_colors = graph_coloring.count_colors(neighbor)
                    
                    current_fitness = (best_conflicts * 1000) + best_n_colors
                    neighbor_fitness = (neighbor_conflicts * 1000) + neighbor_n_colors
                    
                    accept = False
                    if algorithm == "Hill Climbing":
                        if neighbor_fitness < current_fitness:
                            accept = True
                    elif algorithm == "Simulated Annealing":
                        delta = neighbor_fitness - current_fitness
                        if delta < 0 or np.random.rand() < np.exp(-delta / (temperature + 1e-10)):
                            accept = True
                    else:
                        if neighbor_conflicts <= best_conflicts:
                            accept = True
                    
                    if accept:
                        coloring = neighbor.copy()
                        current_conflicts = neighbor_conflicts
                        current_n_colors = neighbor_n_colors
                        
                        if neighbor_fitness < (best_conflicts * 1000 + best_n_colors):
                            best_coloring = neighbor.copy()
                            best_conflicts = neighbor_conflicts
                            best_n_colors = neighbor_n_colors
                    else:
                        current_conflicts = graph_coloring.count_conflicts(coloring)
                        current_n_colors = graph_coloring.count_colors(coloring)
                    
                    if algorithm == "Simulated Annealing":
                        temperature *= alpha_cooling
                    
                    conflicts_history.append(best_conflicts)
                    colors_history.append(best_n_colors)
                    
                    iteration_metric.metric("Phase", "2: Optimization")
                    iteration_metric.metric("Iteration", f"{iteration + 1}/{max_iter}")
                    colors_metric.metric("Colors Used", best_n_colors)
                    conflicts_metric.metric("Conflicts", best_conflicts)
                    
                    update_frequency = viz_update_freq if 'viz_update_freq' in locals() else max(1, max_iter // 50)
                    if iteration % update_frequency == 0 or iteration == max_iter - 1:
                        fig = GraphColoringVisualizer.create_graph_coloring_frame(
                            graph_coloring,
                            coloring=best_coloring,
                            best_coloring=best_coloring,
                            iteration=n + iteration + 1,
                            algorithm_name=f"{algorithm} - Optimization",
                            current_node=current_node
                        )
                        plot_placeholder.pyplot(fig)
                        plt.close(fig)
                    
                    if len(conflicts_history) > 1 and iteration % max(1, update_frequency * 2) == 0:
                        fig_conv, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
                        
                        ax1.plot(conflicts_history, 'r-', linewidth=2.5, label='Conflicts', marker='o', markersize=3)
                        ax1.set_ylabel('Conflicts', fontsize=10, fontweight='bold')
                        ax1.set_title('Conflicts Over Time', fontsize=11, fontweight='bold')
                        ax1.legend(fontsize=9)
                        ax1.grid(True, alpha=0.3)
                        ax1.fill_between(range(len(conflicts_history)), conflicts_history, alpha=0.3, color='red')
                        
                        ax2.plot(colors_history, 'b-', linewidth=2.5, label='Colors Used', marker='s', markersize=3)
                        ax2.set_xlabel('Iteration', fontsize=10, fontweight='bold')
                        ax2.set_ylabel('Colors', fontsize=10, fontweight='bold')
                        ax2.set_title('Colors Used Over Time', fontsize=11, fontweight='bold')
                        ax2.legend(fontsize=9)
                        ax2.grid(True, alpha=0.3)
                        ax2.fill_between(range(len(colors_history)), colors_history, alpha=0.3, color='blue')
                        
                        plt.tight_layout()
                        convergence_placeholder.pyplot(fig_conv)
                        plt.close(fig_conv)
                    
                    progress_bar.progress((n + iteration + 1) / (n + max_iter))
                    time.sleep(animation_speed)
                
                coloring = best_coloring
                n_colors = best_n_colors
                n_conflicts = best_conflicts
            
            # Results
            st.markdown("---")
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("## 🎉 Graph Coloring Solved!")
            
            n_conflicts = graph_coloring.count_conflicts(coloring)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Colors Used", f"{n_colors}")
            with col2:
                st.metric("Nodes", n_nodes)
            with col3:
                st.metric("Conflicts", f"{n_conflicts}")
            with col4:
                status = "✅ Valid" if n_conflicts == 0 else "❌ Invalid"
                st.metric("Status", status)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # === HIỂN THỊ BEST SOLUTION RÕ RÀNG ===
            st.markdown("### 🏆 Best Solution Found:")
            
            col_viz, col_info = st.columns([2, 1])
            
            with col_viz:
                # Visualization FINAL của best coloring
                fig_final = GraphColoringVisualizer.create_graph_coloring_frame(
                    graph_coloring,
                    coloring=coloring,
                    best_coloring=coloring,
                    iteration=len(colors_history) if 'colors_history' in locals() and colors_history else 1,
                    algorithm_name=f"✅ BEST COLORING - {algorithm}"
                )
                st.pyplot(fig_final)
                plt.close(fig_final)
            
            with col_info:
                st.markdown("#### 📋 Coloring Details")
                st.write(f"**🎨 Colors Used:** `{n_colors}`")
                st.write(f"**📍 Nodes:** `{n_nodes}`")
                st.write(f"**🔗 Edges:** `{np.sum(graph_coloring.adj_matrix) // 2}`")
                st.write(f"**⚠️ Conflicts:** `{n_conflicts}`")
                
                if n_conflicts == 0:
                    st.success("✅ Valid coloring! (No conflicts)")
                else:
                    st.error(f"❌ Invalid: {n_conflicts} conflicts found")
                
                st.markdown("---")
                st.markdown("**🎨 Color Assignment:**")
                
                # Show color assignment dạng đẹp
                color_assignment = {}
                for color in range(n_colors):
                    nodes_with_color = [i for i, c in enumerate(coloring) if c == color]
                    color_assignment[f"Color {color}"] = nodes_with_color
                
                for color_name, nodes in color_assignment.items():
                    nodes_str = ', '.join([f'`{n}`' for n in nodes])
                    st.markdown(f"**{color_name}:** {nodes_str}")
            
            with st.expander("📊 Detailed Statistics"):
                st.write(f"**Algorithm:** {algorithm}")
                st.write(f"**Colors Used:** {n_colors}")
                st.write(f"**Nodes:** {n_nodes}")
                st.write(f"**Edges:** {np.sum(graph_coloring.adj_matrix) // 2}")
                st.write(f"**Graph Density:** {(np.sum(graph_coloring.adj_matrix) / (n_nodes * (n_nodes - 1))):.2%}")
                st.write(f"**Conflicts:** {n_conflicts}")
                st.write(f"**Valid Coloring:** {'Yes ✅' if n_conflicts == 0 else 'No ❌'}")
                
                if 'colors_history' in locals() and colors_history and len(colors_history) > 1:
                    st.write(f"**Initial Colors:** {colors_history[0]}")
                    st.write(f"**Final Colors:** {n_colors}")
                    st.write(f"**Color Reduction:** {colors_history[0] - n_colors}")
                    st.write(f"**Iterations:** {len(colors_history)}")
                
                st.markdown("---")
                st.write("**Full Color Assignment:**")
                for color in range(n_colors):
                    nodes_with_color = [i for i, c in enumerate(coloring) if c == color]
                    st.write(f"  Color {color}: {nodes_with_color}")
        
        elif problem_type == "Path Finding (Grid World)":
            # ========================================
            # GRID WORLD / GRAPH SEARCH với REAL-TIME ANIMATION
            # ========================================
            
            grid = GridWorld(grid_size=(grid_size, grid_size), 
                           obstacle_prob=obstacle_prob, seed=42)
            
            st.success(f"Đang chạy {algorithm} trên Grid World {grid_size}x{grid_size}...")
            
            # Setup layout cho real-time visualization
            col1, col2 = st.columns([3, 1])
            
            with col1:
                plot_placeholder = st.empty()
            
            with col2:
                st.markdown("### Metrics")
                nodes_metric = st.empty()
                queue_metric = st.empty()
                path_metric = st.empty()
            
            progress_bar = st.progress(0)
            
            # Manual implementation với real-time animation
            from collections import deque
            import heapq
            
            start = grid.start
            goal = grid.goal
            
            # Initialize
            visited = set()
            came_from = {}
            
            if algorithm == "BFS":
                frontier = deque([start])
            elif algorithm == "DFS":
                frontier = [start]
            else:  # A*
                def manhattan_distance(a, b):
                    return abs(a[0] - b[0]) + abs(a[1] - b[1])
                
                g_score = {start: 0}
                f_score = {start: manhattan_distance(start, goal)}
                frontier = [(f_score[start], start)]
            
            found = False
            path = []
            iteration = 0
            max_iterations = grid_size * grid_size * 2
            
            # Real-time search với visualization từng bước
            while frontier and not found:
                # Get next node
                if algorithm == "BFS":
                    current = frontier.popleft()
                elif algorithm == "DFS":
                    current = frontier.pop()
                else:  # A*
                    _, current = heapq.heappop(frontier)
                
                if current in visited:
                    continue
                
                visited.add(current)
                iteration += 1
                
                # Check if goal
                if current == goal:
                    found = True
                    # Reconstruct path
                    path = []
                    node = current
                    while node in came_from:
                        path.append(node)
                        node = came_from[node]
                    path.append(start)
                    path.reverse()
                
                # Explore neighbors
                row, col = current
                neighbors = []
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_row, new_col = row + dr, col + dc
                    if (0 <= new_row < grid.rows and 
                        0 <= new_col < grid.cols and
                        grid.grid[new_row, new_col] != 1 and
                        (new_row, new_col) not in visited):
                        neighbors.append((new_row, new_col))
                
                # Add to frontier
                for neighbor in neighbors:
                    if algorithm == "BFS":
                        if neighbor not in [item for item in frontier]:
                            frontier.append(neighbor)
                            came_from[neighbor] = current
                    elif algorithm == "DFS":
                        if neighbor not in frontier:
                            frontier.append(neighbor)
                            came_from[neighbor] = current
                    else:  # A*
                        tentative_g = g_score.get(current, float('inf')) + 1
                        if neighbor not in g_score or tentative_g < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            f = tentative_g + manhattan_distance(neighbor, goal)
                            f_score[neighbor] = f
                            heapq.heappush(frontier, (f, neighbor))
                
                # Update metrics
                nodes_metric.metric("Nodes Expanded", len(visited))
                if algorithm in ["BFS", "DFS"]:
                    queue_metric.metric("Frontier Size", len(frontier))
                else:  # A*
                    queue_metric.metric("Frontier Size", len(frontier))
                path_metric.metric("Path Length", len(path) if found else "Searching...")
                
                # Real-time visualization - hiển thị từng bước
                fig, ax = plt.subplots(figsize=(7, 7))
                
                # Draw grid base - obstacles là 1, empty là 0
                grid_display = np.copy(grid.grid).astype(float)
                
                # Đặt giá trị cho obstacles và empty cells trước
                # Obstacles: giá trị thấp (đen/xám đậm) - 0.1
                # Empty cells: giá trị cao (trắng/xanh nhạt) - 0.9
                grid_display[grid.grid == 1] = 0.1  # Obstacles (đen/xám đậm)
                grid_display[grid.grid == 0] = 0.9  # Empty cells (trắng)
                
                # Color visited cells (light blue/gray - các node đã duyệt)
                # CHỈ tô màu nếu KHÔNG phải obstacle
                visited_list = list(visited)
                for v in visited_list:
                    if v != start and v != goal and grid.grid[v[0], v[1]] != 1:
                        grid_display[v[0], v[1]] = 0.75
                
                # Color frontier nodes (light cyan - đang chờ explore)
                # CHỈ tô màu nếu KHÔNG phải obstacle
                if algorithm in ["BFS", "DFS"]:
                    frontier_nodes = [f for f in frontier if isinstance(f, tuple)]
                else:  # A*
                    frontier_nodes = [f for _, f in frontier if isinstance(f, tuple)]
                
                for fnode in frontier_nodes:
                    if (fnode not in visited and fnode != start and fnode != goal and 
                        grid.grid[fnode[0], fnode[1]] != 1):
                        grid_display[fnode[0], fnode[1]] = 0.6
                
                # Color current node (orange - node đang xét)
                # CHỈ tô màu nếu KHÔNG phải obstacle
                if (not found and current != start and current != goal and 
                    grid.grid[current[0], current[1]] != 1):
                    grid_display[current[0], current[1]] = 0.85
                
                # Color start and goal (ưu tiên cao nhất, ghi đè mọi thứ)
                grid_display[start[0], start[1]] = 0.95  # Start (light green)
                grid_display[goal[0], goal[1]] = 0.05   # Goal (red)
                
                # Plot base grid với custom colormap để obstacles rõ ràng
                # Sử dụng colormap có màu đen rõ ràng cho obstacles
                im = ax.imshow(grid_display, cmap='RdYlGn', vmin=0, vmax=1,
                              aspect='equal', interpolation='nearest')
                
                # Add grid lines (vẽ trước obstacles để obstacles ở trên)
                ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
                ax.grid(which="minor", color="black", linestyle='-', linewidth=0.8, alpha=0.25)
                ax.tick_params(which="minor", size=0)
                ax.set_xticks(np.arange(0, grid_size, 1))
                ax.set_yticks(np.arange(0, grid_size, 1))
                
                # Vẽ obstacles rõ ràng bằng hình chữ nhật đen (zorder thấp để ở dưới)
                for i in range(grid.rows):
                    for j in range(grid.cols):
                        if grid.grid[i, j] == 1:
                            # Vẽ hình chữ nhật đen cho obstacle
                            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                facecolor='black', edgecolor='darkgray', 
                                                linewidth=1, alpha=1.0, zorder=2)
                            ax.add_patch(rect)
                
                # Nếu đã tìm thấy path - HIGHLIGHT PATH NGAY!
                if found and len(path) > 1:
                    path_y = [p[1] for p in path]
                    path_x = [p[0] for p in path]
                    
                    # Draw path với highlight rõ ràng (zorder cao để ở trên obstacles)
                    ax.plot(path_y, path_x, color='cyan', linewidth=10, 
                           alpha=0.3, zorder=11, linestyle='-')
                    ax.plot(path_y, path_x, color='deepskyblue', linewidth=7, 
                           alpha=0.5, zorder=12, linestyle='-')
                    ax.plot(path_y, path_x, color='cyan', linewidth=4, 
                           alpha=0.9, zorder=13, linestyle='-',
                           marker='o', markersize=7, markerfacecolor='cyan',
                           markeredgecolor='blue', markeredgewidth=1.5)
                    
                    # Highlight nodes trên path (chỉ highlight nếu không phải obstacle)
                    for px, py in path:
                        if (px, py) != start and (px, py) != goal and grid.grid[px, py] != 1:
                            ax.scatter(py, px, c='yellow', s=300, marker='o',
                                     alpha=0.9, zorder=14, edgecolors='orange', linewidths=2)
                
                # Draw CURRENT node marker (nếu chưa tìm thấy goal và không phải obstacle)
                if not found and grid.grid[current[0], current[1]] != 1:
                    ax.scatter(current[1], current[0], c='orange', s=400, marker='o',
                              alpha=0.8, zorder=16, edgecolors='red', linewidths=2)
                
                # Draw START marker (green circle) - luôn hiển thị
                ax.scatter(start[1], start[0], c='lime', s=600, marker='o',
                          alpha=0.4, zorder=18, edgecolors='none')
                ax.scatter(start[1], start[0], c='lime', s=500, marker='o',
                          edgecolors='darkgreen', linewidths=3, alpha=0.95,
                          zorder=20)
                ax.text(start[1], start[0], 'S', ha='center', va='center',
                       fontsize=16, fontweight='bold', color='black', zorder=21)
                
                # Draw GOAL marker (red circle) - luôn hiển thị
                ax.scatter(goal[1], goal[0], c='red', s=600, marker='o',
                          alpha=0.4, zorder=18, edgecolors='none')
                ax.scatter(goal[1], goal[0], c='red', s=500, marker='o',
                          edgecolors='darkred', linewidths=3, alpha=0.95,
                          zorder=20)
                ax.text(goal[1], goal[0], 'G', ha='center', va='center',
                       fontsize=16, fontweight='bold', color='white', zorder=21)
                
                # Title and labels
                status = "Path Found!" if found else "Searching..."
                ax.set_title(f'{algorithm} - {status}\n'
                           f'Iteration {iteration} | Nodes: {len(visited)} | Frontier: {len(frontier)}',
                           fontsize=13, fontweight='bold', pad=15,
                           color='darkgreen' if found else 'black')
                ax.set_xlabel('Column', fontsize=10, fontweight='bold')
                ax.set_ylabel('Row', fontsize=10, fontweight='bold')
                
                # Legend với obstacles
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='black', edgecolor='darkgray', label='Obstacle'),
                    Patch(facecolor='lime', edgecolor='darkgreen', label='Start'),
                    Patch(facecolor='red', edgecolor='darkred', label='Goal'),
                ]
                if not found:
                    legend_elements.append(Patch(facecolor='orange', edgecolor='red', label='Current'))
                legend_elements.append(Patch(facecolor='lightblue', edgecolor='blue', label='Visited'))
                legend_elements.append(Patch(facecolor='cyan', edgecolor='blue', label='Frontier'))
                if found and len(path) > 1:
                    legend_elements.append(Patch(facecolor='cyan', edgecolor='blue', label='Path'))
                
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                         fontsize=9, framealpha=0.9)
                
                # Equal aspect ratio
                ax.set_aspect('equal')
                
                plt.tight_layout()
                plot_placeholder.pyplot(fig)
                plt.close(fig)
                
                # Update progress
                progress_bar.progress(min(iteration / max_iterations, 1.0))
                
                # Delay để thấy animation
                time.sleep(animation_speed)
                
                # Nếu đã tìm thấy, dừng lại
                if found:
                    break
                
                # Safety check
                if iteration >= max_iterations:
                    break
            
            # Final results
            st.markdown("---")
            if found:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("## Path Found!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Path Length", len(path))
                with col2:
                    st.metric("Nodes Expanded", len(visited))
                with col3:
                    st.metric("Total Iterations", iteration)
                with col4:
                    st.metric("Algorithm", algorithm)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Path details
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.markdown("#### Path Details")
                    st.write(f"**Path Length:** `{len(path)}` steps")
                    st.write(f"**Nodes Expanded:** `{len(visited)}`")
                    st.write(f"**Total Iterations:** `{iteration}`")
                    st.write(f"**Algorithm:** `{algorithm}`")
                
                with col_info2:
                    st.markdown("#### Path Sequence:")
                    path_str = " → ".join([f"`({p[0]},{p[1]})`" for p in path[:20]])
                    if len(path) > 20:
                        path_str += f" ... (total {len(path)} steps)"
                    st.markdown(path_str)
                
                with st.expander("Detailed Statistics"):
                    st.write(f"**Algorithm:** {algorithm}")
                    st.write(f"**Grid Size:** {grid_size} × {grid_size}")
                    st.write(f"**Start:** {start}")
                    st.write(f"**Goal:** {goal}")
                    st.write(f"**Path Length:** {len(path)} steps")
                    st.write(f"**Nodes Expanded:** {len(visited)}")
                    st.write(f"**Total Iterations:** {iteration}")
                    st.write(f"**Path Found:** {'Yes' if found else 'No'}")
                    st.write(f"**Path Sequence:** {' → '.join([str(p) for p in path])}")
            else:
                st.error("No path found or search stopped!")
                
                st.markdown("#### Search Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Nodes Expanded:** {len(visited)}")
                with col2:
                    st.write(f"**Total Iterations:** {iteration}")
                with col3:
                    st.write(f"**Algorithm:** {algorithm}")
    
    else:
        # Welcome screen with team info
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%); 
                    padding: 2rem; border-radius: 15px; margin: 2rem 0; border-left: 5px solid #667eea;'>
            <h2 style='margin: 0 0 1rem 0; color: #667eea;'>👥 Nhóm Sinh Viên Thực Hiện</h2>
            <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <table style='width: 100%; border-collapse: collapse;'>
                    <thead>
                        <tr style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white;'>
                            <th style='padding: 12px; text-align: left; border-radius: 5px 0 0 0;'>STT</th>
                            <th style='padding: 12px; text-align: left;'>Họ và Tên</th>
                            <th style='padding: 12px; text-align: center; border-radius: 0 5px 0 0;'>MSSV</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style='background: #f8f9fa;'>
                            <td style='padding: 12px; border-bottom: 1px solid #dee2e6;'><strong>1</strong></td>
                            <td style='padding: 12px; border-bottom: 1px solid #dee2e6;'>Phạm Phú Hòa</td>
                            <td style='padding: 12px; text-align: center; border-bottom: 1px solid #dee2e6;'><code>23122030</code></td>
                        </tr>
                        <tr style='background: white;'>
                            <td style='padding: 12px; border-bottom: 1px solid #dee2e6;'><strong>2</strong></td>
                            <td style='padding: 12px; border-bottom: 1px solid #dee2e6;'>Đào Sỹ Duy Minh</td>
                            <td style='padding: 12px; text-align: center; border-bottom: 1px solid #dee2e6;'><code>23122041</code></td>
                        </tr>
                        <tr style='background: #f8f9fa;'>
                            <td style='padding: 12px; border-bottom: 1px solid #dee2e6;'><strong>3</strong></td>
                            <td style='padding: 12px; border-bottom: 1px solid #dee2e6;'>Trần Chí Nguyên</td>
                            <td style='padding: 12px; text-align: center; border-bottom: 1px solid #dee2e6;'><code>23122044</code></td>
                        </tr>
                        <tr style='background: white;'>
                            <td style='padding: 12px;'><strong>4</strong></td>
                            <td style='padding: 12px;'>Nguyễn Lâm Phú Quý</td>
                            <td style='padding: 12px; text-align: center;'><code>23122048</code></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown("""
        <div style='background: linear-gradient(135deg, #48c6ef22 0%, #6f86d622 100%); 
                    padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #48c6ef;'>
            <h3 style='margin: 0 0 1rem 0; color: #48c6ef;'>📋 Hướng Dẫn Sử Dụng</h3>
            <ol style='margin: 0; padding-left: 1.5rem; line-height: 1.8;'>
                <li><strong>Chọn bài toán</strong> ở sidebar (Continuous, TSP, Knapsack, Graph Coloring, hoặc Path Finding)</li>
                <li><strong>Chọn thuật toán</strong> (Swarm Intelligence hoặc Traditional Search)</li>
                <li><strong>Điều chỉnh tham số</strong> theo ý muốn</li>
                <li><strong>Nhấn "Run Animation"</strong> để bắt đầu visualization!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Features showcase
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb22 0%, #f5576c22 100%); 
                        padding: 1.5rem; border-radius: 10px; border-left: 5px solid #f093fb;'>
                <h4 style='margin: 0 0 1rem 0; color: #f093fb;'>🎨 Continuous Optimization</h4>
                <ul style='margin: 0; padding-left: 1.5rem; line-height: 1.8;'>
                    <li>3D Surface Plot animation</li>
                    <li>Real-time convergence tracking</li>
                    <li>Particles movement visualization</li>
                    <li>Functions: Sphere, Rastrigin, Rosenbrock, Ackley</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe22 0%, #00f2fe22 100%); 
                        padding: 1.5rem; border-radius: 10px; border-left: 5px solid #4facfe;'>
                <h4 style='margin: 0 0 1rem 0; color: #4facfe;'>🔍 Discrete Optimization</h4>
                <ul style='margin: 0; padding-left: 1.5rem; line-height: 1.8;'>
                    <li>TSP: Tour optimization animation</li>
                    <li>Knapsack: Item selection tracking</li>
                    <li>Graph Coloring: Node coloring process</li>
                    <li>Path Finding: BFS, DFS, A* visualization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

elif main_tab == "📊 Comparison Dashboard":
    # ========================================
    # TAB SO SÁNH THUẬT TOÁN
    # ========================================
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb22 0%, #f5576c22 100%); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; border-left: 5px solid #f093fb;'>
        <h2 style='margin: 0; color: #f093fb;'>📊 Comparison Dashboard</h2>
        <p style='margin: 0.5rem 0 0 0; color: #666;'>So sánh nhiều thuật toán với multiple runs để đánh giá robustness</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================
    # SIDEBAR: CONFIGURATION
    # ========================================
    st.sidebar.markdown("### 🎯 Step 1: Chọn Bài Toán")
    
    # Cấp 1: Chọn loại bài toán chính
    comp_problem_category = st.sidebar.radio(
        "Loại bài toán:",
        ["Continuous Optimization", "Discrete Optimization"],
        key="comp_problem_category",
        help="Chọn loại bài toán tối ưu"
    )
    
    # Cấp 2: Chọn bài toán cụ thể
    if comp_problem_category == "Continuous Optimization":
        comparison_problem_type = "Continuous Optimization"
        
        comp_func_name = st.sidebar.selectbox(
            "Chọn test function:",
            ["Sphere", "Rastrigin", "Rosenbrock", "Ackley"],
            key="comp_func"
        )
        comp_dim = st.sidebar.slider("Số chiều (dimension):", 2, 10, 5, key="comp_dim")
        
    else:  # Discrete Optimization
        comp_discrete_problem = st.sidebar.selectbox(
            "Chọn bài toán cụ thể:",
            ["TSP (Traveling Salesman)", "Knapsack", "Graph Coloring"],
            key="comp_discrete_problem",
            help="Chọn bài toán rời rạc cụ thể"
        )
        
        # Map to comparison_problem_type for backward compatibility
        if comp_discrete_problem == "TSP (Traveling Salesman)":
            comparison_problem_type = "Discrete (TSP)"
            comp_n_cities = st.sidebar.slider("Số thành phố:", 10, 50, 20, key="comp_tsp_cities")
            
        elif comp_discrete_problem == "Knapsack":
            comparison_problem_type = "Discrete (Knapsack)"
            comp_n_items = st.sidebar.slider("Số items:", 10, 100, 30, key="comp_kp_items")
            comp_kp_capacity = st.sidebar.slider("Capacity:", 50, 500, 200, key="comp_kp_cap")
            
        else:  # Graph Coloring
            comparison_problem_type = "Discrete (Graph Coloring)"
            comp_n_vertices = st.sidebar.slider("Số nodes:", 10, 50, 20, key="comp_gc_nodes")
            comp_edge_prob = st.sidebar.slider("Edge probability:", 0.1, 0.9, 0.3, key="comp_gc_prob")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Step 2: Chọn Thuật Toán")
    
    # Algorithm selection with checkboxes
    if comparison_problem_type == "Continuous Optimization":
        st.sidebar.markdown("**Swarm Intelligence:**")
        comp_use_pso = st.sidebar.checkbox("PSO", value=True, key="comp_pso")
        comp_use_aco = st.sidebar.checkbox("ACO", value=True, key="comp_aco")
        comp_use_abc = st.sidebar.checkbox("ABC", value=True, key="comp_abc")
        comp_use_firefly = st.sidebar.checkbox("Firefly", value=False, key="comp_firefly")
        comp_use_cuckoo = st.sidebar.checkbox("Cuckoo", value=False, key="comp_cuckoo")
        
        st.sidebar.markdown("**Traditional Search:**")
        comp_use_hc = st.sidebar.checkbox("Hill Climbing", value=True, key="comp_hc")
        comp_use_sa = st.sidebar.checkbox("Simulated Annealing", value=True, key="comp_sa")
        comp_use_ga = st.sidebar.checkbox("Genetic Algorithm", value=True, key="comp_ga")
        
    elif comparison_problem_type == "Discrete (TSP)":
        st.sidebar.markdown("**Traditional Search:**")
        comp_use_ga = st.sidebar.checkbox("Genetic Algorithm", value=True, key="comp_tsp_ga")
        comp_use_hc = st.sidebar.checkbox("Hill Climbing (2-opt)", value=True, key="comp_tsp_hc")
        comp_use_sa = st.sidebar.checkbox("Simulated Annealing (2-opt)", value=True, key="comp_tsp_sa")
        
    elif comparison_problem_type == "Discrete (Knapsack)":
        st.sidebar.markdown("**Traditional Search:**")
        comp_use_ga = st.sidebar.checkbox("Genetic Algorithm", value=True, key="comp_kp_ga")
        comp_use_hc = st.sidebar.checkbox("Hill Climbing", value=True, key="comp_kp_hc")
        comp_use_sa = st.sidebar.checkbox("Simulated Annealing", value=True, key="comp_kp_sa")
        
    elif comparison_problem_type == "Discrete (Graph Coloring)":
        st.sidebar.markdown("**Metaheuristics:**")
        comp_use_ga = st.sidebar.checkbox("Genetic Algorithm", value=True, key="comp_gc_ga")
        comp_use_hc = st.sidebar.checkbox("Hill Climbing", value=True, key="comp_gc_hc")
        comp_use_sa = st.sidebar.checkbox("Simulated Annealing", value=True, key="comp_gc_sa")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Step 3: Cấu Hình Thí Nghiệm")
    
    comp_n_runs = st.sidebar.slider("Số lần chạy (cho robustness):", 5, 50, 30, step=5, key="comp_runs")
    comp_max_iter = st.sidebar.slider("Max iterations:", 50, 500, 100, step=50, key="comp_iter")
    
    if comparison_problem_type == "Continuous Optimization":
        comp_pop_size = st.sidebar.slider("Population size:", 10, 100, 30, step=10, key="comp_pop")
    
    st.sidebar.markdown("---")
    comp_run_button = st.sidebar.button("🚀 Run Comparison", type="primary", use_container_width=True)
    
    # ========================================
    # MAIN CONTENT: RESULTS
    # ========================================
    
    if comp_run_button:
        # Collect selected algorithms
        selected_algorithms = []
        
        if comparison_problem_type == "Continuous Optimization":
            if comp_use_pso: selected_algorithms.append("PSO")
            if comp_use_aco: selected_algorithms.append("ACO")
            if comp_use_abc: selected_algorithms.append("ABC")
            if comp_use_firefly: selected_algorithms.append("Firefly")
            if comp_use_cuckoo: selected_algorithms.append("Cuckoo")
            if comp_use_hc: selected_algorithms.append("Hill Climbing")
            if comp_use_sa: selected_algorithms.append("Simulated Annealing")
            if comp_use_ga: selected_algorithms.append("Genetic Algorithm")
            
        elif comparison_problem_type == "Discrete (TSP)":
            if comp_use_ga: selected_algorithms.append("Genetic Algorithm")
            if comp_use_hc: selected_algorithms.append("Hill Climbing")
            if comp_use_sa: selected_algorithms.append("Simulated Annealing")
            
        elif comparison_problem_type == "Discrete (Knapsack)":
            if comp_use_ga: selected_algorithms.append("Genetic Algorithm")
            if comp_use_hc: selected_algorithms.append("Hill Climbing")
            if comp_use_sa: selected_algorithms.append("Simulated Annealing")
            
        elif comparison_problem_type == "Discrete (Graph Coloring)":
            if comp_use_ga: selected_algorithms.append("Genetic Algorithm")
            if comp_use_hc: selected_algorithms.append("Hill Climbing")
            if comp_use_sa: selected_algorithms.append("Simulated Annealing")
        
        if len(selected_algorithms) == 0:
            st.error("❌ Vui lòng chọn ít nhất 1 thuật toán!")
        else:
            # Show progress
            st.markdown(f"### 🔬 Đang chạy {len(selected_algorithms)} thuật toán × {comp_n_runs} runs...")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            # Results storage
            all_results = {}
            
            # ========================================
            # RUN EXPERIMENTS
            # ========================================
            total_experiments = len(selected_algorithms) * comp_n_runs
            current_experiment = 0
            
            for algo_name in selected_algorithms:
                all_results[algo_name] = {
                    'best_scores': [],
                    'convergence_curves': [],
                    'runtimes': [],
                    'iterations_to_threshold': [],
                    'final_solutions': []
                }
                
                for run_idx in range(comp_n_runs):
                    status_text.text(f"🏃 {algo_name} - Run {run_idx + 1}/{comp_n_runs}")
                    
                    start_time = time.time()
                    
                    # ========================================
                    # CONTINUOUS OPTIMIZATION
                    # ========================================
                    if comparison_problem_type == "Continuous Optimization":
                        test_func = get_test_function(comp_func_name, comp_dim)
                        
                        if algo_name == "PSO":
                            algo = PSO(n_particles=comp_pop_size, dim=comp_dim, max_iter=comp_max_iter, 
                                      bounds=test_func.bounds)
                            best_pos, best_score = algo.optimize(test_func)
                            history = algo.get_history()
                            
                        elif algo_name == "ACO":
                            algo = ACO(n_ants=comp_pop_size, dim=comp_dim, max_iter=comp_max_iter, 
                                      bounds=test_func.bounds)
                            best_pos, best_score = algo.optimize(test_func)
                            history = algo.get_history()
                            
                        elif algo_name == "ABC":
                            algo = ABC(n_bees=comp_pop_size, dim=comp_dim, max_iter=comp_max_iter, 
                                      bounds=test_func.bounds)
                            best_pos, best_score = algo.optimize(test_func)
                            history = algo.get_history()
                            
                        elif algo_name == "Firefly":
                            algo = FireflyAlgorithm(n_fireflies=comp_pop_size, dim=comp_dim, max_iter=comp_max_iter, 
                                                   bounds=test_func.bounds)
                            best_pos, best_score = algo.optimize(test_func)
                            history = algo.get_history()
                            
                        elif algo_name == "Cuckoo":
                            algo = CuckooSearch(n_nests=comp_pop_size, dim=comp_dim, max_iter=comp_max_iter, 
                                               bounds=test_func.bounds)
                            best_pos, best_score = algo.optimize(test_func)
                            history = algo.get_history()
                            
                        elif algo_name == "Hill Climbing":
                            algo = HillClimbing(dim=comp_dim, max_iter=comp_max_iter, 
                                               bounds=test_func.bounds)
                            best_pos, best_score = algo.optimize(test_func)
                            history = algo.get_history()
                            
                        elif algo_name == "Simulated Annealing":
                            algo = SimulatedAnnealing(dim=comp_dim, max_iter=comp_max_iter, 
                                                     bounds=test_func.bounds)
                            best_pos, best_score = algo.optimize(test_func)
                            history = algo.get_history()
                            
                        elif algo_name == "Genetic Algorithm":
                            algo = GeneticAlgorithm(population_size=comp_pop_size, dim=comp_dim, max_iter=comp_max_iter, 
                                                   bounds=test_func.bounds)
                            best_pos, best_score = algo.optimize(test_func)
                            history = algo.get_history()
                        
                        convergence_curve = history['best_scores']
                        
                    # ========================================
                    # TSP
                    # ========================================
                    elif comparison_problem_type == "Discrete (TSP)":
                        # Create TSP problem
                        tsp = TSP(n_cities=comp_n_cities, seed=run_idx * 1000)
                        
                        if algo_name == "Genetic Algorithm":
                            best_tour, best_distance, history = TSPSolver.genetic_algorithm_tsp(
                                tsp, population_size=50, max_iter=comp_max_iter)
                            best_score = best_distance
                            convergence_curve = history['best_distances']
                            
                        elif algo_name == "Hill Climbing":
                            best_tour, best_distance = TSPSolver.two_opt(tsp, max_iter=comp_max_iter)
                            best_score = best_distance
                            convergence_curve = [best_distance] * comp_max_iter
                            
                        elif algo_name == "Simulated Annealing":
                            # 2-opt with SA doesn't exist, use two_opt
                            best_tour, best_distance = TSPSolver.two_opt(tsp, max_iter=comp_max_iter)
                            best_score = best_distance
                            convergence_curve = [best_distance] * comp_max_iter
                    
                    # ========================================
                    # KNAPSACK
                    # ========================================
                    elif comparison_problem_type == "Discrete (Knapsack)":
                        # Create Knapsack problem
                        knapsack = Knapsack(n_items=comp_n_items, capacity=comp_kp_capacity, seed=run_idx * 1000)
                        
                        if algo_name == "Genetic Algorithm":
                            best_solution, best_value, history = KnapsackSolver.genetic_algorithm(
                                knapsack, population_size=50, max_iter=comp_max_iter)
                            best_score = -best_value  # Negative because we minimize
                            convergence_curve = [-v for v in history['best_values']]
                            
                        elif algo_name == "Hill Climbing":
                            best_solution, best_value, history = KnapsackSolver.hill_climbing(
                                knapsack, max_iter=comp_max_iter)
                            best_score = -best_value
                            convergence_curve = [-v for v in history['best_values']]
                            
                        elif algo_name == "Simulated Annealing":
                            best_solution, best_value, history = KnapsackSolver.simulated_annealing(
                                knapsack, max_iter=comp_max_iter)
                            best_score = -best_value
                            convergence_curve = [-v for v in history['best_values']]
                    
                    # ========================================
                    # GRAPH COLORING
                    # ========================================
                    elif comparison_problem_type == "Discrete (Graph Coloring)":
                        # Create Graph Coloring problem
                        graph_coloring = GraphColoring(n_vertices=comp_n_vertices, edge_probability=comp_edge_prob, 
                                                      seed=run_idx * 1000)
                        
                        if algo_name == "Genetic Algorithm":
                            best_coloring, best_conflicts, history = GraphColoringSolver.genetic_algorithm(
                                graph_coloring, population_size=50, max_iter=comp_max_iter)
                            best_score = best_conflicts
                            convergence_curve = history['best_scores']
                            
                        elif algo_name == "Hill Climbing":
                            best_coloring, best_conflicts, history = GraphColoringSolver.hill_climbing(
                                graph_coloring, max_iter=comp_max_iter)
                            best_score = best_conflicts
                            convergence_curve = history['best_scores']
                            
                        elif algo_name == "Simulated Annealing":
                            best_coloring, best_conflicts, history = GraphColoringSolver.simulated_annealing(
                                graph_coloring, max_iter=comp_max_iter)
                            best_score = best_conflicts
                            convergence_curve = history['best_scores']
                    
                    runtime = time.time() - start_time
                    
                    # Store results
                    all_results[algo_name]['best_scores'].append(best_score)
                    all_results[algo_name]['convergence_curves'].append(convergence_curve)
                    all_results[algo_name]['runtimes'].append(runtime)
                    
                    # Calculate iterations to threshold (when reached 99% of best)
                    if comparison_problem_type == "Continuous Optimization":
                        threshold = test_func.global_optimum + 0.01 * abs(test_func.global_optimum)
                        iter_to_threshold = comp_max_iter
                        for i, score in enumerate(convergence_curve):
                            if score <= threshold:
                                iter_to_threshold = i
                                break
                        all_results[algo_name]['iterations_to_threshold'].append(iter_to_threshold)
                    
                    # Update progress
                    current_experiment += 1
                    progress_bar.progress(current_experiment / total_experiments)
            
            status_text.text("✅ Hoàn thành!")
            progress_bar.progress(1.0)
            
            # ========================================
            # DISPLAY RESULTS
            # ========================================
            st.markdown("---")
            st.markdown("## 📈 Kết Quả So Sánh")
            
            # Create tabs for different views
            result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
                "📈 Convergence", "📊 Performance Metrics", "📦 Robustness", "💾 Export"
            ])
            
            # ========================================
            # TAB 1: CONVERGENCE COMPARISON
            # ========================================
            with result_tab1:
                st.markdown("### 📈 So Sánh Convergence Curves")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                
                colors = plt.cm.tab10(np.linspace(0, 1, len(selected_algorithms)))
                
                for idx, algo_name in enumerate(selected_algorithms):
                    curves = np.array(all_results[algo_name]['convergence_curves'])
                    mean_curve = np.mean(curves, axis=0)
                    std_curve = np.std(curves, axis=0)
                    
                    iterations = np.arange(len(mean_curve))
                    
                    ax.plot(iterations, mean_curve, label=algo_name, color=colors[idx], linewidth=2)
                    ax.fill_between(iterations, mean_curve - std_curve, mean_curve + std_curve, 
                                     alpha=0.2, color=colors[idx])
                
                ax.set_xlabel('Iteration', fontsize=12)
                if comparison_problem_type == "Discrete (Knapsack)":
                    ax.set_ylabel('Best Value (negative for minimization)', fontsize=12)
                else:
                    ax.set_ylabel('Best Score', fontsize=12)
                ax.set_title(f'Convergence Comparison - {comparison_problem_type}', fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
                # Statistics table
                st.markdown("### 📊 Convergence Statistics")
                
                convergence_stats = []
                for algo_name in selected_algorithms:
                    curves = np.array(all_results[algo_name]['convergence_curves'])
                    mean_curve = np.mean(curves, axis=0)
                    
                    # Final convergence value
                    final_mean = mean_curve[-1]
                    final_std = np.std([c[-1] for c in curves])
                    
                    # Improvement rate
                    if len(mean_curve) > 1:
                        improvement = abs(mean_curve[0] - mean_curve[-1])
                    else:
                        improvement = 0
                    
                    convergence_stats.append({
                        'Algorithm': algo_name,
                        'Final Score (Mean)': f"{final_mean:.4f}",
                        'Final Score (Std)': f"{final_std:.4f}",
                        'Total Improvement': f"{improvement:.4f}"
                    })
                
                df_convergence = pd.DataFrame(convergence_stats)
                st.dataframe(df_convergence, use_container_width=True)
            
            # ========================================
            # TAB 2: PERFORMANCE METRICS
            # ========================================
            with result_tab2:
                st.markdown("### 📊 Performance Metrics Table")
                
                metrics_data = []
                for algo_name in selected_algorithms:
                    best_scores = all_results[algo_name]['best_scores']
                    runtimes = all_results[algo_name]['runtimes']
                    
                    metrics_data.append({
                        'Algorithm': algo_name,
                        'Mean Score': f"{np.mean(best_scores):.4f}",
                        'Std Score': f"{np.std(best_scores):.4f}",
                        'Best Score': f"{np.min(best_scores):.4f}",
                        'Worst Score': f"{np.max(best_scores):.4f}",
                        'Mean Time (s)': f"{np.mean(runtimes):.3f}",
                        'Std Time (s)': f"{np.std(runtimes):.3f}",
                        'Success Rate (%)': f"{(np.sum(np.array(best_scores) <= np.min(best_scores) * 1.1) / len(best_scores) * 100):.1f}"
                    })
                
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, use_container_width=True)
                
                # Bar charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### ⏱️ Mean Runtime Comparison")
                    fig, ax = plt.subplots(figsize=(7, 4))
                    
                    runtimes_mean = [np.mean(all_results[algo]['runtimes']) for algo in selected_algorithms]
                    runtimes_std = [np.std(all_results[algo]['runtimes']) for algo in selected_algorithms]
                    
                    colors = plt.cm.viridis(np.linspace(0, 0.8, len(selected_algorithms)))
                    bars = ax.bar(selected_algorithms, runtimes_mean, yerr=runtimes_std, 
                                   capsize=5, color=colors, alpha=0.8, edgecolor='black')
                    
                    ax.set_ylabel('Time (seconds)', fontsize=11)
                    ax.set_title('Mean Runtime (with Std)', fontsize=12, fontweight='bold')
                    ax.tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("#### 🎯 Best Score Comparison")
                    fig, ax = plt.subplots(figsize=(7, 4))
                    
                    scores_mean = [np.mean(all_results[algo]['best_scores']) for algo in selected_algorithms]
                    scores_std = [np.std(all_results[algo]['best_scores']) for algo in selected_algorithms]
                    
                    colors = plt.cm.plasma(np.linspace(0, 0.8, len(selected_algorithms)))
                    bars = ax.bar(selected_algorithms, scores_mean, yerr=scores_std, 
                                   capsize=5, color=colors, alpha=0.8, edgecolor='black')
                    
                    ax.set_ylabel('Score', fontsize=11)
                    ax.set_title('Mean Best Score (with Std)', fontsize=12, fontweight='bold')
                    ax.tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close()
            
            # ========================================
            # TAB 3: ROBUSTNESS (BOX PLOTS)
            # ========================================
            with result_tab3:
                st.markdown("### 📦 Robustness Analysis (Box Plots)")
                st.markdown("Phân tích độ ổn định của thuật toán qua nhiều runs")
                
                # Box plot for best scores
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Best Scores Box Plot
                scores_data = [all_results[algo]['best_scores'] for algo in selected_algorithms]
                
                bp1 = ax1.boxplot(scores_data, labels=selected_algorithms, patch_artist=True, 
                                   notch=True, showmeans=True)
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(selected_algorithms)))
                for patch, color in zip(bp1['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax1.set_ylabel('Best Score', fontsize=12)
                ax1.set_title('Best Score Distribution', fontsize=13, fontweight='bold')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Runtime Box Plot
                runtime_data = [all_results[algo]['runtimes'] for algo in selected_algorithms]
                
                bp2 = ax2.boxplot(runtime_data, labels=selected_algorithms, patch_artist=True, 
                                   notch=True, showmeans=True)
                
                for patch, color in zip(bp2['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax2.set_ylabel('Runtime (seconds)', fontsize=12)
                ax2.set_title('Runtime Distribution', fontsize=13, fontweight='bold')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Statistical summary
                st.markdown("### 📊 Statistical Summary")
                
                stat_summary = []
                for algo_name in selected_algorithms:
                    best_scores = all_results[algo_name]['best_scores']
                    
                    stat_summary.append({
                        'Algorithm': algo_name,
                        'Mean': f"{np.mean(best_scores):.4f}",
                        'Median': f"{np.median(best_scores):.4f}",
                        'Std': f"{np.std(best_scores):.4f}",
                        'Q1 (25%)': f"{np.percentile(best_scores, 25):.4f}",
                        'Q3 (75%)': f"{np.percentile(best_scores, 75):.4f}",
                        'Min': f"{np.min(best_scores):.4f}",
                        'Max': f"{np.max(best_scores):.4f}",
                        'IQR': f"{np.percentile(best_scores, 75) - np.percentile(best_scores, 25):.4f}"
                    })
                
                df_stats = pd.DataFrame(stat_summary)
                st.dataframe(df_stats, use_container_width=True)
                
                # Coefficient of Variation (CV) - measure of relative variability
                st.markdown("### 🎯 Coefficient of Variation (Lower is More Robust)")
                
                cv_data = []
                for algo_name in selected_algorithms:
                    best_scores = all_results[algo_name]['best_scores']
                    mean_score = np.mean(best_scores)
                    std_score = np.std(best_scores)
                    cv = (std_score / abs(mean_score)) * 100 if mean_score != 0 else 0
                    
                    cv_data.append({
                        'Algorithm': algo_name,
                        'CV (%)': f"{cv:.2f}"
                    })
                
                df_cv = pd.DataFrame(cv_data)
                st.dataframe(df_cv, use_container_width=True)
            
            # ========================================
            # TAB 4: EXPORT
            # ========================================
            with result_tab4:
                st.markdown("### 💾 Export Results")
                
                # Export to CSV
                st.markdown("#### 📄 Export Data to CSV")
                
                # Prepare data for CSV
                csv_data = []
                for algo_name in selected_algorithms:
                    for run_idx in range(comp_n_runs):
                        csv_data.append({
                            'Algorithm': algo_name,
                            'Run': run_idx + 1,
                            'Best Score': all_results[algo_name]['best_scores'][run_idx],
                            'Runtime (s)': all_results[algo_name]['runtimes'][run_idx]
                        })
                
                df_export = pd.DataFrame(csv_data)
                csv_string = df_export.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_string,
                    file_name=f"comparison_results_{comparison_problem_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Export LaTeX table
                st.markdown("#### 📜 LaTeX Table (for Report)")
                
                latex_table = "\\begin{table}[h]\n\\centering\n\\caption{Algorithm Comparison Results}\n"
                latex_table += "\\begin{tabular}{|l|c|c|c|c|}\n\\hline\n"
                latex_table += "Algorithm & Mean Score & Std & Mean Time (s) & Success Rate (\\%) \\\\\n\\hline\n"
                
                for algo_name in selected_algorithms:
                    best_scores = all_results[algo_name]['best_scores']
                    runtimes = all_results[algo_name]['runtimes']
                    success_rate = (np.sum(np.array(best_scores) <= np.min(best_scores) * 1.1) / len(best_scores) * 100)
                    
                    latex_table += f"{algo_name} & {np.mean(best_scores):.4f} & {np.std(best_scores):.4f} & "
                    latex_table += f"{np.mean(runtimes):.3f} & {success_rate:.1f} \\\\\n"
                
                latex_table += "\\hline\n\\end{tabular}\n\\end{table}"
                
                st.code(latex_table, language="latex")
                
                # Export Summary JSON
                st.markdown("#### 📦 Export Summary (JSON)")
                
                summary_data = {}
                for algo_name in selected_algorithms:
                    summary_data[algo_name] = {
                        'mean_score': float(np.mean(all_results[algo_name]['best_scores'])),
                        'std_score': float(np.std(all_results[algo_name]['best_scores'])),
                        'best_score': float(np.min(all_results[algo_name]['best_scores'])),
                        'worst_score': float(np.max(all_results[algo_name]['best_scores'])),
                        'mean_runtime': float(np.mean(all_results[algo_name]['runtimes'])),
                        'std_runtime': float(np.std(all_results[algo_name]['runtimes']))
                    }
                
                json_string = json.dumps(summary_data, indent=2)
                
                st.download_button(
                    label="📥 Download JSON Summary",
                    data=json_string,
                    file_name=f"comparison_summary_{comparison_problem_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                st.success("✅ All export options are ready!")
    
    else:
        # Welcome screen for Comparison Dashboard
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("## 👋 Chào mừng đến với Comparison Dashboard!")
        st.markdown("""
        ### 📋 Hướng dẫn sử dụng:
        
        1. **Step 1: Chọn bài toán** ở sidebar (Continuous, TSP, Knapsack, Graph Coloring)
        2. **Step 2: Chọn thuật toán** để so sánh (checkboxes)
        3. **Step 3: Cấu hình thí nghiệm**:
           - Số lần chạy (cho robustness analysis): 30 runs recommended
           - Max iterations: 100-500
           - Problem size: tùy theo loại bài toán
        4. **Nhấn "Run Comparison"** để bắt đầu!
        
        ### 📊 Bạn sẽ nhận được:
        
        #### 📈 Tab 1: Convergence Comparison
        - Multi-line plot: convergence curves của tất cả algorithms
        - Shaded regions: standard deviation bands
        - Statistics table: final scores, improvements
        
        #### 📊 Tab 2: Performance Metrics
        - Comprehensive table: mean, std, best, worst scores
        - Runtime analysis: time complexity comparison
        - Success rate: % runs reaching near-optimal solution
        - Bar charts: visual comparison
        
        #### 📦 Tab 3: Robustness Analysis
        - Box plots: distribution of results across multiple runs
        - Statistical summary: mean, median, quartiles, IQR
        - Coefficient of Variation (CV): measure of consistency
        
        #### 💾 Tab 4: Export
        - CSV: raw data for further analysis
        - LaTeX table: ready for academic reports
        - JSON summary: structured results
        
        ### 🎯 Metrics Theo Yêu Cầu Của Thầy:
        
        ✅ **Convergence Speed**: Iterations to reach threshold, convergence rate  
        ✅ **Computational Complexity**: Time (seconds), space (memory)  
        ✅ **Robustness**: Multiple runs statistics, box plots  
        ✅ **Scalability**: Performance với problem sizes khác nhau  
        
        ### 🤖 Supported Algorithms:
        
        **Swarm Intelligence**: PSO, ACO, ABC, Firefly, Cuckoo  
        **Traditional**: Hill Climbing, Simulated Annealing, Genetic Algorithm  
        **Graph Search**: BFS, DFS, A* (for path finding)  
        
        ### 🧪 Test Problems:
        
        **Continuous**: Sphere, Rastrigin, Rosenbrock, Ackley  
        **Discrete**: TSP, Knapsack, Graph Coloring, Path Finding  
        """)
        st.markdown('</div>', unsafe_allow_html=True)

elif main_tab == "ℹ️ Algorithm Info":
    # ========================================
    # TAB THÔNG TIN THUẬT TOÁN
    # ========================================
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4facfe22 0%, #00f2fe22 100%); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; border-left: 5px solid #4facfe;'>
        <h2 style='margin: 0; color: #4facfe;'>📚 Thông Tin Thuật Toán</h2>
        <p style='margin: 0.5rem 0 0 0; color: #666;'>Chi tiết về các thuật toán Swarm Intelligence và Traditional Search</p>
    </div>
    """, unsafe_allow_html=True)
    
    algo_info = get_algorithm_info()
    
    # Filter by type
    algo_type_filter = st.radio(
        "Lọc theo loại:",
        ["All", "Swarm Intelligence", "Traditional Search", "Graph Search"],
        horizontal=True
    )
    
    # Display algorithms
    for algo_name, info in algo_info.items():
        if algo_type_filter == "All" or \
           (algo_type_filter == "Swarm Intelligence" and info["type"] == "swarm") or \
           (algo_type_filter == "Traditional Search" and info["type"] == "traditional") or \
           (algo_type_filter == "Graph Search" and info["type"] == "graph"):
            
            with st.expander(f"**{info['name']}** ({algo_name})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Mô tả:** {info['description']}")
                    st.markdown(f"**Loại:** {info['type'].capitalize()}")
                    st.markdown(f"**Tốt cho:** {info['best_for']}")
                
                with col2:
                    st.markdown("**Tham số chính:**")
                    for param in info['params']:
                        st.markdown(f"- `{param}`")

else:
    st.info(f"Tab '{main_tab}' đang được phát triển...")
    st.markdown("### 🚧 Coming Soon!")
    st.markdown("""
    Các tính năng sắp có:
    - **Comparison Dashboard**: So sánh nhiều thuật toán cùng lúc
    - **Batch Experiments**: Chạy nhiều runs để tính statistics
    - **Export Results**: Xuất charts và data
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p><strong>Optimization Algorithms Visualization & Comparison</strong></p>
</div>
""", unsafe_allow_html=True)


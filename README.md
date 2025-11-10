# Đồ án 1 - Thuật toán Swarm Intelligence

## Thông tin môn học
- **Môn học:** CSC14003 - Cơ sở Trí tuệ Nhân tạo
- **Khoa:** Công nghệ Thông tin - ĐHKHTN TPHCM

## Mô tả dự án
Dự án này tập trung vào việc implement, phân tích và so sánh các thuật toán swarm intelligence (tối ưu hóa bầy đàn) sử dụng NumPy.

## Nội dung

### 5 Thuật toán Swarm Intelligence
1. **Ant Colony Optimization (ACO)** - Tối ưu hóa đàn kiến
2. **Particle Swarm Optimization (PSO)** - Tối ưu hóa bầy đàn hạt
3. **Artificial Bee Colony (ABC)** - Thuật toán đàn ong nhân tạo
4. **Firefly Algorithm (FA)** - Thuật toán đom đóm
5. **Cuckoo Search (CS)** - Thuật toán chim cúc cu

### 6 Thuật toán tìm kiếm truyền thống (để so sánh)
1. **Hill Climbing** - Leo đồi
2. **Simulated Annealing** - Mô phỏng ủ kim loại
3. **Genetic Algorithm** - Thuật toán di truyền
4. **Breadth-First Search (BFS)** - Tìm kiếm theo chiều rộng
5. **Depth-First Search (DFS)** - Tìm kiếm theo chiều sâu
6. **A* Search** - Tìm kiếm A* (cho path-finding)

### Hàm test
#### Continuous Optimization:
- Sphere Function
- Rastrigin Function
- Rosenbrock Function
- Ackley Function

#### Discrete Optimization:
- Traveling Salesman Problem (TSP)
- Knapsack Problem (KP)
- Graph Coloring (GC)
- Path Finding (GridWorld for BFS/DFS/A*)

## Cấu trúc thư mục
```
Co_So_AI_search/
├── src/                          # Source code
│   ├── test_functions.py         # Các hàm test
│   ├── swarm_intelligence/       # Thuật toán swarm intelligence
│   │   ├── aco.py
│   │   ├── pso.py
│   │   ├── abc.py
│   │   ├── fa.py
│   │   └── cs.py
│   ├── traditional_search/       # Thuật toán truyền thống
│   │   ├── hill_climbing.py
│   │   ├── simulated_annealing.py
│   │   ├── genetic_algorithm.py
│   │   └── graph_search.py           # BFS, DFS, A*
│   ├── discrete_problems/        # Bài toán rời rạc
│   │   ├── tsp.py
│   │   ├── knapsack.py
│   │   └── graph_coloring.py
│   ├── visualization.py          # Module visualization
│   └── comparison.py             # Module so sánh
├── notebooks/                    # Jupyter notebooks
│   ├── 01_test_functions.ipynb
│   ├── 02_swarm_algorithms.ipynb
│   ├── 03_traditional_algorithms.ipynb
│   ├── 04_comparison.ipynb
│   └── 05_report.ipynb
├── results/                      # Kết quả thực nghiệm
├── report/                       # Báo cáo
├── requirements.txt
└── README.md
```

## 🚀 Cài đặt nhanh

### Bước 1: Clone repository

```bash
git clone https://github.com/your-repo/Co_So_AI_DoAn1_Search.git
cd Co_So_AI_DoAn1_Search
```

### Bước 2: Cài đặt dependencies

#### Cách 1: Sử dụng Conda (Khuyến nghị)

```bash
# Tạo môi trường conda từ file environment.yml
conda env create -f environment.yml

# Kích hoạt môi trường
conda activate co_so_ai_doan1_search
```

#### Cách 2: Sử dụng pip

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 3: Chạy ứng dụng

#### Option 1: Interactive Web App (Streamlit) 🌟 KHUYẾN NGHỊ

```bash
# Chạy ứng dụng chính với đầy đủ tính năng
streamlit run main.py

# Hoặc chạy demo animation
streamlit run app_animated.py
```

Ứng dụng sẽ mở tại: http://localhost:8501

#### Option 2: Jupyter Notebooks (Cho phân tích chi tiết)

```bash
# Khởi động Jupyter Notebook
jupyter notebook

# Hoặc Jupyter Lab
jupyter lab
```

Mở file `notebooks/complete_experiments.ipynb` để xem experiments đầy đủ.

#### Option 3: Command Line (Cho scripting)

```python
# Chạy một thuật toán cụ thể
python -c "
from src.swarm_intelligence.pso import PSO
from src.test_functions import get_test_function

func = get_test_function('sphere', dim=10)
pso = PSO(n_particles=30, dim=10, max_iter=100, bounds=func.bounds)
best_pos, best_score = pso.optimize(func, verbose=True)
print(f'Best score: {best_score}')
"
```

## 📱 Hướng dẫn sử dụng Ứng dụng Web

### Tab 1: 🎬 Animation Demo

**Chức năng chính:**
- Visualize thuật toán chạy real-time trên 3D surface
- Xem particles/agents di chuyển và hội tụ
- Theo dõi metrics: best score, convergence, gap to optimum

**Hướng dẫn:**

1. **Chọn Bài toán** (Sidebar):
   - Continuous Optimization (Sphere, Rastrigin, Rosenbrock, Ackley)
   - Discrete Optimization (TSP, Knapsack)
   - Path Finding (Grid World với BFS/DFS/A*)

2. **Chọn Thuật toán**:
   - **Swarm Intelligence**: PSO, ACO, ABC, Firefly, Cuckoo
   - **Traditional Search**: Hill Climbing, Simulated Annealing, Genetic Algorithm
   - **Graph Search**: BFS, DFS, A*

3. **Điều chỉnh Tham số**:
   - Population/Swarm Size (10-100)
   - Max Iterations (10-200)
   - Tham số đặc trưng của từng thuật toán (w, c1, c2 cho PSO, etc.)
   - Animation Speed (0.01-1.0s delay)

4. **Nhấn "Run Animation"**:
   - Xem 3D plot với particles di chuyển
   - Theo dõi convergence graph real-time
   - Xem metrics cập nhật

**Giải thích Visualization:**

- **3D Surface**: Thể hiện landscape của hàm mục tiêu
- **Particles (màu xanh lá → đỏ)**: Population/swarm, màu thể hiện fitness (xanh = tốt, đỏ = xấu)
- **Sao đỏ lớn**: Best solution hiện tại
- **Sao xanh lá**: Global optimum (nếu biết)
- **Camera xoay**: View tự động xoay để xem từ nhiều góc

### Tab 2: 📊 Comparison Dashboard (Đang phát triển)

**Chức năng:**
- So sánh nhiều thuật toán cùng lúc
- Chạy đồng thời và visualize trên cùng surface
- So sánh convergence curves
- Statistical comparison (mean, std, box plots)

### Tab 3: 📈 Batch Experiments (Đang phát triển)

**Chức năng:**
- Chạy multiple runs tự động
- Tính statistics: mean, std, success rate
- Export results (CSV, JSON)
- Generate report-ready figures (PNG, PDF, 300 DPI)

### Tab 4: ℹ️ Algorithm Info

**Chức năng:**
- Xem thông tin chi tiết các thuật toán
- Mô tả cách hoạt động
- Các tham số và ý nghĩa
- Use cases phù hợp

## 🎯 Ví dụ Sử dụng

### Ví dụ 1: Chạy PSO trên Rastrigin Function

```python
from src.swarm_intelligence.pso import PSO
from src.test_functions import get_test_function

# Khởi tạo test function
func = get_test_function('rastrigin', dim=10)

# Khởi tạo PSO
pso = PSO(
    n_particles=30,
    dim=10,
    max_iter=100,
    w=0.7,        # Inertia weight
    c1=1.5,       # Cognitive parameter
    c2=1.5,       # Social parameter
    bounds=func.bounds
)

# Chạy optimization
best_pos, best_score = pso.optimize(func, verbose=True)

# Lấy convergence history
history = pso.get_history()

print(f"Best position: {best_pos}")
print(f"Best score: {best_score}")
print(f"Global optimum: {func.global_optimum}")
```

### Ví dụ 2: So sánh Multiple Algorithms

```python
from src.comparison import AlgorithmComparison
from src.swarm_intelligence.pso import PSO
from src.swarm_intelligence.abc import ABC
from src.traditional_search.genetic_algorithm import GeneticAlgorithm
from src.test_functions import get_test_function

# Setup
func = get_test_function('sphere', dim=10)

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
    'GA': (GeneticAlgorithm, {
        'population_size': 30,
        'dim': 10,
        'max_iter': 100,
        'bounds': func.bounds
    })
}

# Run comparison (30 runs each)
comparison = AlgorithmComparison()
results = comparison.compare_algorithms(
    algorithms,
    func,
    n_trials=30,
    verbose=True
)

# Generate report
report = comparison.generate_report(
    results,
    objective_name="Sphere Function",
    target_score=0.0
)

print(report)

# Create comparison table
df = comparison.create_comparison_table(results)
print(df)
```

### Ví dụ 3: Logging và Export Results

```python
from src.utils.logger import ExperimentLogger
from src.utils.metrics import BenchmarkRunner

# Khởi tạo logger
logger = ExperimentLogger(log_dir="logs", results_dir="results")

# Chạy benchmark
runner = BenchmarkRunner()
algorithms = {...}  # Define như ví dụ 2

comparison_results = runner.compare_algorithms(
    algorithms,
    func,
    n_runs=30,
    verbose=True
)

# Generate comprehensive report với charts
outputs = logger.create_comparison_report(
    comparison_results,
    problem_name="Sphere_10D",
    output_prefix="experiment_001"
)

print("Generated files:")
for key, value in outputs.items():
    print(f"  {key}: {value}")

# Export data
csv_file = logger.export_csv()
json_file = logger.export_json()

print(f"Data exported to: {csv_file}, {json_file}")
```

### Ví dụ 4: Sử dụng Configuration Manager

```python
from src.utils.config import ConfigManager

# Load config
config = ConfigManager("config.yaml")

# Lấy default parameters cho PSO
pso_params = config.get_algorithm_params('PSO')
print(f"PSO params: {pso_params}")

# Override với custom values
custom_params = config.get_algorithm_params('PSO', {
    'n_particles': 50,
    'w': 0.8
})
print(f"Custom params: {custom_params}")

# Validate parameters
try:
    config.validate_params('PSO', custom_params)
    print("✓ Parameters valid")
except ValueError as e:
    print(f"✗ Invalid parameters: {e}")

# Lấy info về test function
rastrigin_info = config.get_test_function_info('rastrigin')
print(f"Rastrigin info: {rastrigin_info}")
```

### Ví dụ 5: Visualization

```python
from src.visualization import OptimizationVisualizer
import matplotlib.pyplot as plt

# Giả sử đã có results từ comparison
histories = [pso_history, abc_history, ga_history]
labels = ['PSO', 'ABC', 'GA']

# Plot convergence comparison
OptimizationVisualizer.plot_convergence(
    histories,
    labels,
    title="Convergence Comparison - Sphere Function",
    save_path="results/plots/convergence_sphere.png",
    log_scale=True
)

# Plot 3D surface
func = get_test_function('rastrigin', dim=2)
OptimizationVisualizer.plot_3d_surface(
    func,
    x_range=(-5.12, 5.12),
    y_range=(-5.12, 5.12),
    save_path="results/plots/rastrigin_surface.png"
)

# Box plot comparison
results_dict = {
    'PSO': pso_scores,
    'ABC': abc_scores,
    'GA': ga_scores
}

OptimizationVisualizer.plot_box_comparison(
    results_dict,
    title="Performance Distribution - Sphere Function",
    save_path="results/plots/boxplot_sphere.png"
)

plt.show()
```

## 📊 Hiểu các Biểu Đồ

### 1. Convergence Plot (Đồ thị Hội tụ)

- **Trục X**: Số iterations/generations
- **Trục Y**: Best score (thường dùng log scale)
- **Đường màu**: Mỗi thuật toán một màu
- **Vùng tô màu**: Standard deviation (độ ổn định)

**Cách đọc:**
- Đường xuống nhanh → Hội tụ nhanh
- Đường phẳng sớm → Bị stuck ở local optimum hoặc đã tìm được optimum
- Vùng tô màu hẹp → Ổn định, robust
- Vùng tô màu rộng → Không ổn định, phụ thuộc nhiều vào khởi tạo ngẫu nhiên

### 2. Box Plot (Biểu đồ Hộp)

- **Box (hộp)**: Chứa 50% dữ liệu giữa (Q1 đến Q3)
- **Đường giữa box**: Median
- **Whiskers (râu)**: Min và Max (hoặc 1.5*IQR)
- **Điểm lẻ**: Outliers
- **X trong box**: Mean

**Cách đọc:**
- Box hẹp → Ít biến động, ổn định
- Box rộng → Biến động nhiều
- Median gần Q1 hoặc Q3 → Phân phối lệch
- Nhiều outliers → Không ổn định

### 3. 3D Surface Plot

- **Surface màu**: Fitness landscape (màu ấm = cao, màu lạnh = thấp)
- **Particles**: Population/swarm hiện tại
- **Màu particles**: Fitness (xanh lá = tốt, đỏ = xấu)
- **Sao đỏ**: Current best
- **Sao xanh**: Global optimum

**Cách đọc:**
- Nhiều valley → Multimodal (nhiều local optima)
- Một valley → Unimodal (một optimum)
- Surface gồ ghề → Khó optimize
- Surface trơn → Dễ optimize

### 4. Parameter Sensitivity Plot

- **Trục X**: Giá trị parameter
- **Trục Y**: Performance metric
- **Error bars**: Standard deviation

**Cách đọc:**
- Đường phẳng → Parameter không ảnh hưởng nhiều
- Đường dốc → Parameter quan trọng, cần tune cẩn thận
- U-shape → Có giá trị optimal ở giữa

## 🛠️ Cấu Trúc Code

### Module Organization

```
src/
├── swarm_intelligence/      # Swarm Intelligence Algorithms
│   ├── pso.py              # Particle Swarm Optimization
│   ├── aco.py              # Ant Colony Optimization
│   ├── abc.py              # Artificial Bee Colony
│   ├── fa.py               # Firefly Algorithm
│   └── cs.py               # Cuckoo Search
│
├── traditional_search/      # Traditional Search Algorithms
│   ├── hill_climbing.py
│   ├── simulated_annealing.py
│   ├── genetic_algorithm.py
│   └── graph_search.py     # BFS, DFS, A*
│
├── discrete_problems/       # Discrete Optimization Problems
│   ├── tsp.py              # Traveling Salesman Problem
│   ├── knapsack.py         # 0/1 Knapsack Problem
│   └── graph_coloring.py   # Graph Coloring Problem
│
├── utils/                   # Utility Modules
│   ├── config.py           # Configuration Management
│   ├── logger.py           # Logging & Export
│   └── metrics.py          # Performance Metrics
│
├── test_functions.py        # Benchmark Test Functions
├── visualization.py         # Visualization Tools
└── comparison.py            # Algorithm Comparison Tools
```

### Thêm Thuật Toán Mới

Để thêm một thuật toán mới, tạo file mới theo template:

```python
"""
My New Algorithm
Description of the algorithm
"""

import numpy as np

class MyNewAlgorithm:
    """
    My New Algorithm
    
    Parameters:
    -----------
    param1 : type
        Description
    ...
    """
    
    def __init__(self, dim=10, max_iter=100, bounds=None, **kwargs):
        self.dim = dim
        self.max_iter = max_iter
        self.bounds = bounds if bounds is not None else np.array([[-100, 100]] * dim)
        
        # History tracking
        self.best_scores_history = []
        self.mean_scores_history = []
    
    def initialize(self):
        """Initialize algorithm state"""
        pass
    
    def optimize(self, objective_function, verbose=False):
        """
        Run optimization
        
        Parameters:
        -----------
        objective_function : callable
            Function to minimize
        verbose : bool
            Print progress
            
        Returns:
        --------
        best_solution : np.ndarray
            Best solution found
        best_score : float
            Best score found
        """
        self.initialize()
        
        for iteration in range(self.max_iter):
            # Your algorithm logic here
            pass
        
        return self.best_solution, self.best_score
    
    def get_history(self):
        """Get convergence history"""
        return {
            'best_scores': np.array(self.best_scores_history),
            'mean_scores': np.array(self.mean_scores_history)
        }
```

Sau đó cập nhật `config.yaml` để thêm default parameters:

```yaml
algorithms:
  MyNewAlgorithm:
    name: "My New Algorithm"
    type: "swarm"  # or "traditional"
    default_params:
      param1: value1
      param2: value2
    param_ranges:
      param1: [min, max]
      param2: [min, max]
    description: "Description of the algorithm"
```

## 📝 Configuration (config.yaml)

File `config.yaml` chứa tất cả cấu hình mặc định:

- **Algorithms**: Default parameters cho mỗi thuật toán
- **Test Functions**: Thông tin về các hàm test
- **Visualization**: Cài đặt cho plots (DPI, colors, styles)
- **Experiments**: Cài đặt cho batch experiments (n_runs, timeout, etc.)
- **Logging**: Cài đặt logging và export

Bạn có thể chỉnh sửa file này hoặc tạo file config riêng:

```python
from src.utils.config import ConfigManager

# Load custom config
config = ConfigManager("my_custom_config.yaml")

# Or modify and save
config.config['algorithms']['PSO']['default_params']['n_particles'] = 50
config.save_config("my_custom_config.yaml")
```

## 📂 Results và Logs

### Directory Structure

```
results/
├── plots/                  # Generated charts (PNG, PDF)
│   ├── convergence_*.png
│   ├── boxplot_*.png
│   └── surface_*.png
├── data/                   # Exported data
│   ├── experiments_*.csv
│   └── experiments_*.json
└── reports/                # LaTeX tables & reports
    └── table_*.tex

logs/
└── *.json                  # Individual experiment logs
```

### Log Format

Mỗi experiment được log dạng JSON:

```json
{
  "exp_id": "PSO_Sphere_20240110_123456",
  "timestamp": "2024-01-10T12:34:56",
  "algorithm": "PSO",
  "problem": "Sphere",
  "parameters": {
    "n_particles": 30,
    "max_iter": 100,
    "w": 0.7,
    "c1": 1.5,
    "c2": 1.5
  },
  "results": {
    "best_score": 0.000123,
    "runtime": 2.5,
    "iterations": 100,
    "success_rate": 0.95
  },
  "metadata": {
    "dim": 10,
    "bounds": [-100, 100]
  }
}
```

## 🔬 Advanced Usage

### 1. Parameter Sweep

```python
from src.utils.config import ConfigManager
from src.utils.logger import ExperimentLogger
import numpy as np

config = ConfigManager()
logger = ExperimentLogger()

# Sweep over w parameter for PSO
w_values = np.linspace(0.1, 1.0, 10)
results = []

for w in w_values:
    params = config.get_algorithm_params('PSO', {'w': w})
    pso = PSO(**params, dim=10, bounds=func.bounds)
    best_pos, best_score = pso.optimize(func)
    results.append(best_score)
    
    # Log experiment
    logger.log_experiment(
        algorithm="PSO",
        problem="Sphere",
        parameters=params,
        results={'best_score': best_score, 'w': w}
    )

# Visualize parameter sensitivity
from src.visualization import OptimizationVisualizer
OptimizationVisualizer.plot_parameter_sensitivity(
    w_values,
    results,
    param_name="Inertia Weight (w)",
    save_path="results/plots/pso_w_sensitivity.png"
)
```

### 2. Parallel Experiments

```python
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def run_trial(trial_id, algo_class, params, func):
    algo = algo_class(**params)
    best_pos, best_score = algo.optimize(func)
    return {'trial': trial_id, 'score': best_score}

# Run 30 trials in parallel
params = {...}
run_func = partial(run_trial, algo_class=PSO, params=params, func=func)

with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(run_func, range(30)))

print(f"Mean score: {np.mean([r['score'] for r in results])}")
```

### 3. Custom Visualization

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create custom animation
fig, ax = plt.subplots(figsize=(10, 10))

def update(frame):
    ax.clear()
    # Your custom visualization code
    # e.g., plot particles, update positions, etc.
    return ax,

anim = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
anim.save('results/plots/custom_animation.gif', writer='pillow')
```

## 🐛 Troubleshooting

### Issue 1: Import Errors

```bash
# Đảm bảo đang ở thư mục root của project
cd Co_So_AI_DoAn1_Search

# Thêm project vào PYTHONPATH (nếu cần)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows CMD
$env:PYTHONPATH += ";$(Get-Location)"      # Windows PowerShell
```

### Issue 2: Streamlit Port Already in Use

```bash
# Sử dụng port khác
streamlit run main.py --server.port 8502

# Cài đặt dependencies
pip install -r requirements.txt

# Test cài đặt
python run_simple_test.py
```

### ⚠️ Nếu gặp lỗi NumPy version conflict:

```bash
# Fix nhanh
pip uninstall numpy scipy seaborn -y
pip install numpy==1.26.4 matplotlib pandas tqdm jupyter

# Test lại
python run_simple_test.py
```

**Xem chi tiết:** [FIX_ERRORS.md](FIX_ERRORS.md)

## 📖 Hướng dẫn sử dụng

### Quick Start - 3 bước

```python
# 1. Import và setup
import numpy as np
from src.test_functions import get_test_function
from src.swarm_intelligence.pso import PSO

np.random.seed(42)
func = get_test_function('sphere', dim=10)

# 2. Khởi tạo và chạy thuật toán
pso = PSO(n_particles=30, dim=10, max_iter=100, bounds=func.bounds)
best_pos, best_score = pso.optimize(func, verbose=True)

# 3. Xem kết quả
print(f"Best score: {best_score:.6f}")
```

### Chạy Demo

```bash
python demo.py
```

### Tài liệu chi tiết

- 📘 **[QUICKSTART.md](QUICKSTART.md)** - Bắt đầu trong 5 phút
- 📗 **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Hướng dẫn đầy đủ
- 📕 **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Tổng quan project
- 📄 **[report/report_template.md](report/report_template.md)** - Template báo cáo

## ✨ Features

- ✅ **5 thuật toán Swarm Intelligence** - PSO, ACO, ABC, FA, CS
- ✅ **6 thuật toán truyền thống** - Hill Climbing, SA, GA, BFS, DFS, A*
- ✅ **4 hàm test continuous** - Sphere, Rastrigin, Rosenbrock, Ackley
- ✅ **4 bài toán discrete** - TSP, Knapsack, Graph Coloring, Path Finding
- ✅ **18+ algorithm implementations** - Comprehensive coverage
- ✅ **Visualization tools** - 3D plots, convergence curves, path visualization
- ✅ **Comparison framework** - Statistical analysis, automated reports
- ✅ **Full documentation** - Templates, guides, examples

## 🎨 Interactive Visualization Apps

### 🎬 Animated Version (XEM PARTICLES DI CHUYỂN!) ⭐⭐⭐
```bash
streamlit run app_animated.py
```
- ✨ **ANIMATION THẬT** - Xem particles di chuyển trên 3D!
- 🔵 **Real-time** - Từng bước hội tụ về optimum
- 🎨 **Beautiful** - Color-coded particles
- 📹 **Demo perfect** - Cho presentations/videos
- 🎓 **Educational** - Hiểu rõ cách algorithms work

**Xem:** `README_ANIMATED.md`

---

### 📊 Simple Version (RECOMMENDED cho báo cáo)
```bash
streamlit run app_simple.py
```
- ✅ **Gọn nhẹ** - Chỉ continuous optimization
- ✅ **All-in-one** - Tất cả plots cùng lúc
- ✅ **Compare** - Nhiều algorithms
- ✅ **Đầy đủ** - 3D surface, convergence, performance, robustness
- ✅ **Perfect** - Đáp ứng 100% yêu cầu đề bài

**Xem:** `README_SIMPLE.md`

---

### Advanced Versions (Optional):

1. **Matplotlib/Seaborn Version**
   ```bash
   streamlit run app_visualization_matplotlib.py
   ```
   - All problem types (continuous, TSP, Knapsack)
   - Real-time animation

2. **Plotly Version**
   ```bash
   streamlit run app_visualization.py
   ```
   - Interactive 3D plots
   - Zoom, pan, rotate

**Xem:** `APP_COMPARISON.md`

---

## 📊 Ví dụ So sánh Thuật toán

```python
from src.comparison import AlgorithmComparison
from src.swarm_intelligence import PSO, ABC

algorithms = {
    'PSO': (PSO, {'n_particles': 30, 'dim': 10, 'max_iter': 100, 'bounds': func.bounds}),
    'ABC': (ABC, {'n_bees': 30, 'dim': 10, 'max_iter': 100, 'bounds': func.bounds})
}

# So sánh với 10 trials
results = AlgorithmComparison.compare_algorithms(algorithms, func, n_trials=10)

# Tạo báo cáo tự động
report = AlgorithmComparison.generate_report(results, objective_name="Sphere Function")
print(report)
```

## 🎨 Visualization

```python
from src.visualization import OptimizationVisualizer

# Plot convergence curves
OptimizationVisualizer.plot_convergence(histories, labels, title="Convergence")

# Plot 3D surface
func_2d = get_test_function('rastrigin', dim=2)
OptimizationVisualizer.plot_3d_surface(func_2d, x_range=(-5, 5), y_range=(-5, 5))

# TSP visualization
OptimizationVisualizer.plot_tsp_tour(tsp, tour, title="TSP Tour")
```

## 👥 Tác giả

**Nhóm sinh viên - Đồ án 1**

| STT | MSSV | Họ và Tên |
|-----|------|-----------|
| 1 | 23122030 | Phạm Phú Hòa |
| 2 | 23122041 | Đào Sỹ Duy Minh |
| 3 | 23122044 | Trần Chí Nguyên |
| 4 | 23122048 | Nguyễn Lâm Phú Quý |

**Môn học:** CSC14003 - Cơ sở Trí tuệ Nhân tạo  
**Khoa:** Công nghệ Thông tin - ĐHKHTN TPHCM  
**Năm học:** 2024-2025

## Tài liệu tham khảo
1. Dorigo, M., & Stützle, T. (2004). Ant colony optimization.
2. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
3. Karaboga, D. (2005). An idea based on honey bee swarm for numerical optimization.
4. Yang, X. S. (2008). Firefly algorithm.
5. Yang, X. S., & Deb, S. (2009). Cuckoo search via Lévy flights.


# HƯỚNG DẪN SỬ DỤNG - Đồ án Swarm Intelligence

## 1. CÀI ĐẶT

### 1.1. Yêu cầu hệ thống

- Python 3.8 trở lên
- pip package manager

### 1.2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## 2. CẤU TRÚC PROJECT

```
Co_So_AI_search/
├── src/                          # Source code
│   ├── test_functions.py         # Các hàm test
│   ├── swarm_intelligence/       # Thuật toán swarm intelligence
│   │   ├── pso.py               # Particle Swarm Optimization
│   │   ├── aco.py               # Ant Colony Optimization
│   │   ├── abc.py               # Artificial Bee Colony
│   │   ├── fa.py                # Firefly Algorithm
│   │   └── cs.py                # Cuckoo Search
│   ├── traditional_search/       # Thuật toán truyền thống
│   │   ├── hill_climbing.py
│   │   ├── simulated_annealing.py
│   │   └── genetic_algorithm.py
│   ├── discrete_problems/        # Bài toán rời rạc
│   │   └── tsp.py               # Traveling Salesman Problem
│   ├── visualization.py          # Module visualization
│   └── comparison.py             # Module so sánh
├── notebooks/                    # Jupyter notebooks
├── results/                      # Kết quả thực nghiệm
├── report/                       # Báo cáo
├── demo.py                       # Demo script
└── README.md
```

## 3. SỬ DỤNG CƠ BẢN

### 3.1. Chạy demo nhanh

```bash
python demo.py
```

Chọn các tùy chọn demo:
1. Continuous Optimization Comparison
2. Traveling Salesman Problem (TSP)
3. 2D Function Visualization
4. Run all demos

### 3.2. Sử dụng từng thuật toán riêng lẻ

#### 3.2.1. PSO (Particle Swarm Optimization)

```python
from src.test_functions import get_test_function
from src.swarm_intelligence.pso import PSO

# Tạo test function
func = get_test_function('sphere', dim=10)

# Khởi tạo PSO
pso = PSO(
    n_particles=30,      # Số lượng particles
    dim=10,              # Số chiều
    max_iter=100,        # Số iterations
    w=0.7,               # Inertia weight
    c1=1.5,              # Cognitive parameter
    c2=1.5,              # Social parameter
    bounds=func.bounds   # Bounds của search space
)

# Chạy optimization
best_position, best_score = pso.optimize(func, verbose=True)

print(f"Best position: {best_position}")
print(f"Best score: {best_score}")

# Lấy history để vẽ convergence
history = pso.get_history()
print(f"Convergence history: {history['best_scores']}")
```

#### 3.2.2. ACO (Ant Colony Optimization)

```python
from src.swarm_intelligence.aco import ACO

aco = ACO(
    n_ants=30,
    dim=10,
    max_iter=100,
    archive_size=50,
    q=0.5,
    xi=0.85,
    bounds=func.bounds
)

best_solution, best_score = aco.optimize(func, verbose=True)
```

#### 3.2.3. ABC (Artificial Bee Colony)

```python
from src.swarm_intelligence.abc import ABC

abc = ABC(
    n_bees=30,
    dim=10,
    max_iter=100,
    limit=20,
    bounds=func.bounds
)

best_solution, best_score = abc.optimize(func, verbose=True)
```

#### 3.2.4. Firefly Algorithm

```python
from src.swarm_intelligence.fa import FireflyAlgorithm

fa = FireflyAlgorithm(
    n_fireflies=30,
    dim=10,
    max_iter=100,
    alpha=0.5,
    beta0=1.0,
    gamma=1.0,
    bounds=func.bounds
)

best_solution, best_score = fa.optimize(func, verbose=True)
```

#### 3.2.5. Cuckoo Search

```python
from src.swarm_intelligence.cs import CuckooSearch

cs = CuckooSearch(
    n_nests=30,
    dim=10,
    max_iter=100,
    pa=0.25,
    beta=1.5,
    bounds=func.bounds
)

best_solution, best_score = cs.optimize(func, verbose=True)
```

### 3.3. Thuật toán truyền thống

#### 3.3.1. Hill Climbing

```python
from src.traditional_search.hill_climbing import HillClimbing

hc = HillClimbing(
    dim=10,
    max_iter=100,
    step_size=1.0,
    bounds=func.bounds
)

best_solution, best_score = hc.optimize(func, verbose=True)
```

#### 3.3.2. Simulated Annealing

```python
from src.traditional_search.simulated_annealing import SimulatedAnnealing

sa = SimulatedAnnealing(
    dim=10,
    max_iter=1000,
    initial_temp=100.0,
    final_temp=1e-3,
    alpha=0.95,
    bounds=func.bounds
)

best_solution, best_score = sa.optimize(func, verbose=True)
```

#### 3.3.3. Genetic Algorithm

```python
from src.traditional_search.genetic_algorithm import GeneticAlgorithm

ga = GeneticAlgorithm(
    population_size=50,
    dim=10,
    max_iter=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    bounds=func.bounds
)

best_solution, best_score = ga.optimize(func, verbose=True)
```

## 4. SO SÁNH THUẬT TOÁN

### 4.1. So sánh cơ bản

```python
from src.comparison import AlgorithmComparison
from src.swarm_intelligence.pso import PSO
from src.swarm_intelligence.abc import ABC

# Định nghĩa các thuật toán cần so sánh
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
    })
}

# Chạy so sánh (10 trials cho mỗi thuật toán)
results = AlgorithmComparison.compare_algorithms(
    algorithms,
    func,
    n_trials=10,
    verbose=False
)

# Tạo báo cáo
report = AlgorithmComparison.generate_report(
    results,
    objective_name="Sphere Function",
    target_score=0.0
)

print(report)

# Tạo comparison table
df = AlgorithmComparison.create_comparison_table(results)
print(df)
```

### 4.2. Tính toán metrics

```python
# Calculate statistics
stats = AlgorithmComparison.calculate_statistics(results['PSO'])
print(f"Mean score: {stats['mean_score']}")
print(f"Std score: {stats['std_score']}")

# Robustness (coefficient of variation)
cv = AlgorithmComparison.robustness_metric(results['PSO'])
print(f"Robustness (CV): {cv}")

# Success rate
sr = AlgorithmComparison.success_rate(results['PSO'], target_score=0.0, tolerance=0.01)
print(f"Success rate: {sr}%")
```

## 5. VISUALIZATION

### 5.1. Plot convergence curves

```python
from src.visualization import OptimizationVisualizer

# Lấy histories từ các thuật toán
histories = [results['PSO'][0]['history'], results['ABC'][0]['history']]
labels = ['PSO', 'ABC']

# Plot convergence
OptimizationVisualizer.plot_convergence(
    histories,
    labels,
    title="Convergence Comparison",
    log_scale=True,
    save_path="results/convergence.png"
)
```

### 5.2. Plot 3D surface của test function

```python
# Visualize 2D function
func_2d = get_test_function('rastrigin', dim=2)

OptimizationVisualizer.plot_3d_surface(
    func_2d,
    x_range=(-5, 5),
    y_range=(-5, 5),
    n_points=100,
    save_path="results/rastrigin_surface.png"
)
```

### 5.3. Box plot comparison

```python
# So sánh phân bố kết quả
results_dict = {
    'PSO': [r['best_score'] for r in results['PSO']],
    'ABC': [r['best_score'] for r in results['ABC']]
}

OptimizationVisualizer.plot_box_comparison(
    results_dict,
    title="Algorithm Comparison",
    ylabel="Best Score",
    save_path="results/box_comparison.png"
)
```

### 5.4. Parameter sensitivity analysis

```python
# Test different values of a parameter
param_values = [10, 20, 30, 40, 50]
results_list = []

for n in param_values:
    pso = PSO(n_particles=n, dim=10, max_iter=100, bounds=func.bounds)
    _, score = pso.optimize(func)
    results_list.append(score)

OptimizationVisualizer.plot_parameter_sensitivity(
    param_values,
    results_list,
    param_name="Number of Particles",
    metric_name="Best Score",
    save_path="results/sensitivity_particles.png"
)
```

## 6. BÀI TOÁN TSP

### 6.1. Tạo và giải TSP

```python
from src.discrete_problems import TSP, TSPSolver

# Tạo TSP instance
tsp = TSP(n_cities=20, seed=42)

# Giải bằng Nearest Neighbor
nn_tour, nn_distance = TSPSolver.nearest_neighbor(tsp)
print(f"Nearest Neighbor: {nn_distance:.2f}")

# Cải thiện bằng 2-opt
opt_tour, opt_distance = TSPSolver.two_opt(tsp)
print(f"2-opt: {opt_distance:.2f}")

# Giải bằng Genetic Algorithm
ga_tour, ga_distance, ga_history = TSPSolver.genetic_algorithm_tsp(
    tsp,
    population_size=100,
    max_iter=500
)
print(f"GA: {ga_distance:.2f}")

# Visualize tour
OptimizationVisualizer.plot_tsp_tour(
    tsp,
    ga_tour,
    title="Best TSP Tour (GA)",
    save_path="results/tsp_tour.png"
)
```

## 7. TEST FUNCTIONS

### 7.1. Các hàm test có sẵn

```python
from src.test_functions import get_test_function

# Sphere Function
sphere = get_test_function('sphere', dim=10)

# Rastrigin Function
rastrigin = get_test_function('rastrigin', dim=10)

# Rosenbrock Function
rosenbrock = get_test_function('rosenbrock', dim=10)

# Ackley Function
ackley = get_test_function('ackley', dim=10)

# Sử dụng
x = np.random.randn(10)
value = sphere(x)
print(f"Sphere(x) = {value}")
```

### 7.2. Thông tin về test functions

```python
# Lấy thông tin
print(f"Bounds: {func.bounds}")
print(f"Global optimum: {func.global_optimum}")
print(f"Dimension: {func.dim}")
```

## 8. TIPS VÀ BEST PRACTICES

### 8.1. Parameter tuning

- **PSO:**
  - `n_particles`: 20-50 (tùy vào độ phức tạp)
  - `w`: 0.7-0.9 (càng nhỏ càng exploitation)
  - `c1, c2`: 1.5-2.0 (cân bằng personal vs social)

- **ABC:**
  - `n_bees`: 30-100
  - `limit`: 10-50 (số lần không cải thiện trước khi abandon)

- **FA:**
  - `alpha`: 0.2-0.5 (randomization)
  - `beta0`: 1.0 (attractiveness)
  - `gamma`: 0.5-2.0 (light absorption)

### 8.2. Chọn thuật toán phù hợp

| Bài toán | Thuật toán đề xuất |
|----------|-------------------|
| Continuous, unimodal | PSO, Hill Climbing |
| Continuous, multimodal | FA, CS, SA |
| Discrete (TSP, scheduling) | ACO, GA |
| High-dimensional | ABC, PSO |

### 8.3. Performance optimization

- Giảm `max_iter` khi test
- Sử dụng `verbose=False` khi chạy nhiều trials
- Parallel execution (có thể implement thêm)

## 9. TROUBLESHOOTING

### 9.1. Lỗi thường gặp

**Lỗi: ModuleNotFoundError**
```bash
# Chạy từ thư mục gốc
cd Co_So_AI_search
python demo.py

# Hoặc thêm path
export PYTHONPATH="${PYTHONPATH}:/path/to/Co_So_AI_search"
```

**Lỗi: Algorithm không hội tụ**
- Tăng `max_iter`
- Điều chỉnh parameters
- Thử thuật toán khác

**Kết quả không ổn định**
- Tăng số trials
- Fixed random seed: `np.random.seed(42)`

## 10. EXAMPLES

### 10.1. Complete example

```python
import numpy as np
from src.test_functions import get_test_function
from src.swarm_intelligence.pso import PSO
from src.visualization import OptimizationVisualizer

# Set seed for reproducibility
np.random.seed(42)

# Create test function
func = get_test_function('rastrigin', dim=10)

# Initialize PSO
pso = PSO(
    n_particles=30,
    dim=10,
    max_iter=100,
    bounds=func.bounds
)

# Run optimization
best_pos, best_score = pso.optimize(func, verbose=True)

# Print results
print(f"\nBest position: {best_pos}")
print(f"Best score: {best_score:.6f}")
print(f"Global optimum: {func.global_optimum}")

# Plot convergence
history = pso.get_history()
OptimizationVisualizer.plot_convergence(
    [history],
    ['PSO'],
    title="PSO on Rastrigin Function"
)
```

## 11. EXPORT RESULTS

### 11.1. Save to CSV

```python
import pandas as pd

# Save comparison table
df = AlgorithmComparison.create_comparison_table(results)
df.to_csv('results/comparison.csv', index=False)
```

### 11.2. Save figures

```python
# Tất cả visualization functions đều có parameter save_path
OptimizationVisualizer.plot_convergence(
    histories,
    labels,
    save_path="results/convergence.png"
)
```

---

## LIÊN HỆ VÀ HỖ TRỢ

Nếu có vấn đề hoặc câu hỏi, vui lòng:
- Mở issue trên GitHub
- Liên hệ giảng viên
- Tham khảo tài liệu tham khảo trong báo cáo

---

**Chúc các bạn thực hiện đồ án thành công!**


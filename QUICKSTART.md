# QUICKSTART GUIDE - 5 phút bắt đầu

## 🚀 Setup trong 3 bước

### Bước 1: Cài đặt dependencies

```bash
# Recommended: NumPy 1.x để tránh xung đột
pip install numpy==1.26.4 matplotlib pandas tqdm jupyter
```

Hoặc:

```bash
pip install -r requirements.txt
```

### Bước 2: Test cài đặt

```bash
# Test nhanh (không cần visualization)
python run_simple_test.py
```

### Bước 3: Chạy demo

```bash
python demo.py
```

---

## ⚠️ GẶP LỖI?

### Lỗi NumPy version conflict:

```bash
pip uninstall numpy scipy seaborn -y
pip install numpy==1.26.4 matplotlib pandas tqdm jupyter
python run_simple_test.py
```

**Chi tiết:** Xem [FIX_ERRORS.md](FIX_ERRORS.md)

---

## 🎯 3 Ví dụ cơ bản

### 1️⃣ Chạy một thuật toán đơn giản

```python
import numpy as np
from src.test_functions import get_test_function
from src.swarm_intelligence.pso import PSO

# Setup
np.random.seed(42)
func = get_test_function('sphere', dim=10)

# Run PSO
pso = PSO(n_particles=30, dim=10, max_iter=100, bounds=func.bounds)
best_pos, best_score = pso.optimize(func, verbose=True)

print(f"\n✓ Best score: {best_score:.6f}")
print(f"✓ Target (global optimum): {func.global_optimum}")
```

**Output mong đợi:**
```
Iteration 10/100: Best = 5.234567, Mean = 12.345678
Iteration 20/100: Best = 2.345678, Mean = 8.234567
...
✓ Best score: 0.001234
✓ Target (global optimum): 0
```

---

### 2️⃣ So sánh 2 thuật toán

```python
from src.comparison import AlgorithmComparison
from src.swarm_intelligence import PSO, ABC

# Define algorithms
algorithms = {
    'PSO': (PSO, {'n_particles': 30, 'dim': 10, 'max_iter': 50, 'bounds': func.bounds}),
    'ABC': (ABC, {'n_bees': 30, 'dim': 10, 'max_iter': 50, 'bounds': func.bounds})
}

# Compare (5 trials each)
results = AlgorithmComparison.compare_algorithms(algorithms, func, n_trials=5)

# Show report
report = AlgorithmComparison.generate_report(results, objective_name="Sphere Function")
print(report)
```

**Output:** Bảng so sánh chi tiết với mean, std, best, worst, time

---

### 3️⃣ Visualize kết quả

```python
from src.visualization import OptimizationVisualizer

# Get histories
histories = [results['PSO'][0]['history'], results['ABC'][0]['history']]
labels = ['PSO', 'ABC']

# Plot convergence
OptimizationVisualizer.plot_convergence(
    histories, 
    labels,
    title="PSO vs ABC Convergence"
)

# Plot 3D function surface
func_2d = get_test_function('rastrigin', dim=2)
OptimizationVisualizer.plot_3d_surface(func_2d, x_range=(-5, 5), y_range=(-5, 5))
```

**Output:** Matplotlib plots hiển thị

---

## 📊 Test tất cả 4 hàm test

```python
functions = ['sphere', 'rastrigin', 'rosenbrock', 'ackley']

for func_name in functions:
    print(f"\n{'='*50}")
    print(f"Testing {func_name.upper()}")
    print('='*50)
    
    func = get_test_function(func_name, dim=10)
    pso = PSO(n_particles=30, dim=10, max_iter=50, bounds=func.bounds)
    _, score = pso.optimize(func, verbose=False)
    
    print(f"✓ Best score: {score:.6f}")
    print(f"✓ Global optimum: {func.global_optimum}")
```

---

## 🗺️ Giải TSP

```python
from src.discrete_problems import TSP, TSPSolver
from src.visualization import OptimizationVisualizer

# Create TSP
tsp = TSP(n_cities=20, seed=42)

# Solve with different methods
nn_tour, nn_dist = TSPSolver.nearest_neighbor(tsp)
opt_tour, opt_dist = TSPSolver.two_opt(tsp)
ga_tour, ga_dist, history = TSPSolver.genetic_algorithm_tsp(tsp, max_iter=100)

print(f"Nearest Neighbor: {nn_dist:.2f}")
print(f"2-opt: {opt_dist:.2f}")
print(f"Genetic Algorithm: {ga_dist:.2f}")
print(f"Improvement: {(nn_dist - ga_dist)/nn_dist*100:.1f}%")

# Visualize
OptimizationVisualizer.plot_tsp_tour(tsp, ga_tour, title="Best TSP Tour")
```

---

## 🎨 Tất cả thuật toán có sẵn

### Swarm Intelligence
```python
from src.swarm_intelligence import PSO, ACO, ABC, FireflyAlgorithm, CuckooSearch

# All have same interface:
algorithm = PSO(...)  # or ACO, ABC, FA, CS
best_solution, best_score = algorithm.optimize(objective_function)
history = algorithm.get_history()
```

### Traditional Search
```python
from src.traditional_search import HillClimbing, SimulatedAnnealing, GeneticAlgorithm

# Same interface
algorithm = HillClimbing(...)  # or SA, GA
best_solution, best_score = algorithm.optimize(objective_function)
```

---

## 📝 Cho báo cáo - Complete workflow

```python
import numpy as np
from src.test_functions import get_test_function
from src.swarm_intelligence import PSO, ACO, ABC, FireflyAlgorithm, CuckooSearch
from src.traditional_search import HillClimbing, SimulatedAnnealing, GeneticAlgorithm
from src.comparison import AlgorithmComparison
from src.visualization import OptimizationVisualizer

# 1. Setup
np.random.seed(42)
func = get_test_function('rastrigin', dim=10)

# 2. Define all algorithms
algorithms = {
    'PSO': (PSO, {'n_particles': 30, 'dim': 10, 'max_iter': 100, 'bounds': func.bounds}),
    'ACO': (ACO, {'n_ants': 30, 'dim': 10, 'max_iter': 100, 'bounds': func.bounds}),
    'ABC': (ABC, {'n_bees': 30, 'dim': 10, 'max_iter': 100, 'bounds': func.bounds}),
    'FA': (FireflyAlgorithm, {'n_fireflies': 30, 'dim': 10, 'max_iter': 100, 'bounds': func.bounds}),
    'CS': (CuckooSearch, {'n_nests': 30, 'dim': 10, 'max_iter': 100, 'bounds': func.bounds}),
    'HC': (HillClimbing, {'dim': 10, 'max_iter': 100, 'bounds': func.bounds}),
    'SA': (SimulatedAnnealing, {'dim': 10, 'max_iter': 1000, 'bounds': func.bounds}),
    'GA': (GeneticAlgorithm, {'population_size': 50, 'dim': 10, 'max_iter': 100, 'bounds': func.bounds})
}

# 3. Run comparison (30 trials)
print("Running comparison (this may take a few minutes)...")
results = AlgorithmComparison.compare_algorithms(algorithms, func, n_trials=30)

# 4. Generate report
report = AlgorithmComparison.generate_report(
    results,
    objective_name="Rastrigin Function (dim=10)",
    target_score=0.0
)
print(report)

# Save to file
with open('results/report.txt', 'w') as f:
    f.write(report)

# 5. Create comparison table
df = AlgorithmComparison.create_comparison_table(results)
print("\n", df)
df.to_csv('results/comparison_table.csv', index=False)

# 6. Visualizations
# Convergence plot
histories = [results[name][0]['history'] for name in algorithms.keys()]
labels = list(algorithms.keys())

OptimizationVisualizer.plot_convergence(
    histories, labels,
    title="Algorithm Convergence - Rastrigin Function",
    log_scale=True,
    save_path='results/convergence.png'
)

# Box plot
results_dict = {name: [r['best_score'] for r in res] 
                for name, res in results.items()}

OptimizationVisualizer.plot_box_comparison(
    results_dict,
    title="Algorithm Comparison - Rastrigin Function",
    ylabel="Best Score",
    save_path='results/box_comparison.png'
)

print("\n✓ All results saved to results/ folder")
print("✓ Ready for report writing!")
```

**Kết quả:**
- `results/report.txt` - Full text report
- `results/comparison_table.csv` - Bảng so sánh
- `results/convergence.png` - Biểu đồ hội tụ
- `results/box_comparison.png` - Box plot

---

## 🔍 Troubleshooting

### ❌ Import Error
```bash
# Make sure you're in project root directory
cd Co_So_AI_search
python your_script.py
```

### ❌ Algorithm không hội tụ
```python
# Tăng max_iter
pso = PSO(..., max_iter=500)  # instead of 100

# Hoặc thử function dễ hơn
func = get_test_function('sphere', dim=10)  # easier than rastrigin
```

### ❌ Plot không hiển thị
```python
# Add at the end
import matplotlib.pyplot as plt
plt.show()
```

---

## 📚 Tài liệu chi tiết

- **Full documentation:** `USAGE_GUIDE.md`
- **Report template:** `report/report_template.md`
- **Project summary:** `PROJECT_SUMMARY.md`
- **Demo script:** `demo.py`

---

## ✅ Next Steps

1. ✓ Chạy quickstart examples
2. ✓ Chạy `demo.py` để xem tất cả features
3. ✓ Đọc `USAGE_GUIDE.md` để biết chi tiết
4. ✓ Sử dụng `report_template.md` để viết báo cáo
5. ✓ Run experiments và thu thập kết quả
6. ✓ Tạo visualizations
7. ✓ Viết báo cáo

---

**🎉 Chúc bạn thành công với đồ án!**


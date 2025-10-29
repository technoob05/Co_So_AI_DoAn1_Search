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

### 3 Thuật toán tìm kiếm truyền thống (để so sánh)
1. **Hill Climbing** - Leo đồi
2. **Simulated Annealing** - Mô phỏng ủ kim loại
3. **Genetic Algorithm** - Thuật toán di truyền

### Hàm test
#### Continuous Optimization:
- Sphere Function
- Rastrigin Function
- Rosenbrock Function
- Ackley Function

#### Discrete Optimization:
- Traveling Salesman Problem (TSP)

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
│   │   └── genetic_algorithm.py
│   ├── discrete_problems/        # Bài toán rời rạc
│   │   └── tsp.py
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

```bash
# Clone repository (hoặc download ZIP)
cd Co_So_AI_search

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
- ✅ **3 thuật toán truyền thống** - Hill Climbing, SA, GA
- ✅ **4 hàm test continuous** - Sphere, Rastrigin, Rosenbrock, Ackley
- ✅ **Bài toán TSP** - với 3 phương pháp giải
- ✅ **Visualization tools** - 3D plots, convergence curves, comparisons
- ✅ **Comparison framework** - Statistical analysis, automated reports
- ✅ **Full documentation** - Templates, guides, examples

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

**Nhóm sinh viên**
- MSSV: __________ - Họ tên: __________
- MSSV: __________ - Họ tên: __________
- MSSV: __________ - Họ tên: __________

*(Cập nhật thông tin nhóm của bạn tại đây)*

## Tài liệu tham khảo
1. Dorigo, M., & Stützle, T. (2004). Ant colony optimization.
2. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
3. Karaboga, D. (2005). An idea based on honey bee swarm for numerical optimization.
4. Yang, X. S. (2008). Firefly algorithm.
5. Yang, X. S., & Deb, S. (2009). Cuckoo search via Lévy flights.


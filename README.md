# 🔬 Đồ Án 1 - Thuật toán Swarm Intelligence

## 📚 Thông Tin Môn Học

- **Môn học:** CSC14003 - Cơ sở Trí tuệ Nhân tạo
- **Khoa:** Công nghệ Thông tin - ĐHKHTN TPHCM
- **Năm học:** 2024-2025

## 📝 Mô Tả Dự Án

Dự án implement, phân tích và so sánh các thuật toán swarm intelligence với các thuật toán tìm kiếm truyền thống.

## 🎯 Nội Dung Dự Án

### 🐝 5 Thuật Toán Swarm Intelligence

1. **PSO** - Particle Swarm Optimization
2. **ACO** - Ant Colony Optimization
3. **ABC** - Artificial Bee Colony
4. **FA** - Firefly Algorithm
5. **CS** - Cuckoo Search

### 🔍 6 Thuật Toán Tìm Kiếm Truyền Thống

1. **Hill Climbing**
2. **Simulated Annealing**
3. **Genetic Algorithm**
4. **BFS** - Breadth-First Search
5. **DFS** - Depth-First Search
6. **A*** - A* Search

### 📊 Bài Toán Test

#### 🎨 Continuous Optimization (4 hàm):

- **Sphere Function** - Hàm đơn giản, unimodal
- **Rastrigin Function** - Hàm phức tạp, nhiều local optima
- **Rosenbrock Function** - Hàm valley hẹp
- **Ackley Function** - Hàm nhiều local optima

#### 🔢 Discrete Optimization (4 bài toán):

- **TSP** - Traveling Salesman Problem
- **Knapsack** - 0/1 Knapsack Problem
- **Graph Coloring** - Graph Coloring Problem
- **Path Finding** - GridWorld (BFS/DFS/A*)

## 📁 Cấu trúc thư mục

```
Co_So_AI_DoAn1_Search/
├── main.py                      # Streamlit app chính
├── src/
│   ├── swarm_intelligence/      # Thuật toán swarm
│   │   ├── pso.py, aco.py, abc.py, fa.py, cs.py
│   ├── traditional_search/      # Thuật toán truyền thống
│   │   ├── hill_climbing.py, simulated_annealing.py
│   │   ├── genetic_algorithm.py, graph_search.py
│   ├── discrete_problems/       # Bài toán rời rạc
│   │   ├── tsp.py, knapsack.py, graph_coloring.py
│   ├── test_functions.py        # Benchmark functions
│   ├── visualization.py
│   ├── visualization_discrete.py
│   └── comparison.py
├── assets/                      # Hình ảnh swarm agents
├── notebooks/                   # Jupyter notebooks
├── results/                     # Kết quả experiments
├── requirements.txt
└── README.md
```

## 🚀 Cài Đặt & Chạy

### 📥 Clone Repository

```bash
git clone https://github.com/your-repo/Co_So_AI_DoAn1_Search.git
cd Co_So_AI_DoAn1_Search
```

### ⚙️ Cài Đặt Dependencies

#### 🐍 Cách 1: Conda (Khuyến nghị)

```bash
conda env create -f environment.yml
conda activate co_so_ai_doan1_search
```

#### 📦 Cách 2: pip

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

### ▶️ Chạy Ứng Dụng

```bash
streamlit run main.py
```

Ứng dụng sẽ mở tại: http://localhost:8501

## 📱 Sử Dụng Ứng Dụng

### 🎬 Tab 1: Visualization & Demo

Visualize thuật toán chạy real-time với 3D surface plot và animation.

**Cách sử dụng:**
1. Chọn bài toán (Continuous/Discrete)
2. Chọn thuật toán (Swarm Intelligence/Traditional)
3. Điều chỉnh tham số (population size, iterations, etc.)
4. Nhấn "Run Animation"

**Visualization:**
- 🐟 **PSO**: Fish (cá)
- 🐜 **ACO**: Ant (kiến)
- 🐝 **ABC**: Bee (ong)
- ✨ **Firefly**: Firefly (đom đóm)
- 🐦 **Cuckoo**: Cuckoo (chim cu-cu)
- 🔴 **Best solution**: Sao đỏ
- 🟢 **Global optimum**: Sao xanh

### 📊 Tab 2: Comparison Dashboard

So sánh nhiều thuật toán với multiple runs:
- Convergence curves với std bands
- Performance metrics (mean, std, best, worst)
- Box plots (robustness analysis)
- Export results (CSV, JSON, LaTeX)

### ℹ️ Tab 3: Algorithm Info

Thông tin chi tiết về từng thuật toán:
- Mô tả cách hoạt động
- Tham số và ý nghĩa
- Use cases phù hợp

## 📊 Hiểu Biểu Đồ

### 📈 Convergence Plot
- **Trục X**: Iterations
- **Trục Y**: Best score (log scale)
- **Đường xuống nhanh**: Hội tụ nhanh
- **Vùng tô màu hẹp**: Ổn định

### 📦 Box Plot
- **Box**: 50% dữ liệu giữa (Q1-Q3)
- **Đường giữa**: Median
- **X**: Mean
- **Box hẹp**: Ít biến động

### 🗻 3D Surface
- **Surface màu**: Fitness landscape
- **Agents**: Population/swarm
- **Sao đỏ**: Current best
- **Sao xanh**: Global optimum

## 🎯 Ví Dụ Sử Dụng

### Chạy PSO trên Rastrigin Function

```python
from src.swarm_intelligence.pso import PSO
from src.test_functions import get_test_function

# Setup
func = get_test_function('rastrigin', dim=10)
pso = PSO(n_particles=30, dim=10, max_iter=100, bounds=func.bounds)

# Run
best_pos, best_score = pso.optimize(func, verbose=True)

print(f"Best score: {best_score}")
print(f"Global optimum: {func.global_optimum}")
```

## ✨ Tính Năng Nổi Bật

- ✅ **5 thuật toán Swarm Intelligence** với visualization đẹp
- ✅ **6 thuật toán truyền thống** đầy đủ
- ✅ **8 bài toán test** (4 continuous + 4 discrete)
- ✅ **Real-time 3D animation** với hình ảnh agents sinh động
- ✅ **Comparison framework** với statistical analysis
- ✅ **Export results** (CSV, JSON, LaTeX, PNG)

## 👥 Tác giả

**Nhóm sinh viên - Đồ án 1**

| STT | MSSV     | Họ và Tên           |
| --- | -------- | ------------------- |
| 1   | 23122030 | Phạm Phú Hòa        |
| 2   | 23122041 | Đào Sỹ Duy Minh     |
| 3   | 23122044 | Trần Chí Nguyên     |
| 4   | 23122048 | Nguyễn Lâm Phú Quý |

**Môn học:** CSC14003 - Cơ sở Trí tuệ Nhân tạo  
**Khoa:** Công nghệ Thông tin - ĐHKHTN TPHCM  
**Năm học:** 2024-2025

## 📚 Tài Liệu Tham Khảo

1. Dorigo, M., & Stützle, T. (2004). Ant colony optimization.
2. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
3. Karaboga, D. (2005). An idea based on honey bee swarm for numerical optimization.
4. Yang, X. S. (2008). Firefly algorithm.
5. Yang, X. S., & Deb, S. (2009). Cuckoo search via Lévy flights.

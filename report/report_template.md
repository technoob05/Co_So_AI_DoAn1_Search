# BÁO CÁO ĐỒ ÁN 1: THUẬT TOÁN SWARM INTELLIGENCE

**Môn học:** CSC14003 - Cơ sở Trí tuệ Nhân tạo  
**Khoa:** Công nghệ Thông tin - ĐHKHTN TPHCM

---

## 1. THÔNG TIN NHÓM

| MSSV | Họ và Tên | Email | Ghi chú |
|------|-----------|-------|---------|
| 12345678 | Nguyễn Văn A | nguyenvana@example.com | Nhóm trưởng |
| 87654321 | Trần Thị B | tranthib@example.com | |
| ... | ... | ... | |

---

## 2. PHÂN CÔNG CÔNG VIỆC

| Thành viên | Công việc được giao | Tỷ lệ hoàn thành (%) |
|------------|---------------------|----------------------|
| Nguyễn Văn A | - Implement PSO, ACO<br>- Viết báo cáo phần lý thuyết<br>- Tạo visualization | 100% |
| Trần Thị B | - Implement ABC, FA, CS<br>- Thực nghiệm và so sánh<br>- Viết báo cáo kết quả | 100% |
| ... | ... | ... |

**Ghi chú:** 
- Tỷ lệ hoàn thành được đánh giá dựa trên công việc đã giao
- Mỗi thành viên cần mô tả chi tiết công việc đã làm

---

## 3. TỰ ĐÁNH GIÁ MỨC ĐỘ HOÀN THÀNH

### 3.1. Yêu cầu bắt buộc

| STT | Yêu cầu | Hoàn thành | Ghi chú |
|-----|---------|-----------|---------|
| 1 | Implement 5 thuật toán Swarm Intelligence | ☑ | PSO, ACO, ABC, FA, CS |
| 2 | So sánh với 3 thuật toán truyền thống | ☑ | Hill Climbing, SA, GA |
| 3 | Test trên hàm Continuous | ☑ | Sphere, Rastrigin, Rosenbrock, Ackley |
| 4 | Test trên bài toán Discrete | ☑ | TSP |
| 5 | Visualization | ☑ | 3D surface, convergence plots |
| 6 | Báo cáo đầy đủ | ☑ | Lý thuyết + thực nghiệm |

### 3.2. Yêu cầu nâng cao (nếu có)

- [ ] Parameter sensitivity analysis
- [ ] Statistical significance testing
- [ ] Additional test problems (Knapsack, Graph Coloring)
- [ ] Advanced visualization techniques
- [ ] Hybrid algorithms

---

## 4. MÔ TẢ CHI TIẾT CÁC THUẬT TOÁN

### 4.1. Particle Swarm Optimization (PSO)

#### 4.1.1. Giới thiệu

Particle Swarm Optimization (PSO) là thuật toán tối ưu hóa lấy cảm hứng từ hành vi bầy đàn của chim và cá. Được phát triển bởi Kennedy và Eberhart (1995).

#### 4.1.2. Nguyên lý hoạt động

PSO mô phỏng hành vi di chuyển của một bầy đàn trong không gian tìm kiếm, trong đó:
- Mỗi hạt (particle) đại diện cho một giải pháp tiềm năng
- Hạt có vị trí và vận tốc
- Hạt di chuyển dựa trên kinh nghiệm cá nhân và kinh nghiệm của bầy đàn

#### 4.1.3. Công thức toán học

**Cập nhật vận tốc:**
```
v_i(t+1) = w * v_i(t) + c1 * r1 * (pbest_i - x_i(t)) + c2 * r2 * (gbest - x_i(t))
```

**Cập nhật vị trí:**
```
x_i(t+1) = x_i(t) + v_i(t+1)
```

Trong đó:
- `v_i`: vận tốc của hạt i
- `x_i`: vị trí của hạt i
- `w`: hệ số quán tính (inertia weight)
- `c1, c2`: hệ số nhận thức và xã hội
- `r1, r2`: số ngẫu nhiên trong [0, 1]
- `pbest_i`: vị trí tốt nhất cá nhân của hạt i
- `gbest`: vị trí tốt nhất toàn cục của bầy đàn

#### 4.1.4. Thuật toán

```
1. Khởi tạo bầy đàn với vị trí và vận tốc ngẫu nhiên
2. Đánh giá fitness của mỗi hạt
3. Lặp cho đến khi đạt điều kiện dừng:
   a. Cập nhật pbest của mỗi hạt
   b. Cập nhật gbest của bầy đàn
   c. Cập nhật vận tốc và vị trí của mỗi hạt
   d. Đánh giá fitness
4. Trả về gbest
```

#### 4.1.5. Ưu điểm

- Đơn giản, dễ implement
- Ít tham số cần điều chỉnh
- Hiệu quả với bài toán continuous optimization
- Hội tụ nhanh

#### 4.1.6. Nhược điểm

- Có thể bị kẹt ở local optimum
- Kém hiệu quả với bài toán discrete
- Phụ thuộc vào tham số

#### 4.1.7. Các tham số quan trọng

| Tham số | Giá trị đề xuất | Ý nghĩa |
|---------|-----------------|---------|
| n_particles | 20-50 | Số lượng hạt trong bầy đàn |
| w | 0.7-0.9 | Hệ số quán tính |
| c1 | 1.5-2.0 | Hệ số nhận thức (cognitive) |
| c2 | 1.5-2.0 | Hệ số xã hội (social) |

#### 4.1.8. Ví dụ minh họa

*[Thêm hình ảnh visualization của PSO]*

---

### 4.2. Ant Colony Optimization (ACO)

#### 4.2.1. Giới thiệu

ACO lấy cảm hứng từ hành vi tìm kiếm thức ăn của đàn kiến thông qua pheromone trails.

#### 4.2.2. Nguyên lý hoạt động

*[Mô tả chi tiết]*

#### 4.2.3. Công thức toán học

*[Công thức]*

#### 4.2.4. Thuật toán

*[Pseudo-code]*

#### 4.2.5. Ưu - Nhược điểm

*[Phân tích]*

---

### 4.3. Artificial Bee Colony (ABC)

*[Tương tự như PSO]*

---

### 4.4. Firefly Algorithm (FA)

*[Tương tự như PSO]*

---

### 4.5. Cuckoo Search (CS)

*[Tương tự như PSO]*

---

## 5. THUẬT TOÁN TÌM KIẾM TRUYỀN THỐNG

### 5.1. Hill Climbing

*[Mô tả chi tiết]*

### 5.2. Simulated Annealing

*[Mô tả chi tiết]*

### 5.3. Genetic Algorithm

*[Mô tả chi tiết]*

---

## 6. HÀM TEST VÀ BÀI TOÁN THỰC NGHIỆM

### 6.1. Hàm Continuous Optimization

#### 6.1.1. Sphere Function

**Công thức:**
```
f(x) = Σ(x_i²)
```

**Đặc điểm:**
- Global minimum: f(0,...,0) = 0
- Domain: [-100, 100]^d
- Unimodal, convex
- Dễ tối ưu hóa

**Visualization:**

*[Thêm hình 3D surface và contour plot]*

---

#### 6.1.2. Rastrigin Function

**Công thức:**
```
f(x) = 10d + Σ[x_i² - 10cos(2πx_i)]
```

**Đặc điểm:**
- Global minimum: f(0,...,0) = 0
- Domain: [-5.12, 5.12]^d
- Highly multimodal
- Nhiều local minima
- Khó tối ưu hóa

---

#### 6.1.3. Rosenbrock Function

*[Tương tự]*

---

#### 6.1.4. Ackley Function

*[Tương tự]*

---

### 6.2. Bài toán Discrete: Traveling Salesman Problem (TSP)

#### 6.2.1. Mô tả bài toán

TSP là bài toán tìm chu trình ngắn nhất đi qua tất cả các thành phố đúng một lần và quay về điểm xuất phát.

#### 6.2.2. Formulation

- Input: Tập n thành phố và ma trận khoảng cách D
- Output: Hoán vị của n thành phố tối thiểu hóa tổng khoảng cách

#### 6.2.3. Độ phức tạp

- Thuộc lớp NP-hard
- Số lượng tour có thể: (n-1)!/2

---

## 7. KẾT QUẢ THỰC NGHIỆM

### 7.1. Thiết lập thực nghiệm

#### 7.1.1. Môi trường

- Python 3.8+
- NumPy 1.24.0
- Matplotlib 3.7.0
- CPU: [Thông tin CPU]
- RAM: [Thông tin RAM]

#### 7.1.2. Tham số thuật toán

| Thuật toán | Tham số | Giá trị |
|------------|---------|---------|
| PSO | n_particles | 30 |
|     | max_iter | 100 |
|     | w | 0.7 |
|     | c1, c2 | 1.5 |
| ACO | n_ants | 30 |
|     | ... | ... |
| ... | ... | ... |

#### 7.1.3. Phương pháp đánh giá

- Số lần chạy: 30 trials cho mỗi thuật toán
- Metrics:
  - Best score
  - Mean score ± Std
  - Convergence speed (iterations to reach 1% of optimum)
  - Computation time
  - Success rate

---

### 7.2. Kết quả trên Sphere Function

#### 7.2.1. Bảng kết quả

| Thuật toán | Mean Score | Std | Best | Worst | Time (s) |
|------------|-----------|-----|------|-------|----------|
| PSO | 0.0012 | 0.0008 | 0.0001 | 0.0045 | 0.523 |
| ACO | 0.0234 | 0.0156 | 0.0089 | 0.0567 | 1.234 |
| ABC | 0.0089 | 0.0045 | 0.0023 | 0.0234 | 0.876 |
| FA | 0.0156 | 0.0098 | 0.0045 | 0.0345 | 1.567 |
| CS | 0.0078 | 0.0034 | 0.0012 | 0.0156 | 0.987 |
| Hill Climbing | 0.156 | 0.089 | 0.045 | 0.345 | 0.234 |
| SA | 0.0345 | 0.0234 | 0.0089 | 0.0789 | 0.456 |
| GA | 0.0234 | 0.0156 | 0.0067 | 0.0456 | 0.678 |

*Ghi chú: Kết quả tốt nhất được in đậm*

#### 7.2.2. Convergence Plot

*[Thêm biểu đồ hội tụ của tất cả thuật toán]*

#### 7.2.3. Box Plot Comparison

*[Thêm box plot so sánh phân bố kết quả]*

#### 7.2.4. Phân tích

- **PSO** cho kết quả tốt nhất trên Sphere function do tính chất unimodal
- **ACO** kém hơn do thiết kế cho bài toán discrete
- **Hill Climbing** dễ bị kẹt ở local optimum
- ...

---

### 7.3. Kết quả trên Rastrigin Function

*[Tương tự như Sphere]*

---

### 7.4. Kết quả trên Rosenbrock Function

*[Tương tự như Sphere]*

---

### 7.5. Kết quả trên Ackley Function

*[Tương tự như Sphere]*

---

### 7.6. Kết quả trên TSP

#### 7.6.1. Thiết lập

- Số thành phố: 20, 50, 100
- Phân bố: Random trong [0, 100] x [0, 100]

#### 7.6.2. Kết quả với TSP-20

| Thuật toán | Mean Distance | Best | Time (s) |
|------------|---------------|------|----------|
| GA | 234.56 | 221.34 | 2.345 |
| SA | 245.67 | 230.12 | 1.234 |
| 2-opt | 256.78 | 245.67 | 0.456 |
| Nearest Neighbor | 345.67 | 345.67 | 0.012 |

#### 7.6.3. Visualization

*[Thêm hình vẽ tour tốt nhất]*

---

### 7.7. So sánh tổng quát

#### 7.7.1. Bảng xếp hạng tổng thể

**Continuous Optimization:**

| Rank | Thuật toán | Điểm trung bình | Ưu điểm |
|------|-----------|----------------|----------|
| 1 | PSO | 8.5/10 | Nhanh, ổn định, dễ implement |
| 2 | CS | 8.0/10 | Khám phá tốt, tránh local optima |
| 3 | ABC | 7.5/10 | Cân bằng exploitation/exploration |
| ... | ... | ... | ... |

**Discrete Optimization (TSP):**

| Rank | Thuật toán | Điểm | Ghi chú |
|------|-----------|------|---------|
| 1 | GA | 9.0/10 | Tốt nhất cho TSP |
| 2 | SA | 7.5/10 | Cân bằng tốc độ/chất lượng |
| ... | ... | ... | ... |

---

## 8. PHÂN TÍCH SENSITIVITY

### 8.1. PSO Parameter Sensitivity

#### 8.1.1. Ảnh hưởng của số lượng particles

*[Biểu đồ và phân tích]*

#### 8.1.2. Ảnh hưởng của inertia weight w

*[Biểu đồ và phân tích]*

---

## 9. KẾT LUẬN

### 9.1. Tổng kết

- Đã implement thành công 5 thuật toán swarm intelligence và 3 thuật toán truyền thống
- Thực nghiệm trên 4 hàm continuous và bài toán TSP
- PSO cho kết quả tốt nhất trên continuous optimization
- GA hiệu quả nhất với TSP

### 9.2. Bài học rút ra

- Swarm intelligence phù hợp với bài toán continuous, multimodal
- Không có thuật toán "tốt nhất" cho mọi bài toán (No Free Lunch Theorem)
- Parameter tuning rất quan trọng
- Trade-off giữa exploration và exploitation

### 9.3. Hướng phát triển

- Hybrid algorithms kết hợp ưu điểm của nhiều thuật toán
- Adaptive parameter tuning
- Parallel implementation để tăng tốc độ
- Áp dụng vào bài toán thực tế

---

## 10. TÀI LIỆU THAM KHẢO

[1] Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN'95 - International Conference on Neural Networks*, 4, 1942-1948.

[2] Dorigo, M., & Stützle, T. (2004). *Ant Colony Optimization*. MIT Press.

[3] Karaboga, D. (2005). An idea based on honey bee swarm for numerical optimization. *Technical Report-tr06*, Erciyes University.

[4] Yang, X. S. (2008). Firefly algorithm, stochastic test functions and design optimisation. *International Journal of Bio-Inspired Computation*, 2(2), 78-84.

[5] Yang, X. S., & Deb, S. (2009). Cuckoo search via Lévy flights. *2009 World Congress on Nature & Biologically Inspired Computing (NaBIC)*, 210-214.

[6] Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671-680.

[7] Holland, J. H. (1992). Genetic algorithms. *Scientific American*, 267(1), 66-73.

---

## PHỤ LỤC

### A. Source Code

*[Link to GitHub repository hoặc attach code]*

### B. Detailed Results

*[Bảng kết quả chi tiết của từng trial]*

### C. Additional Visualizations

*[Thêm các hình ảnh bổ sung]*

---

**Ngày nộp:** [DD/MM/YYYY]  
**Chữ ký nhóm trưởng:** _______________


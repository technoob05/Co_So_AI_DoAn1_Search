# 🚀 BẮT ĐẦU TẠI ĐÂY

## ❗ QUAN TRỌNG: Fix lỗi trước khi chạy

Nếu bạn thấy lỗi `ValueError: numpy.dtype size changed`, làm theo hướng dẫn dưới đây:

---

## ✅ CÁCH FIX NHANH (1 phút)

### Option 1: Auto-fix (RECOMMENDED)

```bash
python install_fix.py
```

Sau đó test:

```bash
python run_simple_test.py
```

---

### Option 2: Manual fix

```bash
# Bước 1: Gỡ packages xung đột
pip uninstall numpy scipy seaborn -y

# Bước 2: Cài NumPy tương thích
pip install numpy==1.26.4

# Bước 3: Cài packages khác
pip install matplotlib pandas tqdm jupyter

# Bước 4: Test
python run_simple_test.py
```

---

## 🎯 SAU KHI FIX XONG

### 1. Test đơn giản (không visualization):

```bash
python run_simple_test.py
```

Output mong đợi:
```
✓ test_functions imported
✓ swarm_intelligence imported
✓ All tests passed!
```

### 2. Chạy demo đầy đủ:

```bash
python demo.py
```

### 3. Test từng thuật toán:

```python
import numpy as np
from src.test_functions import get_test_function
from src.swarm_intelligence.pso import PSO

np.random.seed(42)
func = get_test_function('sphere', dim=10)
pso = PSO(n_particles=30, dim=10, max_iter=100, bounds=func.bounds)
best_pos, best_score = pso.optimize(func, verbose=True)
print(f"Score: {best_score:.6f}")
```

---

## 📚 HƯỚNG DẪN CHI TIẾT

- **Quick start**: `QUICKSTART.md`
- **Fix lỗi**: `FIX_ERRORS.md`
- **Hướng dẫn đầy đủ**: `USAGE_GUIDE.md`
- **Tổng quan**: `PROJECT_SUMMARY.md`

---

## 🔧 TÓM TẮT LỖI VÀ CÁCH FIX

### Lỗi gì?
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

### Tại sao?
NumPy 2.x không tương thích với scipy/seaborn cũ

### Fix thế nào?
Dùng NumPy 1.26.4 thay vì 2.x

### Làm sao?
```bash
pip uninstall numpy -y
pip install numpy==1.26.4
```

---

## ✅ CHECKLIST

- [ ] Đã chạy `python install_fix.py` HOẶC fix manual
- [ ] Test với `python run_simple_test.py` - thấy ✓ All tests passed
- [ ] Chạy được `python demo.py`
- [ ] Đọc `QUICKSTART.md`
- [ ] Bắt đầu làm đồ án!

---

## 🎓 CẤU TRÚC PROJECT

```
Co_So_AI_search/
├── START_HERE.md              ← BẠN ĐANG Ở ĐÂY
├── install_fix.py             ← Chạy để auto-fix
├── run_simple_test.py         ← Test nhanh
├── demo.py                    ← Demo đầy đủ
│
├── FIX_ERRORS.md             ← Chi tiết về errors
├── QUICKSTART.md              ← Quick start guide
├── USAGE_GUIDE.md             ← Hướng dẫn đầy đủ
├── PROJECT_SUMMARY.md         ← Tổng quan project
│
├── src/                       ← Source code
│   ├── swarm_intelligence/    ← 5 thuật toán swarm
│   ├── traditional_search/    ← 3 thuật toán truyền thống
│   ├── discrete_problems/     ← TSP
│   ├── test_functions.py      ← Test functions
│   ├── visualization.py       ← Visualization
│   └── comparison.py          ← Comparison tools
│
└── report/
    └── report_template.md     ← Template báo cáo
```

---

## 💡 TIPS

1. **Luôn chạy từ thư mục gốc** `Co_So_AI_search/`
2. **Test trước khi làm báo cáo** với `run_simple_test.py`
3. **Đọc FIX_ERRORS.md** nếu gặp lỗi khác
4. **Code đã hoàn chỉnh** - bạn chỉ cần chạy và viết báo cáo!

---

## 🚀 NEXT STEPS

```bash
# 1. Fix installation
python install_fix.py

# 2. Test
python run_simple_test.py

# 3. Demo
python demo.py

# 4. Read guide
# Mở QUICKSTART.md

# 5. Start working!
```

---

**Good luck! 🎉**

*Nếu vẫn gặp vấn đề, xem FIX_ERRORS.md*


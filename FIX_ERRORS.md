# 🔧 HƯỚNG DẪN SỬA LỖI

## ❌ Lỗi: NumPy Version Conflict

### Lỗi bạn gặp:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. 
Expected 96 from C header, got 88 from PyObject
```

### Nguyên nhân:

NumPy 2.x không tương thích với scipy/seaborn cũ.

---

## ✅ GIẢI PHÁP 1: Cài đặt lại với NumPy 1.x (RECOMMENDED)

### Bước 1: Gỡ cài đặt packages cũ

```bash
pip uninstall numpy scipy seaborn -y
```

### Bước 2: Cài đặt lại với requirements.txt đã fix

```bash
pip install -r requirements.txt
```

### Bước 3: Test

```bash
python -c "import numpy; print('NumPy:', numpy.__version__)"
python demo.py
```

---

## ✅ GIẢI PHÁP 2: Nâng cấp environment (Nếu Solution 1 không work)

### Option A: Cài đặt trong môi trường mới

```bash
# Tạo virtual environment mới
python -m venv swarm_env

# Activate (Windows)
swarm_env\Scripts\activate

# Activate (Linux/Mac)
source swarm_env/bin/activate

# Cài đặt
pip install numpy==1.26.4 matplotlib pandas tqdm jupyter notebook
```

### Option B: Fix NumPy version cụ thể

```bash
pip install numpy==1.26.4 --force-reinstall
```

---

## ✅ GIẢI PHÁP 3: Chạy không cần Seaborn

Code đã được update để **không bắt buộc** phải có seaborn. 

Nếu không có seaborn, code vẫn chạy bình thường với matplotlib!

```bash
# Chỉ cần numpy và matplotlib
pip install numpy==1.26.4 matplotlib pandas tqdm
python demo.py
```

---

## 🧪 Test sau khi fix

```bash
# Test 1: Import test
python -c "from src.test_functions import get_test_function; print('✓ OK')"

# Test 2: Quick test
python -c "
import numpy as np
from src.test_functions import get_test_function
from src.swarm_intelligence.pso import PSO

func = get_test_function('sphere', dim=5)
pso = PSO(n_particles=10, dim=5, max_iter=10, bounds=func.bounds)
_, score = pso.optimize(func)
print(f'✓ PSO works! Score: {score:.6f}')
"

# Test 3: Full demo
python demo.py
```

---

## 📦 Recommended Versions

Nếu bạn muốn cài đặt từ đầu:

```bash
# Uninstall all
pip uninstall numpy scipy seaborn matplotlib pandas tqdm jupyter -y

# Install fresh (compatible versions)
pip install numpy==1.26.4
pip install matplotlib==3.8.0
pip install pandas==2.1.0
pip install tqdm==4.66.0
pip install jupyter==1.0.0
pip install notebook==7.0.0
```

---

## 🔍 Kiểm tra versions hiện tại

```python
import numpy
import matplotlib
import pandas

print(f"NumPy: {numpy.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Pandas: {pandas.__version__}")
```

**Expected output:**
```
NumPy: 1.26.x
Matplotlib: 3.8.x
Pandas: 2.x.x
```

---

## 🚨 Nếu vẫn gặp lỗi

### Lỗi về import

```python
# Thay vì chạy từ bất kỳ đâu, chạy từ thư mục gốc
cd C:\Users\Admin\Downloads\Co_So_AI_search
python demo.py
```

### Lỗi về path

```python
# Thêm vào đầu script
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

### Lỗi visualization

Nếu không muốn dùng visualization, comment out:

```python
# from src.visualization import OptimizationVisualizer
```

Và skip phần plot trong code.

---

## 💡 Quick Fix Script

Tạo file `fix_install.py`:

```python
import subprocess
import sys

print("Fixing installation...")

# Uninstall conflicting packages
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", 
                "numpy", "scipy", "seaborn"])

# Install compatible versions
subprocess.run([sys.executable, "-m", "pip", "install", 
                "numpy==1.26.4", "matplotlib", "pandas", "tqdm", "jupyter"])

print("✓ Done! Try running: python demo.py")
```

Chạy:
```bash
python fix_install.py
```

---

## ✅ Verified Working Setup

```
Python: 3.8+
NumPy: 1.26.4
Matplotlib: 3.8.0
Pandas: 2.1.0
Tqdm: 4.66.0
```

---

## 📞 Still Having Issues?

1. **Option 1**: Dùng Google Colab
   - Upload toàn bộ folder lên Google Drive
   - Mở notebook trong Colab
   - Chạy: `!pip install numpy==1.26.4 matplotlib pandas tqdm`

2. **Option 2**: Chạy minimal version
   ```bash
   pip install numpy==1.26.4 matplotlib
   # Chỉ chạy thuật toán, không visualization
   ```

3. **Option 3**: Fresh Python environment
   - Cài Python mới (3.9 hoặc 3.10)
   - Tạo venv mới
   - Cài packages

---

**Sau khi fix xong, bắt đầu với:**

```bash
python demo.py
```

Hoặc:

```python
import numpy as np
from src.test_functions import get_test_function
from src.swarm_intelligence.pso import PSO

func = get_test_function('sphere', dim=10)
pso = PSO(n_particles=30, dim=10, max_iter=100, bounds=func.bounds)
best_pos, best_score = pso.optimize(func, verbose=True)
print(f"Success! Best score: {best_score:.6f}")
```

---

**Good luck! 🚀**


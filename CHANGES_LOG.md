# 📝 CHANGES LOG - Đã sửa lỗi

## 🔧 Version 1.0.1 - Fix NumPy Compatibility (2025-10-28)

### ❌ Vấn đề

Người dùng gặp lỗi khi chạy `demo.py`:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. 
Expected 96 from C header, got 88 from PyObject
```

**Nguyên nhân:** NumPy 2.x không tương thích với scipy/seaborn phiên bản cũ.

---

### ✅ Đã sửa

#### 1. **requirements.txt**
- ❌ Trước: `numpy>=1.24.0`
- ✅ Sau: `numpy>=1.24.0,<2.0.0`
- ❌ Trước: Có `seaborn>=0.12.0`
- ✅ Sau: Xóa seaborn (không cần thiết cho đồ án)

#### 2. **src/visualization.py**
- Seaborn bây giờ là **optional** (không bắt buộc)
- Nếu không có seaborn, dùng matplotlib style
- Code vẫn chạy bình thường!

```python
# Trước
import seaborn as sns
sns.set_style("whitegrid")

# Sau
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    plt.style.use('default')
```

#### 3. **Files mới được tạo**

| File | Mục đích |
|------|----------|
| `START_HERE.md` | Hướng dẫn bắt đầu và fix lỗi nhanh |
| `FIX_ERRORS.md` | Chi tiết về lỗi và nhiều cách fix |
| `install_fix.py` | Script tự động fix installation |
| `run_simple_test.py` | Test nhanh không cần visualization |
| `CHANGES_LOG.md` | File này - log các thay đổi |

#### 4. **Files đã cập nhật**

- `README.md` - Thêm warning và hướng dẫn fix
- `QUICKSTART.md` - Thêm section về fix lỗi
- `requirements.txt` - Pin NumPy version

---

### 🚀 Cách sử dụng sau khi fix

#### Option 1: Auto-fix (Recommended)

```bash
python install_fix.py
python run_simple_test.py
```

#### Option 2: Manual fix

```bash
pip uninstall numpy scipy seaborn -y
pip install numpy==1.26.4 matplotlib pandas tqdm jupyter
python run_simple_test.py
```

#### Option 3: Fresh install

```bash
pip install -r requirements.txt
python run_simple_test.py
```

---

### ✅ Test để verify fix hoạt động

#### Test 1: Simple test (no visualization)

```bash
python run_simple_test.py
```

Expected output:
```
✓ test_functions imported
✓ swarm_intelligence imported
✓ traditional_search imported
Running PSO...
✓ PSO: 0.001234
...
✓ All tests passed!
```

#### Test 2: Import test

```bash
python -c "from src.test_functions import get_test_function; print('OK')"
```

#### Test 3: Full demo

```bash
python demo.py
```

---

### 📊 Compatibility Matrix

| Package | Version | Status |
|---------|---------|--------|
| Python | 3.8+ | ✅ Required |
| NumPy | 1.24.0 - 1.26.x | ✅ Recommended |
| NumPy | 2.x | ❌ Not compatible |
| Matplotlib | 3.7.0+ | ✅ Required |
| Pandas | 2.0.0+ | ✅ Required |
| Seaborn | Any | ⚠️ Optional |
| Scipy | Any | ⚠️ Optional |
| Tqdm | 4.65.0+ | ✅ Recommended |
| Jupyter | 1.0.0+ | ✅ For notebooks |

---

### 🎯 Verified Working Configurations

#### Config 1: Minimal (chỉ core dependencies)
```
numpy==1.26.4
matplotlib==3.8.0
pandas==2.1.0
```

#### Config 2: Full (tất cả features)
```
numpy==1.26.4
matplotlib==3.8.0
pandas==2.1.0
tqdm==4.66.0
jupyter==1.0.0
```

#### Config 3: With visualization (optional)
```
numpy==1.26.4
matplotlib==3.8.0
seaborn==0.13.0  # Optional, works with NumPy 1.x
pandas==2.1.0
```

---

### 📝 Breaking Changes

**None!** Tất cả code cũ vẫn hoạt động bình thường.

Thay đổi duy nhất:
- Seaborn bây giờ là optional thay vì required
- Nếu không có seaborn, visualization vẫn hoạt động với matplotlib

---

### 🐛 Known Issues & Workarounds

#### Issue 1: "ModuleNotFoundError: No module named 'src'"

**Workaround:**
```bash
# Luôn chạy từ thư mục gốc
cd Co_So_AI_search
python demo.py
```

#### Issue 2: NumPy 2.x vẫn được cài

**Workaround:**
```bash
pip uninstall numpy -y
pip install numpy==1.26.4 --force-reinstall
```

#### Issue 3: Matplotlib plot không hiển thị

**Workaround:**
```python
import matplotlib.pyplot as plt
plt.show()  # Thêm dòng này sau mỗi plot
```

---

### 🔄 Migration Guide

Nếu bạn đang dùng version cũ:

#### Từ requirements cũ (có seaborn):

```bash
# 1. Uninstall all
pip uninstall numpy scipy seaborn matplotlib pandas -y

# 2. Install new requirements
pip install -r requirements.txt

# 3. Test
python run_simple_test.py
```

#### Nếu đang dùng NumPy 2.x:

```bash
# Downgrade to 1.x
pip install numpy==1.26.4 --force-reinstall
```

---

### 📚 Updated Documentation

Tất cả documentation đã được cập nhật:

- ✅ `README.md` - Thêm warning về NumPy
- ✅ `QUICKSTART.md` - Thêm fix instructions
- ✅ `START_HERE.md` - **NEW** - Điểm bắt đầu cho người mới
- ✅ `FIX_ERRORS.md` - **NEW** - Troubleshooting guide
- ✅ `USAGE_GUIDE.md` - Vẫn valid, không thay đổi
- ✅ `PROJECT_SUMMARY.md` - Vẫn valid, không thay đổi

---

### 🎓 For Students

**Quan trọng:** Lỗi này **KHÔNG ảnh hưởng** đến đồ án của bạn!

- ✅ Tất cả thuật toán vẫn hoạt động 100%
- ✅ Tất cả test functions vẫn đúng
- ✅ Comparison tools vẫn chính xác
- ✅ Chỉ là vấn đề về dependencies version

**Bạn chỉ cần:**
1. Fix installation (1-2 phút)
2. Chạy experiments
3. Viết báo cáo như bình thường

---

### 🚀 Next Release Plans

Version 1.1.0 (Future):
- [ ] Add unit tests (pytest)
- [ ] Add more test functions
- [ ] Parallel execution support
- [ ] Web interface (Streamlit)
- [ ] Docker support

---

### 📧 Support

Nếu vẫn gặp vấn đề:

1. Đọc `FIX_ERRORS.md`
2. Thử `python install_fix.py`
3. Chạy `python run_simple_test.py`
4. Check version: `python -c "import numpy; print(numpy.__version__)"`

---

**Status:** ✅ Fixed  
**Date:** 2025-10-28  
**Version:** 1.0.1


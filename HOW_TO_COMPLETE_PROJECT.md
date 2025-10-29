# 📋 HƯỚNG DẪN HOÀN THÀNH ĐỒ ÁN

## ✅ ĐÃ LÀM XONG

### 1. Code (100%) ✅
- ✅ 5 thuật toán Swarm Intelligence
- ✅ 3 thuật toán truyền thống
- ✅ 4 hàm test continuous
- ✅ TSP (discrete)
- ✅ Visualization tools
- ✅ Comparison framework

### 2. Notebook Experiments (100%) ✅
- ✅ **`notebooks/03_complete_experiments.ipynb`** - QUAN TRỌNG NHẤT!
  - Chạy tất cả experiments
  - Thu thập số liệu
  - Tạo visualizations
  - Export CSV

---

## 🎯 CÒN PHẢI LÀM (3 việc chính)

### 1. CHẠY NOTEBOOK VÀ LẤY KẾT QUẢ ⏳

**Bước 1:** Fix installation (nếu chưa)
```bash
python install_fix.py
```

**Bước 2:** Chạy notebook
```bash
cd notebooks
jupyter notebook 03_complete_experiments.ipynb
```

Hoặc nếu dùng VS Code:
- Mở file `notebooks/03_complete_experiments.ipynb`
- Click "Run All"

**Kết quả sau khi chạy:**
- Folder `results/plots/`: 16 hình ảnh (convergence, boxplot, surface, TSP)
- Folder `results/`: 6 CSV files với số liệu

**Thời gian:** ~30-45 phút (tùy máy)

---

### 2. VIẾT BÁO CÁO ⏳

**Template có sẵn:** `report/report_template.md`

**Các bước:**

#### Bước 1: Điền thông tin nhóm
```markdown
| MSSV | Họ và Tên | Email | Ghi chú |
|------|-----------|-------|---------|
| 12345678 | Nguyễn Văn A | ... | Nhóm trưởng |
| ... | ... | ... | ... |
```

#### Bước 2: Điền phân công công việc
```markdown
| Thành viên | Công việc | Hoàn thành |
|------------|-----------|-----------|
| Nguyễn Văn A | Implement PSO, ACO | 100% |
| ... | ... | ... |
```

#### Bước 3: Copy kết quả từ CSV
- Mở `results/continuous_optimization_results.csv`
- Copy vào bảng trong báo cáo
- Format cho đẹp

#### Bước 4: Thêm hình ảnh
- Copy các file từ `results/plots/`
- Paste vào báo cáo
- Thêm caption cho mỗi hình

#### Bước 5: Phân tích kết quả
- Giải thích tại sao thuật toán này tốt hơn
- So sánh swarm vs traditional
- Nhận xét về từng test function

#### Bước 6: Export to PDF
```bash
# Dùng Pandoc
pandoc report_template.md -o report.pdf --pdf-engine=xelatex

# Hoặc dùng online converter
# https://www.markdowntopdf.com/
```

**Checklist báo cáo:**
- [ ] Thông tin nhóm đã điền
- [ ] Phân công công việc đã điền
- [ ] Có đủ 25 trang
- [ ] Tất cả hình ảnh đã có caption
- [ ] Không bị cắt hình ở page break
- [ ] Tài liệu tham khảo APA format
- [ ] Tiếng Việt chuẩn
- [ ] Exported to PDF đẹp

---

### 3. TẠO DEMO VIDEO ⏳

**Yêu cầu:** >5 phút, upload YouTube

**Nội dung đề xuất:**

**Phút 0-1: Giới thiệu**
- Giới thiệu đồ án
- Giới thiệu nhóm
- Mục tiêu

**Phút 1-2: Giải thích thuật toán**
- Chọn 1-2 thuật toán để giải thích chi tiết
- Vẽ diagram/flowchart
- Giải thích intuition

**Phút 2-4: Demo code**
- Show structure project
- Chạy `run_simple_test.py`
- Chạy 1 cell trong notebook
- Show kết quả

**Phút 4-5: Kết quả**
- Show plots từ `results/plots/`
- Show bảng comparison
- Nhận xét

**Phút 5+: Kết luận**
- Tóm tắt findings
- Lessons learned
- Q&A (nếu có)

**Tools ghi video:**
- OBS Studio (free)
- Zoom (record meeting)
- Screen recorder built-in (Windows: Win+G)

**Script mẫu:**
```
"Xin chào, nhóm chúng em xin giới thiệu đồ án về Thuật toán 
Swarm Intelligence.

[Show slide title]

Đồ án này implement 5 thuật toán swarm intelligence: PSO, ACO, 
ABC, FA, và CS, so sánh với 3 thuật toán truyền thống.

[Show code structure]

Chúng em đã test trên 4 hàm continuous và 1 bài toán TSP.

[Demo chạy code]

...
"
```

---

### 4. PUSH LÊN GITHUB ⏳

```bash
# Tạo repo trên GitHub
# Rồi:

git init
git add .
git commit -m "Initial commit - Swarm Intelligence Project"
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

**Thêm vào README.md:**
- Link demo video
- Hướng dẫn run code
- Requirements

---

### 5. ĐÓNG GÓI NỘP BÀI ⏳

```bash
# Tạo thư mục nộp
mkdir Group_XX

# Copy files cần nộp
cp report.pdf Group_XX/
cp -r src Group_XX/
cp -r results Group_XX/
cp -r notebooks Group_XX/
cp README.md Group_XX/
cp requirements.txt Group_XX/

# Nén
zip -r Group_XX.zip Group_XX/
```

**Nội dung file nộp:**
```
Group_XX.zip
├── report.pdf                    # Báo cáo
├── README.md                     # Hướng dẫn
├── requirements.txt
├── src/                          # Source code
├── notebooks/                    # Jupyter notebooks
└── results/                      # Kết quả (nếu <25MB)
```

**Trong report.pdf phải có:**
- Link GitHub repo
- Link demo video (YouTube/Drive)
- Link results (nếu file >25MB)

---

## 📅 TIMELINE ĐỀ XUẤT

### Ngày 1-2: Chạy Experiments
- [ ] Fix installation
- [ ] Chạy `notebooks/03_complete_experiments.ipynb`
- [ ] Kiểm tra kết quả

### Ngày 3-5: Viết Báo cáo
- [ ] Điền thông tin nhóm
- [ ] Copy kết quả từ CSV
- [ ] Thêm hình ảnh
- [ ] Phân tích
- [ ] Export PDF

### Ngày 6: Demo Video
- [ ] Chuẩn bị script
- [ ] Ghi video
- [ ] Edit
- [ ] Upload YouTube

### Ngày 7: Hoàn tất
- [ ] Push GitHub
- [ ] Đóng gói file nộp
- [ ] Review lần cuối
- [ ] Nộp bài

---

## ⚠️ CHECKLIST TRƯỚC KHI NỘP

### Code
- [ ] Tất cả code chạy được
- [ ] README.md đầy đủ
- [ ] Push lên GitHub
- [ ] Link GitHub trong báo cáo

### Báo cáo
- [ ] Thông tin nhóm đầy đủ
- [ ] Phân công công việc chi tiết
- [ ] Tối thiểu 25 trang
- [ ] Tất cả hình ảnh rõ ràng
- [ ] Không bị cắt hình
- [ ] References APA format
- [ ] PDF format đẹp

### Demo Video
- [ ] >5 phút
- [ ] Upload YouTube
- [ ] Link trong báo cáo
- [ ] Public/Unlisted

### File nộp
- [ ] Format: Group_XX.zip
- [ ] Size <25MB (hoặc có Drive link)
- [ ] Có đủ: report + code + README

---

## 💡 TIPS

### Viết báo cáo:
1. **Không copy-paste code vào báo cáo** - Chỉ pseudo-code
2. **Thêm nhiều hình ảnh** - Visualization rất quan trọng
3. **Phân tích sâu** - Đừng chỉ list số
4. **So sánh có chứng cứ** - Dùng bảng, plots
5. **Cite đúng format** - APA style

### Demo video:
1. **Chuẩn bị script** - Đừng improvise
2. **Test mic/camera** - Audio rõ ràng
3. **Screen clean** - Đóng tabs không cần thiết
4. **Practice trước** - Tối thiểu 1 lần
5. **Keep it simple** - Đừng quá phức tạp

### GitHub:
1. **README.md đẹp** - First impression matters
2. **Organize well** - Clear structure
3. **.gitignore** - Đừng push __pycache__
4. **Commit messages** - Clear và meaningful

---

## 🆘 NẾU GẶP VẤN ĐỀ

### Lỗi khi chạy notebook:
```bash
# Try
python install_fix.py
python run_simple_test.py

# Nếu vẫn lỗi, chạy từng phần trong notebook
```

### Notebook chạy quá lâu:
```python
# Giảm n_trials
CONFIG['n_trials'] = 10  # thay vì 30

# Hoặc giảm iterations
CONFIG['max_iter_swarm'] = 50  # thay vì 100
```

### PDF quá lớn:
- Compress hình ảnh trước khi thêm vào
- Dùng tool như TinyPNG
- Hoặc giảm resolution plots

### Video quá lớn để upload:
- Compress video (Handbrake)
- Upload Google Drive thay vì YouTube
- Hoặc split thành 2 parts

---

## 📞 FILES QUAN TRỌNG NHẤT

### Must-read:
1. `START_HERE.md` - Bắt đầu
2. `CHECKLIST_REQUIREMENTS.md` - Check đã làm gì
3. `HOW_TO_COMPLETE_PROJECT.md` - File này!

### Must-run:
1. `install_fix.py` - Fix installation
2. `notebooks/03_complete_experiments.ipynb` - Chạy experiments

### Must-use:
1. `report/report_template.md` - Template báo cáo
2. `results/` - Kết quả experiments

---

## ✅ READY TO GO!

Bạn đã có:
- ✅ Code hoàn chỉnh
- ✅ Notebook experiments
- ✅ Template báo cáo
- ✅ Tất cả tools cần thiết

**Next step:** 
1. Fix installation (`python install_fix.py`)
2. Chạy notebook (`notebooks/03_complete_experiments.ipynb`)
3. Viết báo cáo (dùng template)
4. Tạo demo video
5. Nộp bài!

---

**Good luck! 🚀**

*Estimated total time: 2-3 ngày (nếu làm focused)*


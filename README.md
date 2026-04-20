# 🛡️ AI-Assisted Software Defect Prediction Tool
### Công cụ dự đoán lỗi phần mềm sử dụng Machine Learning

---

## 📋 Giới Thiệu

Dự án **P7 - AI Tool for Software Defect Prediction** là công cụ dự đoán lỗi phần mềm sử dụng Machine Learning. Công cụ này giúp phát hiện các module có khả năng chứa lỗi dựa trên các metrics của code.

**Môn học:** Software Measurement & Analysis

---

## 📁 Cấu Trúc Dự Án

```
defect-prediction-tool/
├── app.py                      # Ứng dụng Streamlit chính
├── requirements.txt            # Các thư viện Python
├── run.bat                    # File chạy nhanh (Windows)
├── README.md                  # File hướng dẫn này
│
├── src/                       # Thư mục source code
│   ├── __init__.py
│   ├── preprocessing.py       # Tiền xử lý dữ liệu
│   ├── features.py           # Trích xuất đặc trưng
│   ├── models.py             # ML models (LR, RF, NN)
│   ├── evaluation.py         # Đánh giá model
│   ├── code_metrics_extractor.py  # Trích xuất metrics từ source code
│   └── database.py           # SQLite database cho lịch sử
│
├── data/                      # Dữ liệu
│   ├── kc1.arff             # Dataset NASA KC1 (2,109 samples)
│   ├── kc2.arff             # Dataset NASA KC2 (522 samples)
│   └── kc1_kc2.csv         # Dataset KC1 + KC2 (2,631 samples)
│
├── models/                    # Lưu trữ model đã train
│   └── __init__.py          # ModelManager class
│
├── dashboard/                 # Trang Dashboard
│   └── Dashboard.py          # Biểu đồ trực quan
│
├── reports/                  # Xuất báo cáo
│   └── ReportGenerator.py    # Tạo báo cáo (CSV, TXT)
│
├── database/                 # SQLite database
│   └── history.db           # Lưu lịch sử phân tích
│
└── venv/                     # Virtual environment
```

---

## 🚀 Cài Đặt

### Yêu cầu
- Python 3.8+
- Windows/Mac/Linux

### Cách 1: Cài đặt thủ công

```bash
# Clone hoặc tải project
cd defect-prediction-tool

# Tạo virtual environment (khuyến nghị)
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Cài đặt các thư viện
pip install -r requirements.txt
```

### Cách 2: Sử dụng file có sẵn

```bash
# Chạy file run.bat (Windows)
run.bat
```

---

## ▶️ Chạy Ứng Dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại: **http://localhost:8501**

---

## 📊 Các Tính Năng

### 1. 🏠 Trang Chủ (Home)
- Màn hình chào, giới thiệu tổng quan
- Hướng dẫn sử dụng

### 2. 📊 Dashboard
- Tổng quan dự án
- Biểu đồ phân bố dữ liệu (pie chart)
- Biểu đồ radar so sánh các model
- Hoạt động gần nhất

### 3. 📂 Tải Dữ Liệu
Có 2 cách tải dữ liệu:

| Cách | Mô tả |
|------|-------|
| **Từ CSV** | Upload file CSV với cột `LABEL` (0/1) |
| **Từ Source Code** | Upload trực tiếp file .py, .java, .js, .c, .cpp... |

**Dataset có sẵn:** `data/kc1_kc2.csv` (2,631 samples, 16.5% defect)

### 4. 🤖 Huấn Luyện Model
Huấn luyện 3 model Machine Learning:

| Model | Mô tả |
|-------|--------|
| **Logistic Regression** | Model tuyến tính, dễ diễn giải |
| **Random Forest** | Ensemble method, xử lý tốt phi tuyến |
| **Neural Network (MLP)** | Mạng nơ-ron nhiều lớp |

**Cấu hình:**
- Tỷ lệ test: 10-40% (mặc định 20%)
- Random State: 42
- Stratified Split: Giữ nguyên tỷ lệ nhãn

### 5. 🔮 Dự Đoán
- **Sử dụng dữ liệu test**: Xem kết quả dự đoán trên tập test
- **Nhập thủ công**: Nhập metrics để dự đoán cho module mới

**Kết quả:**
- Xác suất lỗi của từng model
- Dự đoán tổng hợp (ensemble)
- Mức độ rủi ro: 🔴 Cao / 🟡 Trung bình / 🟢 Thấp

### 6. 📊 Đánh Giá
Hiển thị các metrics đánh giá:

| Metric | Mô tả |
|--------|-------|
| Accuracy | Độ chính xác tổng thể |
| Precision | Độ chính xác dự đoán positive |
| Recall | Tỷ lệ phát hiện lỗi |
| F1-Score | Trung bình điều hòa |
| ROC-AUC | Khả năng phân loại (quan trọng nhất) |

**Biểu đồ:**
- So sánh bar chart
- Confusion Matrix heatmap
- Feature Importance (Random Forest)
- ROC Curve
- Precision-Recall Curve

### 7. 💾 Xuất Báo Cáo
- Xuất kết quả đánh giá (CSV)
- Xuất kết quả dự đoán (CSV)
- Tạo báo cáo text chi tiết

### 8. 📜 Lịch Sử
- Xem danh sách các phiên phân tích
- Xem chi tiết metrics của từng phiên
- Biểu đồ LOC và Complexity

### 9. 📁 Models
- **Lưu model**: Lưu model đã train vào thư mục `models/`
- **Danh sách**: Xem các model đã lưu

### 10. ℹ️ Giới Thiệu
- Thông tin về dự án
- Các metrics được hỗ trợ
- Công nghệ sử dụng

---

## 📥 Định Dạng Dữ Liệu Đầu Vào

### CSV File
File CSV cần có:
- Các cột features (metrics)
- Cột target: `LABEL` hoặc `defects` (0 = không lỗi, 1 = có lỗi)

**Ví dụ:**
```csv
loc,branchCount,complexity,LABEL
100,5,3,0
250,15,8,1
...
```

### Metrics Được Hỗ Trợ (từ Source Code)
| Metric | Mô tả |
|--------|-------|
| LOC | Số dòng code |
| LOC_BLANK | Số dòng trắng |
| LOC_COMMENTS | Số dòng comment |
| FUNCTION_COUNT | Số hàm |
| CLASS_COUNT | Số lớp |
| CYCLOMATIC_COMPLEXITY | Độ phức tạp Cyclomatic |
| DECISION_COUNT | Số điểm quyết định |
| COMMENT_RATIO | Tỷ lệ comment |

### Metrics từ Dataset NASA (KC1/KC2)
| Metric | Mô tả |
|--------|-------|
| loc | Lines of Code |
| v(g) | Cyclomatic Complexity |
| ev(g) | Essential Complexity |
| iv(g) | Design Complexity |
| n | Halstead Total |
| v | Halstead Volume |
| l | Halstead Level |
| d | Halstead Difficulty |
| i | Halstead Intelligence |
| e | Halstead Effort |
| b | Halstead Estimated Bugs |
| t | Halstead Time Estimator |
| lOCode | Lines of Code |
| lOComment | Lines of Comments |
| lOBlank | Lines of Blank |
| uniq_Op | Unique Operators |
| uniq_Opnd | Unique Operands |
| total_Op | Total Operators |
| total_Opnd | Total Operands |
| branchCount | Branch Count |

---

## 📈 Kết Quả Huấn Luyện (Dataset KC1+KC2)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|-----------|---------|
| Logistic Regression | 0.745 | 0.357 | 0.694 | 0.472 | **0.796** |
| Random Forest | 0.804 | 0.419 | 0.500 | 0.456 | 0.783 |
| Neural Network | 0.851 | 0.667 | 0.185 | 0.290 | 0.793 |

**Dataset:** 2,631 samples (16.5% defect rate)

---

## 🛠️ Công Nghệng

| Công nghệ | Mô tả |
|-------- Sử Dụ---|--------|
| **Python 3.x** | Ngôn ngữ lập trình |
| **Streamlit** | Framework web |
| **Scikit-learn** | Machine Learning |
| **TensorFlow/Keras** | Neural Network (nếu có) |
| **Plotly** | Trực quan hóa |
| **Pandas/NumPy** | Xử lý dữ liệu |
| **SQLite** | Lưu trữ lịch sử |

---

## 📚 Nguồn Tham Khảo

### Dataset
- [NASA Metrics Data](https://github.com/klainfo/NASADatasetDirectory)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/467/pcms+software+defect+prediction)
- [OpenML](https://www.openml.org/search?type=data)

### Tài liệu
- Menzies, T., & Di Stefano, J. (2004). "Assessing Predictors of Software Defects"
- NASA MDP (Metrics Data Program) Documentation

---

## 📅 Lộ Trình Dự Án (7 Tuần)

| Tuần | Nội dung |
|------|----------|
| Tuần 1 | Setup, đề xuất project |
| Tuần 2 | Thiết kế kiến trúc hệ thống |
| Tuần 3-4 | Cài đặt, tiền xử lý, train model |
| Tuần 5 | Dashboard, trực quan hóa |
| Tuần 6 | Đánh giá, so sánh model |
| Tuần 7 | Báo cáo, hoàn thiện |

---

## 👥 Tác Giả

**Dự án:** P7 - AI Tool for Software Defect Prediction

**Môn học:** Software Measurement & Analysis

**Nhóm:** 5 sinh viên

---

## 📝 License

Dự án được phát triển cho mục đích học tập.

---

*Developed for Software Measurement & Analysis Course Project*

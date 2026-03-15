"""
Training Guide for AI Defect Prediction Tool
Hướng dẫn huấn luyện model để đạt kết quả tối ưu
"""

# ============================================================
# HƯỚNG DẪN TRAIN MODEL ĐỂ ĐẠT KẾT QUẢ TỐT NHẤT
# ============================================================

# ============================================================
# BƯỚC 1: CHUẨN BỊ DỮ LIỆU
# ============================================================

"""
Dữ liệu tốt cần có:
1. Số lượng mẫu: Tối thiểu 100-200 mẫu (càng nhiều càng tốt)
2. Cân bằng nhãn: Tỷ lệ defect/không defect nên ~20-30%
3. Đầy đủ features: Các metrics như LOC, Complexity, Coupling...

Cách chuẩn bị dữ liệu:
- Cách 1: Sử dụng dữ liệu thực tế từ dự án của bạn
- Cách 2: Sử dụng dataset chuẩn như NASA KC1, KC2 (có sẵn trên UCI)

Nếu sử dụng CSV, cần có:
- Các cột metrics (LOC, CYCLOMATIC_COMPLEXITY, v.v.)
- Cột LABEL hoặc defects (0 = không lỗi, 1 = có lỗi)
"""

# ============================================================
# BƯỚC 2: CÁC THAM SỐ HUẤN LUYỆN TỐI ƯU
# ============================================================

"""
Test Size: 0.2 - 0.3 (20-30% dữ liệu cho test)
Random State: 42 (để kết quả reproducible)
Stratified Split: True (giữ nguyên tỷ lệ nhãn)
"""

# ============================================================
# BƯỚC 3: ĐÁNH GIÁ KẾT QUẢ
# ============================================================

"""
Các metrics quan trọng:
- Accuracy: Độ chính xác tổng thể
- Precision: Tỷ lệ dự đoán đúng trong các dự đoán positive
- Recall: Tỷ lệ phát hiện ra các trường hợp có lỗi
- F1-Score: Trung bình điều hòa của Precision và Recall
- ROC-AUC: Khả năng phân loại tổng thể (>= 0.7 là tốt)

Mục tiêu:
- F1-Score >= 0.7
- ROC-AUC >= 0.7
- Recall >= 0.6 (quan trọng trong dự đoán lỗi)
"""

# ============================================================
# CÁCH SỬ DỤNG TRONG APP
# ============================================================

"""
1. Vào menu "Tải Dữ Liệu"
2. Upload file CSV có cột LABEL
   HOẶC
   Upload nhiều file code (>= 10 files)

3. Vào menu "Huấn Luyện Model"
4. Bấm nút "Tạo Dữ Liệu Mẫu" nếu chưa có dữ liệu
5. Bấm "Huấn Luyện Tất Cả Models"

6. Xem kết quả trong menu "Đánh Giá"
"""

print("=" * 60)
print("HUONG DAN TRAIN MODEL")
print("=" * 60)
print("1. Chuẩn bị dữ liệu CSV với cột LABEL (0/1)")
print("2. Upload dữ liệu trong menu 'Tai Du Lieu'")
print("3. Vao menu 'Huan Luyen Model' de train")
print("4. Xem ket qua trong menu 'Danh Gia'")
print("=" * 60)

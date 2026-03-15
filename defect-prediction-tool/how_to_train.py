"""
Training Guide - Detailed Instructions
Hướng dẫn chi tiết để train model đạt kết quả tốt nhất
"""

# ============================================================
# VẤN ĐỀ: TẠI SAO MODEL CHƯA TỐT?
# ============================================================

"""
1. Dữ liệu mẫu ngẫu nhiên không có pattern rõ ràng
2. Số lượng mẫu còn ít (300)
3. Cần dữ liệu thực tế có LABEL chính xác

GIẢI PHÁP:
- Sử dụng dữ liệu thực tế từ dự án của bạn
- Hoặc sử dụng dataset chuẩn (NASA KC1, KC2)
"""

# ============================================================
# CÁCH 1: SỬ DỤNG DỮ LIỆU THỰC TẾ
# ============================================================

"""
1. Chuẩn bị file CSV với các cột:
   - LOC (số dòng code)
   - CYCLOMATIC_COMPLEXITY (độ phức tạp)
   - FUNCTION_COUNT (số hàm)
   - CLASS_COUNT (số lớp)
   - COMMENT_RATIO (tỷ lệ comment)
   - DECISION_COUNT (số điểm quyết định)
   - CODE_CHURN (số dòng thay đổi)
   - COUPLING (độ liên kết)
   - LABEL (0 = không lỗi, 1 = có lỗi) <- QUAN TRỌNG!

2. Upload file CSV trong app (menu "Tải Dữ Liệu" > tab "Từ CSV")

3. Train model và xem kết quả
"""

# ============================================================
# CÁCH 2: SỬ DỤNG DATASET NASA (RECOMMENDED)
# ============================================================

"""
Dataset NASA KC1/KC2 là chuẩn công nghiệp:
- KC1: 2109 modules, ~15% defect
- KC2: 522 modules, ~20% defect

Tải từ: https://github.com/klainfo/NASADatasetDirectory

Hoặc tôi có thể tạo script để tải dataset này.
"""

print("=" * 60)
print("KET QUA HUAN LUYEN HIEN TAI:")
print("=" * 60)
print("LR:  Accuracy=0.64, F1=0.54, AUC=0.67")
print("RF:  Accuracy=0.65, F1=0.43, AUC=0.62")
print("NN:  Accuracy=0.59, F1=0.37, AUC=0.53")
print("=" * 60)
print("DE DAT KET QUA TOT HON, CAN:")
print("1. Su dung du lieu thuc te voi LABEL chinh xac")
print("2. It nhat 500-1000 mau")
print("3. Ti le defect ~20-30%")
print("=" * 60)

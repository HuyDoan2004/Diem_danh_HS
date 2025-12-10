# CẬP NHẬT HỆ THỐNG

## [OK] Đã hoàn thành

### 1. Dọn dẹp hệ thống
- [OK] Xóa `run.bat` (thay bằng `start.bat`)
- [OK] Xóa `main_realtime.py`, `enroll.py`, `anti_spoof.py`, `track.py`, `events.py`
- [OK] Xóa `configs/model.yaml`, `configs/room.yaml`

### 2. Thêm chức năng Webcam
Bây giờ hệ thống hỗ trợ **cả 2 loại camera**:

#### **Option 1: RealSense D435i (Depth Camera)**
- Thu thập dữ liệu RGB + Depth
- Độ chính xác cao hơn với thông tin 3D
- Chống spoofing tốt hơn

#### **Option 2: Webcam thường**
- Không cần phần cứng đặc biệt
- Dễ dàng test và demo
- Chỉ cần camera laptop/USB

**Cách sử dụng:**
```bash
# Thu thập dữ liệu
python src/collect_data.py
# Chọn: 1 = RealSense, 2 = Webcam

# Chạy hệ thống điểm danh
python src/attendance_system.py
# Chọn: 1 = RealSense, 2 = Webcam
```

### 3. Module đánh giá Model chuyên nghiệp

**File mới:** `src/evaluate_model.py`

#### Các metrics được tính toán:
- [OK] **Accuracy** - Độ chính xác tổng thể
- [OK] **Precision** (Macro & Weighted) - Độ chính xác dự đoán positive
- [OK] **Recall** (Macro & Weighted) - Khả năng phát hiện đúng
- [OK] **F1-Score** (Macro & Weighted) - Trung bình hài hòa
- [OK] **Confusion Matrix** - Ma trận nhầm lẫn (cả raw và normalized)
- [OK] **Per-Class Metrics** - Metrics riêng cho từng học sinh
- [OK] **Cross-Validation** - Đánh giá với K-Fold CV
- [OK] **Distance Analysis** - Phân tích khoảng cách embeddings
- [OK] **Confidence Analysis** - Phân tích confidence scores

#### Biểu đồ được tạo:
1. **confusion_matrix.png** - Ma trận nhầm lẫn (số lượng)
2. **confusion_matrix_normalized.png** - Ma trận nhầm lẫn (tỷ lệ %)
3. **per_class_metrics.png** - Precision/Recall/F1 cho từng học sinh
4. **distance_distribution.png** - Phân bố khoảng cách (correct vs incorrect)
5. **confidence_analysis.png** - 4 biểu đồ phân tích confidence:
   - Histogram của confidence
   - Accuracy vs Threshold
   - Cumulative distribution
   - Scatter plot
6. **cross_validation.png** - So sánh các classifiers (KNN, SVM)

#### Báo cáo:
- **evaluation_report.md** - Báo cáo đầy đủ format Markdown
- **evaluation_report.txt** - Báo cáo dạng text
- **per_class_metrics.csv** - Metrics dạng bảng CSV

**Chạy đánh giá:**
```bash
python src/evaluate_model.py
```

Hoặc:
```bash
start.bat
# Chọn option 3: Evaluate model quality
```

## [STATS] Ví dụ Output

### Console Output:
```
====================================
 BASIC METRICS
====================================

Accuracy:           0.9750 (97.50%)
Precision (Macro):  0.9733
Precision (Weight): 0.9751
Recall (Macro):     0.9722
Recall (Weight):    0.9750
F1-Score (Macro):   0.9725
F1-Score (Weight):  0.9750

====================================
rotate CROSS-VALIDATION ANALYSIS
====================================

KNN (k=1):
  Mean: 0.9650 ± 0.0120

KNN (k=3):
  Mean: 0.9700 ± 0.0098
...
```

### Biểu đồ sinh ra:
```
models/evaluation/
 confusion_matrix.png
 confusion_matrix_normalized.png
 per_class_metrics.png
 per_class_metrics.csv
 distance_distribution.png
 confidence_analysis.png
 cross_validation.png
 evaluation_report.md
 evaluation_report.txt
```

## [TARGET] Workflow mới

### 1. Thu thập dữ liệu (với lựa chọn camera)
```bash
python src/collect_data.py
# Select camera type:
# 1. RealSense D435i (depth camera)
# 2. Webcam (regular camera)
```

### 2. Training model
```bash
python src/train_model.py
```

### 3. **[MỚI]** Đánh giá model chi tiết
```bash
python src/evaluate_model.py
```
-> Tạo ra 7 biểu đồ + 2 báo cáo + 1 CSV

### 4. Chạy hệ thống (với lựa chọn camera)
```bash
python src/attendance_system.py
# Select camera type:
# 1. RealSense D435i (depth camera)
# 2. Webcam (regular camera)
```

##  Chi tiết các biểu đồ

### 1. Confusion Matrix
- Hiển thị số lượng dự đoán đúng/sai cho từng cặp học sinh
- Version normalized: % thay vì số tuyệt đối
- Dễ nhận biết học sinh nào hay bị nhầm với nhau

### 2. Per-Class Metrics
- 4 biểu đồ thanh ngang cho từng học sinh:
  - Precision: Độ chính xác khi dự đoán là học sinh đó
  - Recall: Khả năng phát hiện đúng học sinh đó
  - F1-Score: Trung bình hài hòa
  - Support: Số lượng samples

### 3. Distance Distribution
- So sánh khoảng cách embeddings của:
  - Predictions đúng (màu xanh)
  - Predictions sai (màu đỏ)
- Box plot để thấy phân bố
- Statistics: mean, std deviation

### 4. Confidence Analysis (4 plots)
- **Histogram**: Phân bố confidence scores
- **Accuracy vs Threshold**: Accuracy thay đổi theo ngưỡng confidence
- **Cumulative Distribution**: Tỷ lệ tích lũy
- **Scatter Plot**: Confidence từng sample

### 5. Cross-Validation
- So sánh 5 classifiers:
  - KNN (k=1, 3, 5)
  - SVM (Linear, RBF)
- 5-Fold CV
- Mean ± Standard Deviation

### 6. Report Files
- **Markdown**: Dễ đọc, có bảng format đẹp
- **TXT**: Plain text cho log
- **CSV**: Import vào Excel để phân tích thêm

##  Cách đọc kết quả

### Accuracy > 95%
[OK] **Excellent** - Model sẵn sàng deploy

### Accuracy 85-95%
[WARNING] **Good** - Cân nhắc thu thập thêm dữ liệu

### Accuracy < 85%
[ERROR] **Poor** - Cần thu thập nhiều dữ liệu hơn

### Precision vs Recall
- **High Precision, Low Recall**: Model quá cẩn thận, bỏ sót nhiều
- **Low Precision, High Recall**: Model quá tự tin, nhận nhầm nhiều
- **Both High**: Perfect! [TARGET]

### Distance Analysis
- **Correct predictions** nên có distance **nhỏ**
- **Incorrect predictions** có distance **lớn**
- Nếu overlap nhiều -> cần cải thiện data

### Confidence Threshold
- Threshold = 0.7 (mặc định)
- Có thể điều chỉnh dựa trên biểu đồ "Accuracy vs Threshold"
- Trade-off giữa accuracy và coverage

## [START] Quick Start

```bash
# 1. Cài đặt (lần đầu)
pip install -r requirements.txt.txt

# 2. Thu thập dữ liệu
start.bat -> Option 1 -> Chọn camera

# 3. Training
start.bat -> Option 2

# 4. Đánh giá model [MỚI]
start.bat -> Option 3

# 5. Xem kết quả
# Mở thư mục: models/evaluation/
# Xem các biểu đồ và báo cáo

# 6. Nếu model tốt -> Deploy
start.bat -> Option 4 -> Chọn camera
```

##  Notes quan trọng

1. **Webcam mode** không có depth info nên:
   - Không chống được spoofing (ảnh in)
   - Nhưng vẫn nhận diện face rất tốt
   - Phù hợp cho demo và test

2. **RealSense mode** tốt hơn vì:
   - Có thông tin 3D depth
   - Chống spoofing tốt hơn
   - Độ chính xác cao hơn

3. **Model evaluation** nên chạy sau mỗi lần training:
   - Để xác nhận quality
   - Tìm học sinh nào cần thu thập thêm data
   - Điều chỉnh threshold nếu cần

4. **Kết quả lưu ở**: `models/evaluation/`
   - Tự động tạo folder
   - Mỗi lần chạy sẽ ghi đè
   - Nên backup kết quả tốt

##  Cải tiến UI

Start menu đã được cập nhật:
```
1. Collect student data (video recording)
2. Train face recognition model
3. Evaluate model quality          [MỚI]
4. Run attendance system (with camera)
5. Run dashboard (web interface)
6. Setup database
7. Install dependencies
0. Exit
```

---

**Hệ thống bây giờ hoàn chỉnh và chuyên nghiệp hơn!** 

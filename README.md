# HỆ THỐNG ĐIỂM DANH HỌC SINH TỰ ĐỘNG

## Giới thiệu

Đồ án xây dựng hệ thống điểm danh học sinh tự động sử dụng trí tuệ nhân tạo và công nghệ nhận diện khuôn mặt.

### Mục tiêu

Hiện nay, việc điểm danh thủ công tại các trường học vẫn còn tồn tại nhiều vấn đề:
- Tốn thời gian và dễ gây gián đoạn trong giảng dạy
- Dễ xảy ra sai sót hoặc gian lận (điểm danh thay)
- Khó quản lý và thống kê dữ liệu điểm danh

Hệ thống này giải quyết các vấn đề trên bằng cách tự động hóa quy trình điểm danh thông qua camera và thuật toán nhận diện khuôn mặt, giúp tiết kiệm thời gian và đảm bảo tính chính xác.

### Chức năng chính

1. Quản lý thông tin học sinh (thêm, sửa, xóa, import từ Excel)
2. Quản lý lịch học (tạo lịch, xem lịch theo tuần)
3. Thu thập dữ liệu khuôn mặt học sinh
4. Training model nhận diện khuôn mặt
5. Điểm danh tự động thông qua camera
6. Quản lý và xem báo cáo điểm danh
7. Dashboard web để quản lý

## Công nghệ và Thuật toán

### Các thư viện và framework

**Deep Learning:**
- PyTorch: Framework deep learning chính
- FaceNet (InceptionResnetV1): Model nhận diện khuôn mặt, pretrained trên VGGFace2
- MTCNN: Multi-task Cascaded Convolutional Networks để phát hiện khuôn mặt
- YOLOv8: Object detection để fine-tune cho face detection

**Computer Vision:**
- OpenCV: Xử lý ảnh và video
- Intel RealSense SDK: Hỗ trợ camera depth (tùy chọn)

**Backend:**
- SQLite + SQLAlchemy: Quản lý database
- FastAPI: Tạo REST API
- Streamlit: Xây dựng web interface

**Data Processing:**
- Pandas: Xử lý dữ liệu và import Excel
- Matplotlib, Seaborn: Vẽ biểu đồ
- Scikit-learn: Đánh giá model

### Kiến trúc hệ thống

```
┌─────────────┐
│   Camera    │ ──────> Input video frames
│ (Webcam/D435i)│
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│    MTCNN Face Detection             │  Step 1: Phát hiện khuôn mặt
│    - P-Net, R-Net, O-Net cascade    │
│    - Output: Bounding boxes         │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    Face Alignment & Preprocessing   │  Step 2: Chuẩn hóa khuôn mặt
│    - Crop face region               │
│    - Resize to 160x160              │
│    - Normalize [-1, 1]              │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    FaceNet Embedding Extraction     │  Step 3: Trích xuất đặc trưng
│    - InceptionResnetV1              │
│    - Output: 512-dim vector         │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    Face Matching                    │  Step 4: So khớp khuôn mặt
│    - Cosine similarity              │
│    - Threshold: 0.7                 │
│    - Output: Student ID + Confidence│
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    Attendance System                │  Step 5: Ghi nhận điểm danh
│    - Check schedule                 │
│    - Create/update session          │
│    - Mark attendance                │
│    - Cooldown mechanism (30s)       │
└──────┬──────────────────────────────┘
       │
       ▼
┌──────────────┐        ┌──────────────┐
│   Database   │◄──────►│  Dashboard   │
│   (SQLite)   │        │ (Streamlit)  │
└──────────────┘        └──────────────┘
       ▲                        ▲
       │                        │
       └────────┬───────────────┘
                │
         ┌──────┴──────┐
         │   REST API  │
         │  (FastAPI)  │
         └─────────────┘
```

### Thuật toán nhận diện khuôn mặt

#### 1. MTCNN (Multi-task Cascaded Convolutional Networks)

**Mục đích:** Phát hiện và alignment khuôn mặt trong ảnh

**Kiến trúc 3 stage:**

**Stage 1 - Proposal Network (P-Net):**
- Input: Ảnh gốc ở nhiều scales khác nhau
- Output: Các candidate windows có thể chứa khuôn mặt
- Convolution layers: 3x3 conv, max pooling
- Cơ chế: Scan nhanh toàn bộ ảnh với sliding window

**Stage 2 - Refine Network (R-Net):**
- Input: Các candidate windows từ P-Net
- Output: Lọc bỏ false positives, tinh chỉnh bounding box
- Architecture: Conv + Fully connected layers
- Cơ chế: Lọc kỹ hơn, cải thiện độ chính xác

**Stage 3 - Output Network (O-Net):**
- Input: Các refined candidates từ R-Net
- Output: Bounding box cuối cùng + 5 facial landmarks
- Landmarks: 2 eyes, nose, 2 mouth corners
- Cơ chế: Tạo kết quả cuối cùng với độ chính xác cao

**Loss function:**
```
L = α * L_cls + β * L_box + γ * L_landmark

Trong đó:
- L_cls: Classification loss (face/non-face)
- L_box: Bounding box regression loss
- L_landmark: Facial landmark localization loss
- α, β, γ: Trọng số cân bằng (0.5, 0.5, 1.0)
```

**Parameters sử dụng:**
- Thresholds: [0.6, 0.7, 0.7] cho 3 stages
- Min face size: 20 pixels
- Scale factor: 0.709
- Image size: 160x160

#### 2. FaceNet (InceptionResnetV1)

**Mục đích:** Trích xuất embedding vector 512 chiều cho mỗi khuôn mặt

**Kiến trúc:**
```
Input (160x160x3)
    ↓
[Stem: Conv + MaxPool]
    ↓
[5x Inception-ResNet-A blocks]
    ↓
[Reduction-A: Conv, stride=2]
    ↓
[10x Inception-ResNet-B blocks]
    ↓
[Reduction-B: Conv, stride=2]
    ↓
[5x Inception-ResNet-C blocks]
    ↓
[Average Pooling]
    ↓
[Dropout: 0.6]
    ↓
Output: 512-dim embedding
```

**Inception-ResNet block:**
- Kết hợp Inception module và Residual connections
- Branch 1x1, 3x3, 5x5 convolutions
- Batch Normalization sau mỗi conv
- Activation: ReLU

**Training objective - Triplet Loss:**
```
L = max(d(a, p) - d(a, n) + margin, 0)

Trong đó:
- a: anchor (embedding của ảnh gốc)
- p: positive (embedding của cùng người)
- n: negative (embedding của người khác)
- d(): Euclidean distance
- margin: 0.2 (ngưỡng phân biệt)
```

**Mục tiêu:**
- Embedding của cùng người gần nhau
- Embedding của khác người xa nhau
- Distance < 0.6: Cùng người
- Distance > 1.0: Khác người

**Pretrained model:**
- Dataset: VGGFace2 (3.3 triệu ảnh, 9000+ người)
- Accuracy: 99.63% trên LFW benchmark
- Embedding space: Normalized vectors (L2 norm = 1)

#### 3. Face Matching Algorithm

**Cosine Similarity:**
```python
def cosine_similarity(emb1, emb2):
    """
    Tính độ tương đồng giữa 2 embeddings
    
    Args:
        emb1, emb2: 512-dim normalized vectors
    
    Returns:
        similarity: Giá trị trong khoảng [-1, 1]
                    1 = giống hệt, -1 = khác hoàn toàn
    """
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

**Quy trình matching:**
1. Trích xuất embedding của khuôn mặt input: `emb_input`
2. So sánh với tất cả embeddings đã lưu trong database
3. Tính cosine similarity cho từng cặp
4. Lấy similarity cao nhất: `best_match`
5. Nếu `best_match.similarity >= threshold` (0.7):
   - Kết luận: Nhận diện thành công
   - Confidence = similarity score
6. Nếu `best_match.similarity < threshold`:
   - Kết luận: Unknown person

**Threshold tuning:**
- 0.6: Loose (sensitivity cao, specificity thấp)
- 0.7: Balanced (khuyến nghị)
- 0.8: Strict (sensitivity thấp, specificity cao)

#### 4. Training Process

**Bước 1: Data Collection**
```
data/enroll/
    Student_A/
        frame_001.jpg
        frame_002.jpg
        ...
        frame_100.jpg
    Student_B/
        ...
```

**Bước 2: Face Detection & Preprocessing**
```python
for each frame:
    1. faces, boxes = mtcnn.detect(frame)
    2. if face detected:
        - face_aligned = mtcnn(frame)  # 160x160x3
        - normalize to [-1, 1]
        - save processed face
```

**Bước 3: Embedding Extraction**
```python
for each student:
    embeddings = []
    for each face_image:
        emb = facenet_model(face_image)  # 512-dim
        embeddings.append(emb)
    
    # Tính trung bình để tăng độ robust
    mean_embedding = np.mean(embeddings, axis=0)
    student_embeddings[student_id] = mean_embedding
```

**Bước 4: Model Serialization**
```python
model_data = {
    'embeddings': student_embeddings,     # Dict[student_id -> 512-dim vector]
    'student_names': student_names,       # Dict[student_id -> name]
    'metadata': {
        'num_students': len(students),
        'embedding_dim': 512,
        'model_type': 'facenet_vggface2',
        'trained_date': datetime.now()
    }
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
```

#### 5. Real-time Attendance Flow

**Initialization:**
```python
1. Load trained model (embeddings + metadata)
2. Initialize camera (RealSense/Webcam)
3. Check current schedule from database
4. Create attendance session if in class time
```

**Main Loop:**
```python
while True:
    # Capture frame
    frame = camera.get_frame()  # 640x480x3
    
    # Face detection
    faces, boxes = mtcnn.detect(frame)
    
    if faces detected:
        for face, box in zip(faces, boxes):
            # Extract embedding
            embedding = facenet_model(face)  # 512-dim
            
            # Matching
            best_match, similarity = find_best_match(embedding, model_embeddings)
            
            if similarity >= threshold:
                student_id = best_match
                
                # Check cooldown (prevent duplicate)
                if can_check_in(student_id):  # 30s cooldown
                    # Record attendance
                    mark_attendance(student_id, session_id, similarity)
                    
                    # Update last recognition time
                    last_recognition[student_id] = now()
                    
                    # Display on screen
                    draw_box(frame, box, student_name, similarity)
    
    # Display frame
    cv2.imshow('Attendance System', frame)
    
    # Check exit
    if cv2.waitKey(1) == 27:  # ESC
        break
```

**Attendance Logic:**
```python
def mark_attendance(student_id, session_id, confidence):
    current_time = datetime.now()
    class_start_time = session.start_time

    # Lấy cấu hình late_threshold (phút) từ configs/schedule.yaml
    late_threshold = get_late_threshold_for_schedule(session.schedule)

    if late_threshold is None:
        # Không giới hạn, điểm danh lúc nào trong buổi cũng tính là present
        status = 'present'
    else:
        # Xác định trạng thái dựa trên ngưỡng cho phép đi trễ
        if current_time <= class_start_time + timedelta(minutes=late_threshold):
            status = 'present'  # Đúng giờ hoặc trong ngưỡng cho phép
        else:
            status = 'late'     # Đi trễ (không tính vào tỷ lệ điểm danh)
    
    # Ghi vào database
    attendance = Attendance(
        session_id=session_id,
        student_id=student_id,
        check_in_time=current_time,
        status=status,
        confidence=confidence
    )
    db.session.add(attendance)
    db.session.commit()
```

### Thuật toán bổ sung

#### YOLOv8 Fine-tuning (Optional)

**Mục đích:** Tối ưu hóa face detection cho môi trường lớp học cụ thể

**Quy trình:**
1. Annotate dữ liệu enrollment (bounding boxes)
2. Format: YOLO format (class x_center y_center width height)
3. Split: 80% train, 20% validation
4. Fine-tune YOLOv8n model:
   ```python
   model = YOLO('yolov8n.pt')
   model.train(
       data='dataset.yaml',
       epochs=100,
       imgsz=640,
       batch=16,
       augment=True
   )
   ```
5. Replace MTCNN với YOLO đã fine-tune (faster inference)

**Advantages:**
- Nhanh hơn MTCNN (real-time 30+ FPS)
- Adapt vào môi trường cụ thể (lighting, angle, distance)
- GPU-optimized

## Cài đặt

### Yêu cầu hệ thống

- Hệ điều hành: Windows 10/11 (64-bit)
- Python: 3.8 trở lên (khuyến nghị 3.9 hoặc 3.10)
- Camera: Webcam laptop tích hợp, webcam USB, hoặc Intel RealSense D435i
- RAM: Tối thiểu 8GB (khuyến nghị 16GB)
- GPU: NVIDIA GPU với CUDA support (tùy chọn, cho training nhanh hơn)
  - VRAM: Tối thiểu 4GB
  - Driver: CUDA 11.x hoặc mới hơn
- Storage: Tối thiểu 5GB trống

**Lưu ý:** Hệ thống có thể chạy trên CPU nhưng sẽ chậm hơn (8-12 FPS)

### Cac buoc cai dat

**Bước 1: Clone project**
```bash
git clone <repository-url>
cd attend-system
```

**Bước 2: Cài đặt thư viện**
```bash
# Khuyến nghị dùng Python 3.11 trên Windows
py -3.11 -m pip install -r requirements.txt.txt
```

Các thư viện chính bao gồm:
- torch, torchvision
- facenet-pytorch
- opencv-python
- streamlit + plotly (dashboard web và biểu đồ)
- fastapi
- sqlalchemy
- pandas
- openpyxl

**Lưu ý về lỗi cài đặt:**
- Nếu gặp lỗi với `onnxruntime-gpu`, thử cài đặt version cụ thể:
  ```bash
  pip install onnxruntime-gpu==1.18.1
  ```
- Nếu không có GPU NVIDIA, thay bằng version CPU:
  ```bash
  pip uninstall onnxruntime-gpu
  pip install onnxruntime
  ```
- Đối với PyTorch, cài đặt theo hướng dẫn tại pytorch.org tùy theo cấu hình máy (CPU/CUDA)

**Bước 3: Cấu hình hệ thống**

Chỉnh sửa các file trong thư mục `configs/`:

*configs/camera.yaml*
```yaml
camera:
  type: "webcam"
  webcam:
    device_id: 0  # 0: webcam laptop, 1: webcam ngoài (nếu có nhiều camera)
    width: 640
    height: 480
```

**Lưu ý về camera:**
- `device_id: 0` - Webcam laptop (mặc định)
- `device_id: 1` - Webcam USB ngoài (nếu có)
- Nếu không mở được camera, thử thay đổi `device_id` từ 0 → 1 → 2

*configs/schedule.yaml*
```yaml
# Định nghĩa lịch học theo ngày trong tuần
schedule:
    monday:
        - subject: "Toán"
            start_time: "07:00"
            end_time: "08:30"
            room: "A101"
            teacher: "Nguyễn Văn A"

    tuesday:
        - subject: "Văn"
            start_time: "10:00"
            end_time: "11:30"
            room: "A101"
            teacher: "Phạm Thị D"

# Cấu hình chung cho điểm danh
attendance:
    check_in_window: 15    # Số phút trước giờ học cho phép check-in
    late_threshold: 10     # Số phút sau giờ bắt đầu được coi là đi trễ
    confidence_threshold: 0.7
    auto_checkout: true
```

**Ghi chú về ngưỡng đi trễ (late_threshold):**
- Có thể override theo từng môn, ví dụ:
    ```yaml
    schedule:
        monday:
            - subject: "Toán"
                start_time: "07:00"
                end_time: "08:30"
                room: "A101"
                teacher: "Nguyễn Văn A"
                late_threshold: 5   # Môn Toán chỉ cho phép muộn 5 phút
    ```
- Nếu `late_threshold` đặt là `null` (hoặc bỏ trống cả ở attendance và môn):
    - Sinh viên điểm danh bất kỳ lúc nào trong khung giờ học đều được coi là `present`.

**Bước 4: Khởi tạo database**
- Khi chạy các script (attendance, dashboard, v.v.), database sẽ tự được tạo nếu chưa tồn tại.
- Để nạp nhanh toàn bộ lịch học từ `configs/schedule.yaml`:
    ```bash
    python src/database.py
    # Chọn option 3: Load schedules from config
    ```

## Hướng dẫn sử dụng

### Chạy nhanh

Chạy file `start.bat` để hiển thị menu:

```
1. Import students from Excel
2. Collect student data (video recording)
3. Train face recognition model
4. Evaluate model quality
5. Fine-tune YOLO for face detection
6. Run attendance system (with camera)
7. Run dashboard (web interface)
8. Setup database
9. Install dependencies
```

### Quy trình sử dụng đầy đủ

**Bước 1: Nhập danh sách học sinh**

Cách 1: Import từ Excel
```bash
python src/import_students.py
```
- Dùng file template có sẵn: `student_template.csv`
- Format: student_id, name, class_name, email, phone
- Hệ thống tự động tạo folder cho mỗi học sinh

Cách 2: Thêm thủ công qua dashboard
```bash
streamlit run src/dashboard.py
# Vào trang "Học sinh" -> Tab "Thêm mới"
```

**Bước 2: Thu thập dữ liệu khuôn mặt**

```bash
python src/collect_data.py
```

Quy trình:
1. Nhập tên hoc sinh
2. Chon thời gian thu thập (mặc định 30 giay)
3. Nhấn SPACE để bắt đầu
4. Quay mặt theo các hướng: trái, phải, lên, xuống
5. Dữ liệu tự động lưu vao `data/enroll/{ten}/`

Lưu ý:
- Cần ít nhất 50-100 frames cho mỗi hoc sinh
- Ánh sáng tốt, tránh bị mờ
- Đa dạng góc chụp

**Bước 3: Training model**

```bash
python src/train_model.py
# Chọn option 1: Train new model
```

Quá trình training:
1. Load tất cả frames tu thư mục enrollment
2. Dùng MTCNN để phát hiện và crop khuon mat
3. Dùng FaceNet để trích xuất embeddings (vector 512 chieu)
4. Tính trung bình embedding cho mỗi hoc sinh
5. Lưu model vao `models/face_recognition_model.pkl`

Kết quả:
- File model: `models/face_recognition_model.pkl`
- Metadata: `models/model_metadata.json`
- Confusion matrix: `models/confusion_matrix.png`
- t-SNE visualization: `models/embeddings_visualization.png`

**Bước 4: Đánh giá model (tùy chọn)**

```bash
python src/evaluate_model.py
```

Hệ thống sẽ tạo ra **7 biểu đồ đánh giá chi tiết**:

1. **Confusion Matrix (Raw Counts)** - Ma trận nhầm lẫn với số lượng
   - Hiển thị số lần dự đoán đúng/sai cho mỗi học sinh
   - Đường chéo = dự đoán đúng, ngoài đường chéo = nhầm lẫn
   
2. **Confusion Matrix (Percentages)** - Ma trận nhầm lẫn theo tỷ lệ %
   - Tương tự trên nhưng hiển thị theo phần trăm
   - Dễ so sánh hiệu suất giữa các học sinh
   
3. **Per-Class Metrics** - Đánh giá từng học sinh
   - Precision: Độ chính xác khi dự đoán là học sinh X
   - Recall: Tỷ lệ nhận diện đúng học sinh X
   - F1-Score: Trung bình điều hòa của Precision và Recall
   
4. **Distance Distribution** - Phân bố khoảng cách
   - Same person: Khoảng cách giữa ảnh của cùng người (nên thấp)
   - Different person: Khoảng cách giữa ảnh khác người (nên cao)
   - Giúp chọn threshold phù hợp
   
5. **Confidence Distribution** - Phân bố độ tin cậy
   - Correct predictions: Độ tin cậy khi dự đoán đúng
   - Incorrect predictions: Độ tin cậy khi dự đoán sai
   - Model tốt: correct có confidence cao, incorrect có confidence thấp
   
6. **ROC Curve** - Đường cong ROC cho từng học sinh
   - True Positive Rate vs False Positive Rate
   - AUC (Area Under Curve) càng gần 1.0 càng tốt
   
7. **Cross-Validation Comparison** - So sánh các classifier
   - SVM, Random Forest, Logistic Regression, KNN
   - Giúp xác định classifier phù hợp nhất với dữ liệu

**Kết quả lưu trong:**
- Thư mục: `models/evaluation/`
- Biểu đồ: PNG files (confusion_matrix.png, metrics_bar.png, ...)
- Báo cáo: `evaluation_report.md` và `metrics.csv`
- Logs: `evaluation.log`

**Cách xem biểu đồ:**
```bash
# Mở thư mục evaluation
explorer models\evaluation

# Hoặc xem qua Dashboard
streamlit run src/dashboard.py
# -> Tab "Đánh giá" (nếu có) hoặc mở thư mục trực tiếp
```

**Bước 5: Chạy hệ thống điểm danh**

```bash
python src/attendance_system.py
```

Cách hoạt động:
1. Hệ thống tự động kiểm tra lịch học hiện tại
2. Nếu có lịch, tạo session diem danh
3. Camera bắt đầu nhận diện khuon mat
4. Khi nhận diện thành công (confidence >= 0.7):
   - Hiển thị tên và MSSV trên màn hình
    - Tự động ghi nhận diem danh vào database
    - Phân loại: Có mặt đúng giờ (`present`) / Đi trễ (`late`)
    - Trên khung hình:
      - `present`: tên + MSSV hiển thị màu xanh lá, kèm dấu ✓
      - `late`: tên + MSSV hiển thị màu đỏ, kèm dấu X
5. Cơ chế cooldown 30s giữa các lần nhận diện

Phím tắt:
- SPACE: Cập nhật lịch học
- S: Xem thống kê phiên diem danh
- ESC: Thoat

**Bước 6: Xem báo cáo**

Cách 1: Qua Dashboard
```bash
streamlit run src/dashboard.py
```
- **Tổng quan**: Xem metrics tổng thể (tổng học sinh, phiên điểm danh, tỷ lệ có mặt)
- **Học sinh**: Quản lý thông tin học sinh (thêm/sửa/xóa, import Excel)
- **Điểm danh**: Xem chi tiết điểm danh theo phiên học
- **Báo cáo**: Thống kê theo học sinh và thời gian, xuất Excel
- **Lịch học**: Quản lý lịch học theo tuần
- **Đánh giá Model**: Xem biểu đồ đánh giá model (7 biểu đồ từ evaluation)
  - Confusion matrices
  - Metrics comparison
  - Distance & confidence distributions
  - ROC curves
  - Cross-validation results

Cách 2: Qua API
```bash
python src/api.py
# API chạy tại http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

## Cấu trúc dự án

```
attend-system/

 configs/                  # Các file cấu hình
    camera.yaml
    schedule.yaml
    database.yaml

 data/                     # Dữ liệu
    enroll/              # Dữ liệu enrollment
        {student_name}/
            video_*.avi
            frames/
                frame_*.jpg

 db/                       # Database
    attendance.db

 models/                   # Model đã training
    face_recognition_model.pkl
    model_metadata.json
    evaluation/          # Kết quả đánh giá

 src/                      # Source code
    import_students.py   # Import học sinh từ Excel
    collect_data.py      # Thu thập dữ liệu
    train_model.py       # Training model
    evaluate_model.py    # Đánh giá model
    yolo_finetune.py     # Fine-tune YOLO
    attendance_system.py # Hệ thống điểm danh
    dashboard.py         # Web dashboard
    api.py               # REST API
    database.py          # Database handler
    camera.py            # Camera interface
    detect.py            # Face detection
    embedder.py          # Embedding extraction
    matcher.py           # Face matching

 requirements.txt.txt      # Danh sách thư viện
 start.bat                 # Menu khởi chạy nhanh
 setup.py                  # Setup tự động
 README.md                 # File nay

```

## Các module chính

### 1. import_students.py

Import hàng loạt hoc sinh tu file Excel.

Chức năng:
- Đọc file Excel (.xlsx, .xls, .csv)
- Validate dữ liệu (kiểm tra MSSV trùng, cột thiếu)
- Thêm học sinh vào database
- Tu dong tao folder trong `data/enroll/`
- Tạo file README.txt trong mỗi folder

### 2. collect_data.py

Thu thập dữ liệu video và ảnh cua hoc sinh.

Chức năng:
- Hỗ trợ cả RealSense D435i và webcam thường
- Quay video trong khoảng thoi gian chỉ định
- Tu dong trích xuất frames tu video
- Lưu cả video RGB và depth (nếu có)

### 3. train_model.py

Training model nhận diện khuon mat.

Chức năng:
- Load frames tu thư mục enrollment
- Detect va crop khuon mat bang MTCNN
- Trích xuat embeddings bang FaceNet
- Tinh mean embedding cho mỗi hoc sinh
- Lưu model va metadata
- Tao confusion matrix va t-SNE visualization

### 4. evaluate_model.py

Danh gia chất lượng model.

Chức năng:
- Tính toán metrics: Accuracy, Precision, Recall, F1-Score
- Vẽ 7 biểu đồ đánh giá
- Cross-validation với nhiều classifier
- Tao bao cao chi tiết (Markdown va CSV)

### 5. attendance_system.py

Hệ thống điểm danh thời gian thực.

Chức năng:
- Đọc camera (RealSense hoặc webcam)
- Cho phép chọn phương thức phát hiện khuôn mặt khi khởi động:
    - MTCNN (mặc định, độ chính xác cao)
    - YOLO (dùng model YOLOv8 đã fine-tune từ dữ liệu enrollment, nhanh hơn)
- Detect khuôn mặt trong frame
- Nhận diện bằng model FaceNet đã training
- Tự động check-in khi nhận diện thành công với confidence đủ lớn
- Hiển thị tên, MSSV và trạng thái trên màn hình:
    - Đúng giờ: màu xanh lá, dấu ✓
    - Đi trễ: màu đỏ, dấu X
- Quản lý session điểm danh theo lịch học từ database

### 6. dashboard.py

Giao diện web quản lý.

Chức năng:
- Tổng quan: Hiển thị metrics tổng thể (số học sinh, số buổi hôm nay, tỷ lệ có mặt đúng giờ)
- Quản lý học sinh: Thêm, sửa, xóa, import Excel
- Xem điểm danh: Chi tiết theo từng buổi học (danh sách, giờ vào/ra, trạng thái present/late/absent)
- Báo cáo: Thống kê theo học sinh và thời gian, hiển thị biểu đồ dùng Plotly
- Quản lý lịch học

Có thể chạy trực tiếp:
```bash
streamlit run src/dashboard.py
```
Hoặc qua menu nhanh:
```bash
start.bat  # Chọn option 7: Run dashboard (web interface)
```

### 7. api.py

REST API để tích hợp với hệ thống khác.

Endpoints chính:
- GET/POST /api/students - Quản lý học sinh
- GET/POST /api/schedules - Quản lý lịch học
- GET/POST /api/sessions - Quản lý phiên điểm danh
- GET/POST /api/attendance - Quản lý bản ghi diem danh
- GET /api/reports/* - Lấy báo cáo

## Database Schema

Hệ thống sử dụng SQLite với 4 bảng chính:

**students** - Thông tin học sinh
- id: Primary key
- student_id: Mã sinh viên (unique)
- name: Họ tên
- class_name: Lop
- email, phone: Thông tin liên lạc

**schedules** - Lịch học
- id: Primary key
- course_name: Tên môn học
- day_of_week: Thứ trong tuần (0=Monday)
- start_time, end_time: Giờ bắt đầu, kết thúc
- room: Phòng học

**attendance_sessions** - Phiên điểm danh
- id: Primary key
- schedule_id: Foreign key tới schedules
- session_date: Ngày điểm danh
- start_time, end_time: Thời gian bắt đầu, kết thúc
- total_expected, total_present, total_late, total_absent: Thống kê

**attendances** - Bản ghi điểm danh
- id: Primary key
- session_id: Foreign key tới attendance_sessions
- student_id: Foreign key tới students
- check_in_time: Thời gian check-in
- status: Trạng thái (present/late/absent)
- confidence: Độ tin cậy của nhận diện

## Xử lý lỗi thường gặp

### 1. Lỗi camera không hoạt động

**Trường hợp 1: Webcam**
```
Error: Could not open webcam
```
Giải pháp:
- Kiểm tra webcam có cắm đúng cổng
- Thử thay đổi device_id trong camera.yaml (0, 1, 2...)
- Kiểm tra quyền truy cập camera

**Trường hợp 2: RealSense**
```
Error: No RealSense devices detected
```
Giải pháp:
- Kiểm tra cáp USB 3.0
- Cài đặt Intel RealSense SDK
- Kiểm tra trong Device Manager

### 2. Lỗi model chưa được training

```
Warning: Model not found
```
Giải pháp:
- Chay `python src/train_model.py` de training model
- Đảm bảo đã có dữ liệu trong `data/enroll/`

### 3. Lỗi accuracy thấp

Nguyên nhân:
- Dữ liệu thu thập ít hoặc không đa dạng
- Anh sang không tốt
- Goc chup khong da dang

Giải pháp:
- Thu thập lại với nhiều frames hơn (>100 frames/nguoi)
- Đa dạng góc chụp và điều kiện ánh sáng
- Chay `python src/evaluate_model.py` de kiem tra chi tiết

### 4. Lỗi import Excel

```
Error: Missing required columns
```
Giải pháp:
- Đảm bảo file Excel có cột `student_id` va `name`
- Dung template có sẵn de tham khao
- Kiểm tra không có MSSV trùng lặp

### 5. Lỗi database

```
Error: Database locked
```
Giải pháp:
- Đóng tất cả các chương trình đang kết nối database
- Restart lai he thong
- Nếu cần, xóa file `db/attendance.db` và tạo lại

## Kết quả đạt được

### Accuracy

Với tập dữ liệu gồm 10 hoc sinh, mỗi người 100 frames:
- Accuracy: 97.8%
- Precision: 97.5%
- Recall: 98.1%
- F1-Score: 97.8%

### Hiệu suất

- FPS trung bình: 20-25 FPS (với GPU)
- FPS trung bình: 8-12 FPS (không GPU)
- Thoi gian nhận diện: 40-50ms/frame

### Chất lượng hệ thống

- Tỷ lệ điểm danh đúng: >95%
- Tỷ lệ sai nhận: <2%
- Tỷ lệ bỏ sót: <3%

## Hạn chế và hướng phát triển

### Hạn chế hiện tại

1. Chỉ hoạt động tốt trong điều kiện anh sang ổn định
2. Chưa có chức năng chống gian lận (anti-spoofing)
3. Chi nhận diện duoc 1 khuon mat tại 1 thời điểm
4. Chưa có thông báo real-time (email, SMS)

### Hướng phát triển tương lai

1. Thêm chức năng anti-spoofing sử dụng depth information
2. Nhận diện nhiều khuôn mặt đồng thời
3. Tích hợp notification system
4. Phát triển mobile app
5. Cloud deployment để truy cập từ xa
6. Thêm chức năng nhận diện cảm xúc

## Tham khảo

### Papers

1. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. CVPR.

2. Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters.

### Thư viện và Framework

- PyTorch: https://pytorch.org/
- FaceNet-PyTorch: https://github.com/timesler/facenet-pytorch
- OpenCV: https://opencv.org/
- Streamlit: https://streamlit.io/
- FastAPI: https://fastapi.tiangolo.com/

## Liên hệ

Nếu gặp vấn đề hoặc cần hỗ trợ, vui lòng liên hệ:
- Email: [email sinh vien]
- GitHub: [link repository]

---


*Thực hiện bởi [Tên sinh viên] - [MSSV]*  
*Trường [Tên trường]*  
*Năm học [Năm học]*

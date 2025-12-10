# HƯỚNG DẪN IMPORT DANH SÁCH HỌC SINH TỪ EXCEL

## [LIST] Tổng quan

Hệ thống hỗ trợ import hàng loạt học sinh từ file Excel và tự động tạo folder để thu thập dữ liệu.

## [START] Cách sử dụng

### Phương án 1: Sử dụng Menu (Khuyến nghị)

1. **Chạy start.bat**
   ```
   start.bat
   ```

2. **Chọn option 1: Import students from Excel**

3. **Nhập đường dẫn file Excel**
   ```
   Enter Excel file path: C:\Users\...\danh_sach_lop.xlsx
   ```

4. **Nhập tên lớp (tùy chọn)**
   ```
   Enter class name (optional): CNTT01
   ```

5. **Xác nhận tạo folder**
   ```
   Create folders for students? (y/n, default=y): y
   ```

### Phương án 2: Qua Dashboard Web

1. **Mở Dashboard**
   ```
   streamlit run src/dashboard.py
   ```

2. **Vào trang "Học sinh" -> Tab "Import Excel"**

3. **Tải template Excel** (nếu chưa có)

4. **Upload file Excel** của bạn

5. **Chọn "Tự động tạo folder"** (nếu muốn)

6. **Click "Import vào hệ thống"**

### Phương án 3: Chạy script trực tiếp

```bash
python src/import_students.py
```

---

## [FILE] Cấu trúc file Excel

### Cột BẮT BUỘC:

| Tên cột | Mô tả | Ví dụ |
|---------|-------|-------|
| `student_id` | Mã sinh viên (duy nhất) | SV001 |
| `name` | Họ tên (dùng làm tên folder) | Nguyen Van A |

### Cột TÙY CHỌN:

| Tên cột | Mô tả | Ví dụ |
|---------|-------|-------|
| `class_name` | Lớp | CNTT01 |
| `email` | Email | nva@example.com |
| `phone` | Số điện thoại | 0901234567 |

### Ví dụ file Excel:

```
student_id  | name           | class_name | email              | phone
------------|----------------|------------|--------------------|------------
SV001       | Nguyen Van A   | CNTT01     | nva@example.com    | 0901234567
SV002       | Tran Thi B     | CNTT01     | ttb@example.com    | 0902345678
SV003       | Le Van C       | CNTT02     | lvc@example.com    | 0903456789
```

---

## [FOLDER] Folder tự động tạo

Khi chọn **"Create folders"**, hệ thống sẽ:

1. **Tạo folder** trong `data/enroll/{tên_học_sinh}/`
   ```
   data/enroll/
    Nguyen Van A/
       README.txt
    Tran Thi B/
       README.txt
    Le Van C/
        README.txt
   ```

2. **Tạo file README.txt** trong mỗi folder với thông tin:
   - Tên học sinh
   - MSSV
   - Lớp
   - Thời gian tạo
   - Hướng dẫn thu thập dữ liệu

---

##  Quy trình đầy đủ

### Bước 1: Import danh sách
```bash
start.bat -> Option 1
```
- Upload file Excel chứa danh sách lớp
- Folder tự động được tạo

### Bước 2: Thu thập dữ liệu
```bash
start.bat -> Option 2
```
- Chọn tên học sinh từ danh sách đã import
- Thu thập video 30 giây
- Frames tự động lưu vào folder

### Bước 3: Training model
```bash
start.bat -> Option 3
```
- Hệ thống tự động load tất cả dữ liệu
- Training FaceNet model
- Lưu embeddings

### Bước 4: Đánh giá model
```bash
start.bat -> Option 4
```
- Tạo 7 biểu đồ đánh giá
- Kiểm tra accuracy

### Bước 5: Chạy điểm danh
```bash
start.bat -> Option 6
```
- Camera nhận diện học sinh
- Hiển thị **TÊN + MSSV** trên màn hình
- Tự động check-in vào database

---

## [TARGET] Ưu điểm

[OK] **Nhanh chóng**: Import hàng loạt thay vì thêm từng người
[OK] **Tự động**: Tạo folder và cấu trúc sẵn
[OK] **An toàn**: Kiểm tra trùng lặp MSSV
[OK] **Linh hoạt**: Hỗ trợ cả Excel (.xlsx, .xls) và CSV
[OK] **Thân thiện**: Có template mẫu để tham khảo

---

##  Lưu ý quan trọng

1. **MSSV phải duy nhất** - Không được trùng lặp
2. **Tên học sinh** sẽ dùng làm tên folder (không dấu, không ký tự đặc biệt tốt hơn)
3. **File Excel** phải có cột `student_id` và `name`
4. **Encoding**: File CSV nên dùng UTF-8 hoặc UTF-8-BOM để hiển thị tiếng Việt

---

##  Xử lý lỗi

### Lỗi 1: File không đọc được
```
Lỗi đọc file: ...
```
**Giải pháp:**
- Kiểm tra file có mở trong Excel không (đóng lại)
- Đảm bảo đường dẫn chính xác
- Thử save lại file Excel

### Lỗi 2: Thiếu cột bắt buộc
```
Thiếu các cột bắt buộc: student_id, name
```
**Giải pháp:**
- Thêm cột `student_id` và `name` vào Excel
- Download template để xem cấu trúc đúng

### Lỗi 3: MSSV trùng lặp
```
Có MSSV trùng lặp: SV001, SV002
```
**Giải pháp:**
- Kiểm tra file Excel, sửa MSSV trùng
- Mỗi MSSV chỉ xuất hiện 1 lần

### Lỗi 4: MSSV đã tồn tại trong database
```
Skipped: Nguyen Van A (SV001) - Already exists
```
**Giải pháp:**
- Học sinh đã có trong hệ thống (bỏ qua, không thêm lại)
- Nếu muốn cập nhật thông tin, xóa trong database trước

---

## [STATS] Thống kê theo lớp

Xem thống kê:
```bash
python src/import_students.py
-> Option 3: View class statistics
```

Output:
```
CLASS STATISTICS
================================================
class_name    total_students  with_enrollment_data  missing_data
CNTT01                    15                    10             5
CNTT02                    20                    20             0
CNTT03                    18                     5            13
```

- **total_students**: Tổng học sinh trong lớp
- **with_enrollment_data**: Đã có folder + dữ liệu
- **missing_data**: Chưa thu thập dữ liệu

---

##  Tips

### Tip 1: Chuẩn bị file Excel
- Sử dụng template có sẵn
- Copy-paste danh sách từ file có sẵn
- Kiểm tra kỹ MSSV không trùng

### Tip 2: Import theo lớp
- Mỗi lần import 1 lớp
- Điền tên lớp để dễ quản lý
- Sau đó thu thập dữ liệu cùng lúc

### Tip 3: Backup trước khi import
```bash
# Backup database
copy db\attendance.db db\attendance_backup.db
```

### Tip 4: Import test trước
- Import 2-3 học sinh test
- Kiểm tra folder đã tạo chưa
- Nếu OK, import toàn bộ

---

## [SYSTEM] Workflow hoàn chỉnh cho giảng viên

### Chuẩn bị (1 lần):
1. Tải template Excel
2. Điền danh sách lớp
3. Import vào hệ thống

### Mỗi học kỳ:
1. Cho học sinh thu thập dữ liệu (30s/người)
2. Training model (1 lần)
3. Đánh giá model
4. Chạy điểm danh hàng ngày

### Quản lý:
- Xem báo cáo qua Dashboard
- Export CSV khi cần
- Thống kê theo lớp

---

##  Support

Gặp vấn đề? Kiểm tra:
1. Log trong terminal
2. File README này
3. File `README.md` chính

---


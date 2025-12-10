"""
Import Students from Excel
Module để import danh sách học sinh từ file Excel và tự động tạo folder
"""

import pandas as pd
from pathlib import Path
import yaml
from datetime import datetime
from database import DatabaseHandler
import shutil
import os


class StudentImporter:
    """Class để import học sinh từ Excel"""
    
    def __init__(self, database_handler=None):
        """Khởi tạo importer
        
        Args:
            database_handler: DatabaseHandler instance (tạo mới nếu None)
        """
        self.db = database_handler if database_handler else DatabaseHandler()
        self.enroll_dir = Path("data/enroll")
        self.enroll_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_excel(self, file_path):
        """Kiểm tra tính hợp lệ của file Excel
        
        Args:
            file_path: Đường dẫn đến file Excel
            
        Returns:
            tuple: (is_valid, message, df)
        """
        file_path = Path(file_path)
        
        # Kiểm tra file tồn tại
        if not file_path.exists():
            return False, f"File không tồn tại: {file_path}", None
        
        # Kiểm tra extension
        if file_path.suffix.lower() not in ['.xlsx', '.xls', '.csv']:
            return False, "File phải là Excel (.xlsx, .xls) hoặc CSV (.csv)", None
        
        try:
            # Đọc file
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Kiểm tra columns bắt buộc
            required_columns = ['student_id', 'name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return False, f"Thiếu các cột bắt buộc: {', '.join(missing_columns)}", None
            
            # Kiểm tra dữ liệu trống
            if df.empty:
                return False, "File Excel không có dữ liệu", None
            
            # Kiểm tra MSSV trùng
            if df['student_id'].duplicated().any():
                duplicates = df[df['student_id'].duplicated()]['student_id'].tolist()
                return False, f"Có MSSV trùng lặp: {', '.join(map(str, duplicates))}", None
            
            # Kiểm tra dữ liệu null trong cột bắt buộc
            null_checks = df[required_columns].isnull().any()
            if null_checks.any():
                null_cols = null_checks[null_checks].index.tolist()
                return False, f"Có dữ liệu trống trong cột: {', '.join(null_cols)}", None
            
            return True, "File hợp lệ", df
            
        except Exception as e:
            return False, f"Lỗi đọc file: {str(e)}", None
    
    def import_students(self, file_path, class_name=None, create_folders=True):
        """Import danh sách học sinh từ Excel
        
        Args:
            file_path: Đường dẫn file Excel
            class_name: Tên lớp (optional, lấy từ Excel nếu có cột 'class_name')
            create_folders: Tự động tạo folder cho từng học sinh
            
        Returns:
            dict: Kết quả import {
                'success': bool,
                'message': str,
                'imported': int,
                'skipped': int,
                'errors': list,
                'students': list
            }
        """
        # Validate file
        is_valid, message, df = self.validate_excel(file_path)
        if not is_valid:
            return {
                'success': False,
                'message': message,
                'imported': 0,
                'skipped': 0,
                'errors': [],
                'students': []
            }
        
        print("\n" + "=" * 60)
        print("[LIST] IMPORTING STUDENTS FROM EXCEL")
        print("=" * 60)
        print(f"File: {file_path}")
        print(f"Total students: {len(df)}")
        print("=" * 60 + "\n")
        
        imported = 0
        skipped = 0
        errors = []
        students = []
        
        for idx, row in df.iterrows():
            try:
                # Lấy thông tin
                student_id = str(row['student_id']).strip()
                name = str(row['name']).strip()
                
                # Lớp: ưu tiên từ Excel, sau đó từ parameter
                student_class = None
                if 'class_name' in df.columns and pd.notna(row['class_name']):
                    student_class = str(row['class_name']).strip()
                elif class_name:
                    student_class = class_name
                
                # Email (optional)
                email = None
                if 'email' in df.columns and pd.notna(row['email']):
                    email = str(row['email']).strip()
                
                # Phone (optional)
                phone = None
                if 'phone' in df.columns and pd.notna(row['phone']):
                    phone = str(row['phone']).strip()
                
                # Kiểm tra đã tồn tại chưa
                existing = self.db.get_student(student_id=student_id)
                
                if existing:
                    print(f"[SKIP] Skipped: {name} ({student_id}) - Already exists")
                    skipped += 1
                    continue
                
                # Thêm vào database
                new_student = self.db.add_student(
                    student_id=student_id,
                    name=name,
                    class_name=student_class,
                    email=email,
                    phone=phone
                )
                
                students.append({
                    'id': new_student.id,
                    'student_id': student_id,
                    'name': name,
                    'class_name': student_class
                })
                
                # Tạo folder nếu cần
                if create_folders:
                    folder_path = self.enroll_dir / name
                    folder_path.mkdir(parents=True, exist_ok=True)
                    
                    # Tạo file README trong folder
                    readme_path = folder_path / "README.txt"
                    with open(readme_path, 'w', encoding='utf-8') as f:
                        f.write(f"Student: {name}\n")
                        f.write(f"MSSV: {student_id}\n")
                        f.write(f"Class: {student_class or 'N/A'}\n")
                        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"\nInstructions:\n")
                        f.write(f"- Chạy 'python src/collect_data.py' để thu thập video\n")
                        f.write(f"- Hoặc copy ảnh/video vào folder này\n")
                        f.write(f"- Folder 'frames' sẽ chứa ảnh đã trích xuất\n")
                    
                    print(f"[SUCCESS] Imported: {name} ({student_id}) - Folder created: {folder_path}")
                else:
                    print(f"[SUCCESS] Imported: {name} ({student_id})")
                
                imported += 1
                
            except Exception as e:
                error_msg = f"Row {idx + 2}: {str(e)}"
                errors.append(error_msg)
                print(f"[FAIL] Error: {error_msg}")
        
        # Summary
        print("\n" + "=" * 60)
        print("IMPORT SUMMARY")
        print("=" * 60)
        print(f"[SUCCESS] Imported: {imported}")
        print(f"[SKIP] Skipped: {skipped}")
        print(f"[FAIL] Errors: {len(errors)}")
        print("=" * 60)
        
        if errors:
            print("\nERROR DETAILS:")
            for error in errors:
                print(f"  - {error}")
        
        return {
            'success': True,
            'message': f"Imported {imported} students, skipped {skipped}",
            'imported': imported,
            'skipped': skipped,
            'errors': errors,
            'students': students
        }
    
    def export_template(self, output_path="student_template.xlsx"):
        """Tạo file Excel template mẫu
        
        Args:
            output_path: Đường dẫn file output
        """
        # Tạo DataFrame mẫu
        template_data = {
            'student_id': ['SV001', 'SV002', 'SV003'],
            'name': ['Nguyen Van A', 'Tran Thi B', 'Le Van C'],
            'class_name': ['CNTT01', 'CNTT01', 'CNTT02'],
            'email': ['nva@example.com', 'ttb@example.com', 'lvc@example.com'],
            'phone': ['0901234567', '0902345678', '0903456789']
        }
        
        df = pd.DataFrame(template_data)
        
        # Ghi file
        output_path = Path(output_path)
        df.to_excel(output_path, index=False)
        
        print(f"[SUCCESS] Template exported to: {output_path}")
        print("\nColumn descriptions:")
        print("  - student_id: Mã sinh viên (bắt buộc, duy nhất)")
        print("  - name: Họ tên (bắt buộc, dùng làm tên folder)")
        print("  - class_name: Lớp (optional)")
        print("  - email: Email (optional)")
        print("  - phone: Số điện thoại (optional)")
        
        return output_path
    
    def get_class_statistics(self):
        """Lấy thống kê theo lớp
        
        Returns:
            pd.DataFrame: Thống kê học sinh theo lớp
        """
        students = self.db.get_all_students()
        
        if not students:
            return pd.DataFrame()
        
        # Tạo DataFrame
        data = []
        for student in students:
            data.append({
                'student_id': student.student_id,
                'name': student.name,
                'class_name': student.class_name or 'Unassigned',
                'has_data': (self.enroll_dir / student.name).exists()
            })
        
        df = pd.DataFrame(data)
        
        # Thống kê theo lớp
        stats = df.groupby('class_name').agg({
            'student_id': 'count',
            'has_data': 'sum'
        }).rename(columns={
            'student_id': 'total_students',
            'has_data': 'with_enrollment_data'
        })
        
        stats['missing_data'] = stats['total_students'] - stats['with_enrollment_data']
        
        return stats


def main():
    """Main function với menu tương tác"""
    importer = StudentImporter()
    
    while True:
        print("\n" + "=" * 60)
        print(" STUDENT IMPORT SYSTEM")
        print("=" * 60)
        print("\nOptions:")
        print("1. Import students from Excel")
        print("2. Export Excel template")
        print("3. View class statistics")
        print("4. Create folders for existing students")
        print("0. Exit")
        print("=" * 60)
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice == '1':
            # Import students
            file_path = input("\nEnter Excel file path: ").strip().strip('"')
            
            if not file_path:
                print("[FAIL] File path is required!")
                continue
            
            class_name = input("Enter class name (optional, press Enter to skip): ").strip()
            if not class_name:
                class_name = None
            
            create_folders_input = input("Create folders for students? (y/n, default=y): ").strip().lower()
            create_folders = create_folders_input != 'n'
            
            result = importer.import_students(file_path, class_name, create_folders)
            
            if result['success']:
                print(f"\n[SUCCESS] {result['message']}")
            else:
                print(f"\n[FAIL] {result['message']}")
        
        elif choice == '2':
            # Export template
            output_path = input("\nEnter output path (default=student_template.xlsx): ").strip()
            if not output_path:
                output_path = "student_template.xlsx"
            
            importer.export_template(output_path)
        
        elif choice == '3':
            # View statistics
            print("\n" + "=" * 60)
            print("CLASS STATISTICS")
            print("=" * 60)
            
            stats = importer.get_class_statistics()
            
            if stats.empty:
                print("\nNo students in database yet.")
            else:
                print("\n" + stats.to_string())
                print("\nLegend:")
                print("  - total_students: Tổng số học sinh")
                print("  - with_enrollment_data: Đã có folder enrollment")
                print("  - missing_data: Chưa có dữ liệu enrollment")
        
        elif choice == '4':
            # Create folders for existing students
            print("\n" + "=" * 60)
            print("CREATE FOLDERS FOR EXISTING STUDENTS")
            print("=" * 60)
            
            students = importer.db.get_all_students()
            
            if not students:
                print("\nNo students in database.")
                continue
            
            created = 0
            existed = 0
            
            for student in students:
                folder_path = importer.enroll_dir / student.name
                
                if folder_path.exists():
                    existed += 1
                    continue
                
                # Tạo folder
                folder_path.mkdir(parents=True, exist_ok=True)
                
                # Tạo README
                readme_path = folder_path / "README.txt"
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(f"Student: {student.name}\n")
                    f.write(f"MSSV: {student.student_id}\n")
                    f.write(f"Class: {student.class_name or 'N/A'}\n")
                    f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                print(f"[SUCCESS] Created folder: {folder_path}")
                created += 1
            
            print("\n" + "=" * 60)
            print(f"[SUCCESS] Created: {created} folders")
            print(f"[SKIP] Already existed: {existed} folders")
            print("=" * 60)
        
        elif choice == '0':
            print("\nGoodbye!")
            break
        
        else:
            print("\n[FAIL] Invalid choice! Please try again.")


if __name__ == "__main__":
    main()

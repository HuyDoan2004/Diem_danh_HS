"""
Setup Script - First Time Setup
Thiết lập ban đầu cho hệ thống
"""

import subprocess
import sys
from pathlib import Path


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def check_python_version():
    """Kiểm tra Python version"""
    print_header("CHECKING PYTHON VERSION")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[FAIL] Python 3.8+ is required!")
        return False
    
    print("[SUCCESS] Python version OK")
    return True


def install_dependencies():
    """Cài đặt dependencies"""
    print_header("INSTALLING DEPENDENCIES")
    
    requirements = Path("requirements.txt.txt")
    
    if not requirements.exists():
        print("[FAIL] requirements.txt.txt not found!")
        return False
    
    print("Installing packages... (this may take a while)")
    
    try:
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            str(requirements)
        ])
        
        print("\n[SUCCESS] Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] Installation failed: {e}")
        return False


def create_directories():
    """Tạo các thư mục cần thiết"""
    print_header("CREATING DIRECTORIES")
    
    directories = [
        "data/enroll",
        "db",
        "db/models",
        "db/backups",
        "models"
    ]
    
    for dir_path in directories:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"[SUCCESS] Created: {dir_path}")
    
    print("\n[SUCCESS] All directories created")
    return True


def setup_database():
    """Setup database"""
    print_header("SETTING UP DATABASE")
    
    try:
        from src.database import DatabaseHandler
        
        db = DatabaseHandler()
        
        # Load schedules from config
        print("Loading schedules from config...")
        db.load_schedules_from_config()
        
        print("\n[SUCCESS] Database setup complete")
        return True
        
    except Exception as e:
        print(f"[FAIL] Database setup failed: {e}")
        return False


def print_next_steps():
    """Hiển thị các bước tiếp theo"""
    print_header("SETUP COMPLETE!")
    
    print(" Hệ thống đã sẵn sàng!\n")
    print("Các bước tiếp theo:\n")
    print("1⃣  Thu thập dữ liệu học sinh:")
    print("   python src/collect_data.py\n")
    
    print("2⃣  Training model:")
    print("   python src/train_model.py\n")
    
    print("3⃣  Đăng ký học sinh vào database:")
    print("   python src/database.py\n")
    
    print("4⃣  Chạy hệ thống điểm danh:")
    print("   python src/attendance_system.py\n")
    
    print("5⃣  Xem dashboard:")
    print("   streamlit run src/dashboard.py\n")
    
    print("Hoặc sử dụng menu:")
    print("   start.bat\n")
    
    print(" Đọc README.md để biết thêm chi tiết!")
    print("=" * 60)


def main():
    """Main setup process"""
    print("=" * 60)
    print("  HỆ THỐNG ĐIỂM DANH HỌC SINH")
    print("  First Time Setup")
    print("=" * 60)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing dependencies", install_dependencies),
        ("Creating directories", create_directories),
        ("Setting up database", setup_database)
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n[ERROR] Setup failed at: {step_name}")
            print("Please fix the errors and run setup again.")
            return False
    
    print_next_steps()
    return True


if __name__ == "__main__":
    success = main()
    
    if success:
        input("\nPress Enter to exit...")
    else:
        input("\nPress Enter to exit...")
        sys.exit(1)

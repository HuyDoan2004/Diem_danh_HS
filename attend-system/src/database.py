"""
Database Models and Handler
Quản lý database cho hệ thống điểm danh
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta
import yaml
from pathlib import Path

Base = declarative_base()


class Student(Base):
    """Model cho học sinh"""
    __tablename__ = 'students'
    
    id = Column(Integer, primary_key=True)
    student_id = Column(String(50), unique=True, nullable=False)  # MSSV
    name = Column(String(100), nullable=False)
    class_name = Column(String(50))
    email = Column(String(100))
    phone = Column(String(20))
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    attendances = relationship("Attendance", back_populates="student")
    
    def __repr__(self):
        return f"<Student(id={self.student_id}, name={self.name})>"


class Schedule(Base):
    """Model cho lịch học"""
    __tablename__ = 'schedules'
    
    id = Column(Integer, primary_key=True)
    subject = Column(String(100), nullable=False)
    day_of_week = Column(String(20), nullable=False)  # monday, tuesday, ...
    start_time = Column(String(10), nullable=False)  # HH:MM
    end_time = Column(String(10), nullable=False)    # HH:MM
    room = Column(String(50))
    teacher = Column(String(100))
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    sessions = relationship("AttendanceSession", back_populates="schedule")
    
    def __repr__(self):
        return f"<Schedule(subject={self.subject}, day={self.day_of_week}, time={self.start_time}-{self.end_time})>"


class AttendanceSession(Base):
    """Model cho buổi học (session)"""
    __tablename__ = 'attendance_sessions'
    
    id = Column(Integer, primary_key=True)
    schedule_id = Column(Integer, ForeignKey('schedules.id'), nullable=False)
    date = Column(DateTime, nullable=False)
    status = Column(String(20), default='scheduled')  # scheduled, ongoing, completed, cancelled
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    schedule = relationship("Schedule", back_populates="sessions")
    attendances = relationship("Attendance", back_populates="session")
    
    def __repr__(self):
        return f"<AttendanceSession(id={self.id}, date={self.date}, status={self.status})>"


class Attendance(Base):
    """Model cho điểm danh"""
    __tablename__ = 'attendances'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('attendance_sessions.id'), nullable=False)
    student_id = Column(Integer, ForeignKey('students.id'), nullable=False)
    
    check_in_time = Column(DateTime)
    check_out_time = Column(DateTime)
    status = Column(String(20), default='absent')  # present, late, absent
    confidence = Column(Float)  # Confidence score của face recognition
    
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    session = relationship("AttendanceSession", back_populates="attendances")
    student = relationship("Student", back_populates="attendances")
    
    def __repr__(self):
        return f"<Attendance(student={self.student_id}, session={self.session_id}, status={self.status})>"


class DatabaseHandler:
    """Handler để quản lý database operations"""
    
    def __init__(self, config_path="configs/database.yaml"):
        """Khởi tạo database connection"""
        self.config = self.load_config(config_path)
        
        db_config = self.config['database']
        if db_config['type'] == 'sqlite':
            db_path = Path(db_config['path'])
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(f"sqlite:///{db_path}")
        else:
            # PostgreSQL/MySQL connection string
            conn_str = f"{db_config['type']}://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            self.engine = create_engine(conn_str)
        
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        print(f"[SUCCESS] Database initialized: {db_config['type']}")
    
    def load_config(self, config_path):
        """Load database configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_session(self):
        """Tạo database session"""
        return self.Session()
    
    # Student operations
    def add_student(self, student_id, name, class_name=None, email=None, phone=None):
        """Thêm học sinh mới"""
        session = self.get_session()
        try:
            student = Student(
                student_id=student_id,
                name=name,
                class_name=class_name,
                email=email,
                phone=phone
            )
            session.add(student)
            session.commit()
            print(f"[SUCCESS] Added student: {name} ({student_id})")
            return student
        except Exception as e:
            session.rollback()
            print(f"[FAIL] Error adding student: {e}")
            return None
        finally:
            session.close()
    
    def get_student(self, student_id=None, name=None):
        """Lấy thông tin học sinh"""
        session = self.get_session()
        try:
            if student_id:
                return session.query(Student).filter_by(student_id=student_id).first()
            elif name:
                return session.query(Student).filter_by(name=name).first()
            return None
        finally:
            session.close()
    
    def get_all_students(self):
        """Lấy danh sách tất cả học sinh"""
        session = self.get_session()
        try:
            return session.query(Student).all()
        finally:
            session.close()
    
    # Schedule operations
    def add_schedule(self, subject, day_of_week, start_time, end_time, room=None, teacher=None):
        """Thêm lịch học"""
        session = self.get_session()
        try:
            schedule = Schedule(
                subject=subject,
                day_of_week=day_of_week.lower(),
                start_time=start_time,
                end_time=end_time,
                room=room,
                teacher=teacher
            )
            session.add(schedule)
            session.commit()
            print(f"[SUCCESS] Added schedule: {subject} on {day_of_week} at {start_time}")
            return schedule
        except Exception as e:
            session.rollback()
            print(f"[FAIL] Error adding schedule: {e}")
            return None
        finally:
            session.close()
    
    def get_schedule_for_day(self, day_of_week):
        """Lấy lịch học cho một ngày"""
        session = self.get_session()
        try:
            return session.query(Schedule).filter_by(day_of_week=day_of_week.lower()).all()
        finally:
            session.close()
    
    def get_current_schedule(self):
        """Lấy lịch học hiện tại (dựa vào thời gian)"""
        now = datetime.now()
        day_of_week = now.strftime('%A').lower()
        current_time = now.strftime('%H:%M')
        
        session = self.get_session()
        try:
            schedules = session.query(Schedule).filter_by(day_of_week=day_of_week).all()
            
            for schedule in schedules:
                if schedule.start_time <= current_time <= schedule.end_time:
                    return schedule
            
            return None
        finally:
            session.close()
    
    # Attendance Session operations
    def create_attendance_session(self, schedule_id, date=None):
        """Tạo buổi điểm danh mới"""
        session = self.get_session()
        try:
            if date is None:
                date = datetime.now()
            
            att_session = AttendanceSession(
                schedule_id=schedule_id,
                date=date,
                status='ongoing'
            )
            session.add(att_session)
            session.commit()
            print(f"[SUCCESS] Created attendance session for schedule_id={schedule_id}")
            return att_session
        except Exception as e:
            session.rollback()
            print(f"[FAIL] Error creating session: {e}")
            return None
        finally:
            session.close()
    
    def get_active_session(self):
        """Lấy buổi học đang diễn ra"""
        session = self.get_session()
        try:
            return session.query(AttendanceSession).filter_by(status='ongoing').first()
        finally:
            session.close()
    
    def get_session_by_date(self, date):
        """Lấy sessions theo ngày"""
        session = self.get_session()
        try:
            start_date = datetime.combine(date, datetime.min.time())
            end_date = datetime.combine(date, datetime.max.time())
            return session.query(AttendanceSession).filter(
                AttendanceSession.date >= start_date,
                AttendanceSession.date <= end_date
            ).all()
        finally:
            session.close()
    
    # Attendance operations
    def check_in(self, session_id, student_id, confidence, check_in_time=None):
        """Điểm danh check-in"""
        session = self.get_session()
        try:
            if check_in_time is None:
                check_in_time = datetime.now()
            
            # Check if already checked in
            existing = session.query(Attendance).filter_by(
                session_id=session_id,
                student_id=student_id
            ).first()
            
            if existing:
                print(f"[WARNING]  Student already checked in")
                return existing
            
            # Determine status (present or late)
            att_session = session.query(AttendanceSession).get(session_id)
            schedule = att_session.schedule
            
            # Parse schedule start time
            start_time_parts = schedule.start_time.split(':')
            schedule_start = datetime.combine(
                check_in_time.date(),
                datetime.min.time().replace(hour=int(start_time_parts[0]), minute=int(start_time_parts[1]))
            )
            
            # Load late threshold from config
            late_threshold = 10  # minutes
            late_time = schedule_start + timedelta(minutes=late_threshold)
            
            status = 'present' if check_in_time <= late_time else 'late'
            
            attendance = Attendance(
                session_id=session_id,
                student_id=student_id,
                check_in_time=check_in_time,
                status=status,
                confidence=confidence
            )
            
            session.add(attendance)
            session.commit()
            
            print(f"[SUCCESS] Check-in: Student {student_id} - Status: {status}")
            return attendance
            
        except Exception as e:
            session.rollback()
            print(f"[FAIL] Error during check-in: {e}")
            return None
        finally:
            session.close()
    
    def check_out(self, session_id, student_id, check_out_time=None):
        """Điểm danh check-out"""
        session = self.get_session()
        try:
            if check_out_time is None:
                check_out_time = datetime.now()
            
            attendance = session.query(Attendance).filter_by(
                session_id=session_id,
                student_id=student_id
            ).first()
            
            if not attendance:
                print(f"[WARNING]  No check-in record found")
                return None
            
            attendance.check_out_time = check_out_time
            attendance.updated_at = datetime.now()
            session.commit()
            
            print(f"[SUCCESS] Check-out: Student {student_id}")
            return attendance
            
        except Exception as e:
            session.rollback()
            print(f"[FAIL] Error during check-out: {e}")
            return None
        finally:
            session.close()
    
    def get_attendance_by_session(self, session_id):
        """Lấy danh sách điểm danh theo session"""
        session = self.get_session()
        try:
            return session.query(Attendance).filter_by(session_id=session_id).all()
        finally:
            session.close()
    
    def get_student_attendance_history(self, student_id, start_date=None, end_date=None):
        """Lấy lịch sử điểm danh của học sinh"""
        session = self.get_session()
        try:
            query = session.query(Attendance).filter_by(student_id=student_id)
            
            if start_date:
                query = query.join(AttendanceSession).filter(
                    AttendanceSession.date >= start_date
                )
            if end_date:
                query = query.join(AttendanceSession).filter(
                    AttendanceSession.date <= end_date
                )
            
            return query.all()
        finally:
            session.close()
    
    def get_attendance_statistics(self, student_id):
        """Thống kê điểm danh của học sinh"""
        session = self.get_session()
        try:
            attendances = session.query(Attendance).filter_by(student_id=student_id).all()
            
            total = len(attendances)
            present = len([a for a in attendances if a.status == 'present'])
            late = len([a for a in attendances if a.status == 'late'])
            absent = len([a for a in attendances if a.status == 'absent'])
            
            return {
                'total': total,
                'present': present,
                'late': late,
                'absent': absent,
                'attendance_rate': (present + late) / total * 100 if total > 0 else 0
            }
        finally:
            session.close()
    
    def load_schedules_from_config(self, config_path="configs/schedule.yaml"):
        """Load lịch học từ config file vào database"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        schedule_data = config['schedule']
        
        print("\n[SCHEDULE] Loading schedules from config...")
        for day, subjects in schedule_data.items():
            for subject_info in subjects:
                self.add_schedule(
                    subject=subject_info['subject'],
                    day_of_week=day,
                    start_time=subject_info['start_time'],
                    end_time=subject_info['end_time'],
                    room=subject_info.get('room'),
                    teacher=subject_info.get('teacher')
                )
        
        print("[SUCCESS] Schedules loaded successfully")


def main():
    """Test database operations"""
    print("=" * 60)
    print("[DATABASE]  DATABASE MANAGEMENT SYSTEM")
    print("=" * 60)
    
    db = DatabaseHandler()
    
    print("\n1. Add student")
    print("2. List all students")
    print("3. Load schedules from config")
    print("4. View today's schedule")
    print("5. View attendance statistics")
    print("0. Exit")
    
    choice = input("\nSelect option: ").strip()
    
    if choice == "1":
        student_id = input("Student ID: ").strip()
        name = input("Name: ").strip()
        class_name = input("Class (optional): ").strip() or None
        db.add_student(student_id, name, class_name)
    
    elif choice == "2":
        students = db.get_all_students()
        print(f"\n[LIST] Total students: {len(students)}")
        for s in students:
            print(f"   - {s.student_id}: {s.name} ({s.class_name})")
    
    elif choice == "3":
        db.load_schedules_from_config()
    
    elif choice == "4":
        day = datetime.now().strftime('%A').lower()
        schedules = db.get_schedule_for_day(day)
        print(f"\n[SCHEDULE] Schedule for {day}:")
        for s in schedules:
            print(f"   - {s.start_time}-{s.end_time}: {s.subject} ({s.room})")
    
    elif choice == "5":
        student_id = input("Student ID: ").strip()
        student = db.get_student(student_id=student_id)
        if student:
            stats = db.get_attendance_statistics(student.id)
            print(f"\n[STATS] Attendance Statistics for {student.name}:")
            print(f"   Total sessions: {stats['total']}")
            print(f"   Present: {stats['present']}")
            print(f"   Late: {stats['late']}")
            print(f"   Absent: {stats['absent']}")
            print(f"   Attendance rate: {stats['attendance_rate']:.1f}%")


if __name__ == "__main__":
    main()

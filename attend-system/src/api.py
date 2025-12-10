"""
FastAPI REST API for Attendance System
Cung cấp các endpoints để tích hợp với hệ thống khác
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Optional
import csv
import io

from database import DatabaseHandler

app = FastAPI(
    title="Attendance System API",
    description="REST API cho hệ thống điểm danh học sinh",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = DatabaseHandler()


# Pydantic models
class StudentCreate(BaseModel):
    student_id: str
    name: str
    class_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class ScheduleCreate(BaseModel):
    subject: str
    day_of_week: str
    start_time: str
    end_time: str
    room: Optional[str] = None
    teacher: Optional[str] = None


class AttendanceCheck(BaseModel):
    student_id: str
    session_id: int
    confidence: float


# Health check
@app.get("/")
def root():
    """API health check"""
    return {
        "status": "online",
        "service": "Attendance System API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


# Student endpoints
@app.get("/api/students")
def get_students():
    """Lấy danh sách tất cả học sinh"""
    students = db.get_all_students()
    return {
        "total": len(students),
        "students": [
            {
                "id": s.id,
                "student_id": s.student_id,
                "name": s.name,
                "class_name": s.class_name,
                "email": s.email,
                "phone": s.phone,
                "created_at": s.created_at.isoformat()
            }
            for s in students
        ]
    }


@app.get("/api/students/{student_id}")
def get_student(student_id: str):
    """Lấy thông tin chi tiết một học sinh"""
    student = db.get_student(student_id=student_id)
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    stats = db.get_attendance_statistics(student.id)
    
    return {
        "student": {
            "id": student.id,
            "student_id": student.student_id,
            "name": student.name,
            "class_name": student.class_name,
            "email": student.email,
            "phone": student.phone,
        },
        "statistics": stats
    }


@app.post("/api/students")
def create_student(student: StudentCreate):
    """Thêm học sinh mới"""
    result = db.add_student(
        student_id=student.student_id,
        name=student.name,
        class_name=student.class_name,
        email=student.email,
        phone=student.phone
    )
    
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create student (ID may already exist)")
    
    return {
        "success": True,
        "student_id": result.student_id,
        "message": f"Student {result.name} created successfully"
    }


# Schedule endpoints
@app.get("/api/schedule")
def get_schedule(day: Optional[str] = None):
    """Lấy lịch học (theo ngày nếu có)"""
    if day:
        schedules = db.get_schedule_for_day(day.lower())
    else:
        # Get current day's schedule
        current_day = datetime.now().strftime('%A').lower()
        schedules = db.get_schedule_for_day(current_day)
    
    return {
        "day": day or datetime.now().strftime('%A'),
        "total": len(schedules),
        "schedules": [
            {
                "id": s.id,
                "subject": s.subject,
                "day_of_week": s.day_of_week,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "room": s.room,
                "teacher": s.teacher
            }
            for s in schedules
        ]
    }


@app.get("/api/schedule/current")
def get_current_schedule():
    """Lấy lịch học hiện tại (đang diễn ra)"""
    schedule = db.get_current_schedule()
    
    if not schedule:
        return {
            "message": "No class scheduled at the moment",
            "schedule": None
        }
    
    return {
        "schedule": {
            "id": schedule.id,
            "subject": schedule.subject,
            "day_of_week": schedule.day_of_week,
            "start_time": schedule.start_time,
            "end_time": schedule.end_time,
            "room": schedule.room,
            "teacher": schedule.teacher
        }
    }


@app.post("/api/schedule")
def create_schedule(schedule: ScheduleCreate):
    """Thêm lịch học mới"""
    result = db.add_schedule(
        subject=schedule.subject,
        day_of_week=schedule.day_of_week,
        start_time=schedule.start_time,
        end_time=schedule.end_time,
        room=schedule.room,
        teacher=schedule.teacher
    )
    
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create schedule")
    
    return {
        "success": True,
        "schedule_id": result.id,
        "message": f"Schedule for {result.subject} created successfully"
    }


# Attendance session endpoints
@app.get("/api/sessions")
def get_sessions(date: Optional[str] = None):
    """Lấy danh sách sessions (theo ngày nếu có)"""
    if date:
        try:
            session_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        session_date = datetime.now().date()
    
    sessions = db.get_session_by_date(session_date)
    
    return {
        "date": session_date.isoformat(),
        "total": len(sessions),
        "sessions": [
            {
                "id": s.id,
                "schedule_id": s.schedule_id,
                "subject": s.schedule.subject,
                "date": s.date.isoformat(),
                "status": s.status,
                "room": s.schedule.room,
                "teacher": s.schedule.teacher
            }
            for s in sessions
        ]
    }


@app.get("/api/sessions/active")
def get_active_session():
    """Lấy session đang diễn ra"""
    session = db.get_active_session()
    
    if not session:
        return {
            "message": "No active session",
            "session": None
        }
    
    return {
        "session": {
            "id": session.id,
            "schedule_id": session.schedule_id,
            "subject": session.schedule.subject,
            "date": session.date.isoformat(),
            "status": session.status,
            "room": session.schedule.room,
            "teacher": session.schedule.teacher
        }
    }


# Attendance endpoints
@app.get("/api/attendance")
def get_attendance(
    session_id: Optional[int] = None,
    date: Optional[str] = None
):
    """Lấy dữ liệu điểm danh"""
    if session_id:
        attendances = db.get_attendance_by_session(session_id)
    elif date:
        try:
            att_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        sessions = db.get_session_by_date(att_date)
        attendances = []
        for session in sessions:
            attendances.extend(db.get_attendance_by_session(session.id))
    else:
        # Today's attendance
        today = datetime.now().date()
        sessions = db.get_session_by_date(today)
        attendances = []
        for session in sessions:
            attendances.extend(db.get_attendance_by_session(session.id))
    
    return {
        "total": len(attendances),
        "attendances": [
            {
                "id": a.id,
                "session_id": a.session_id,
                "student_id": a.student.student_id,
                "student_name": a.student.name,
                "check_in_time": a.check_in_time.isoformat() if a.check_in_time else None,
                "check_out_time": a.check_out_time.isoformat() if a.check_out_time else None,
                "status": a.status,
                "confidence": a.confidence
            }
            for a in attendances
        ]
    }


@app.post("/api/attendance/check-in")
def check_in(attendance: AttendanceCheck):
    """Điểm danh check-in"""
    # Get student
    student = db.get_student(student_id=attendance.student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Check-in
    result = db.check_in(
        session_id=attendance.session_id,
        student_id=student.id,
        confidence=attendance.confidence
    )
    
    if not result:
        raise HTTPException(status_code=400, detail="Check-in failed")
    
    return {
        "success": True,
        "student_name": student.name,
        "status": result.status,
        "check_in_time": result.check_in_time.isoformat(),
        "message": f"{student.name} checked in successfully"
    }


@app.get("/api/attendance/statistics")
def get_attendance_statistics(student_id: str):
    """Lấy thống kê điểm danh của học sinh"""
    student = db.get_student(student_id=student_id)
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    stats = db.get_attendance_statistics(student.id)
    
    return {
        "student_id": student.student_id,
        "student_name": student.name,
        "statistics": stats
    }


# Report endpoints
@app.get("/api/report/attendance.csv")
def export_attendance_csv(date: Optional[str] = None):
    """Xuất báo cáo điểm danh dạng CSV"""
    if date:
        try:
            report_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        report_date = datetime.now().date()
    
    sessions = db.get_session_by_date(report_date)
    
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'Date', 'Subject', 'Room', 'Teacher',
        'Student ID', 'Student Name', 'Class',
        'Check In', 'Check Out', 'Status', 'Confidence'
    ])
    
    # Data
    for session in sessions:
        attendances = db.get_attendance_by_session(session.id)
        for att in attendances:
            writer.writerow([
                session.date.strftime('%Y-%m-%d'),
                session.schedule.subject,
                session.schedule.room or '',
                session.schedule.teacher or '',
                att.student.student_id,
                att.student.name,
                att.student.class_name or '',
                att.check_in_time.strftime('%H:%M:%S') if att.check_in_time else '',
                att.check_out_time.strftime('%H:%M:%S') if att.check_out_time else '',
                att.status,
                f"{att.confidence:.2f}" if att.confidence else ''
            ])
    
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=attendance_{report_date}.csv"
        }
    )


@app.get("/api/report/summary")
def get_summary_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Báo cáo tổng hợp"""
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    students = db.get_all_students()
    
    summary = []
    for student in students:
        stats = db.get_attendance_statistics(student.id)
        summary.append({
            "student_id": student.student_id,
            "name": student.name,
            "class_name": student.class_name,
            "total_sessions": stats['total'],
            "present": stats['present'],
            "late": stats['late'],
            "absent": stats['absent'],
            "attendance_rate": stats['attendance_rate']
        })
    
    return {
        "period": {
            "start_date": start_date,
            "end_date": end_date
        },
        "total_students": len(students),
        "summary": summary
    }


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("[START] Starting Attendance System API")
    print("=" * 60)
    print("\nAPI Documentation: http://127.0.0.1:8000/docs")
    print("Alternative docs: http://127.0.0.1:8000/redoc")
    print("\n" + "=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

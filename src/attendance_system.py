"""
Real-time Attendance System with Schedule Management
Hệ thống điểm danh tự động với quản lý lịch học
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from ultralytics import YOLO
from PIL import Image
from datetime import datetime, timedelta
import yaml
import pickle
from pathlib import Path
import threading
import time
from collections import defaultdict

from database import DatabaseHandler


class AttendanceSystem:
    """Hệ thống điểm danh tự động với RealSense D435i"""
    
    def __init__(self, 
                 model_path="models/face_recognition_model.pkl",
                 camera_config="configs/camera.yaml",
                 schedule_config="configs/schedule.yaml",
                 use_webcam=False,
                 detection_method="mtcnn"):
        """Khởi tạo hệ thống
        
        Args:
            model_path: Đường dẫn model
            camera_config: Config camera
            schedule_config: Config lịch học
            use_webcam: True = webcam, False = RealSense D435i
        """
        
        print("=" * 60)
        print("[SYSTEM] ATTENDANCE SYSTEM INITIALIZATION")
        print("=" * 60)
        
        # Load configs
        self.camera_config = self.load_config(camera_config)
        self.schedule_config = self.load_config(schedule_config)
        
        # Camera mode
        self.use_webcam = use_webcam
        self.detection_method = detection_method  # "mtcnn" hoặc "yolo"
        
        # Initialize database
        self.db = DatabaseHandler()
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[SUCCESS] Using device: {self.device}")
        
        # Initialize face detection (MTCNN)
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=False
        )

        # Initialize YOLO detector (optional)
        self.yolo_model = None
        if self.detection_method == "yolo":
            weights_path = Path("yolo_training/face_detection/weights/best.pt")
            if weights_path.exists():
                try:
                    print(f"[SUCCESS] Loading YOLO model from: {weights_path}")
                    self.yolo_model = YOLO(str(weights_path))
                except Exception as e:
                    print(f"[WARNING]  Failed to load YOLO model: {e}")
                    print("          Falling back to MTCNN detector.")
                    self.detection_method = "mtcnn"
            else:
                print(f"[WARNING]  YOLO weights not found at {weights_path}")
                print("          Run 'Fine-tune YOLO' first or use MTCNN.")
                self.detection_method = "mtcnn"
        
        # Initialize face recognition model
        self.face_model = InceptionResnetV1(
            classify=False,
            pretrained='vggface2'
        ).to(self.device)
        self.face_model.eval()
        
        # Load trained embeddings
        self.load_embeddings(model_path)
        
        # Camera setup
        self.pipeline = None
        self.align = None
        self.cap = None
        
        # Attendance tracking
        self.current_session = None
        self.last_recognition = defaultdict(lambda: datetime.min)
        self.recognition_cooldown = timedelta(seconds=30)  # Cooldown giữa các lần nhận diện
        self.checked_in_students = set()  # Lưu danh sách student_id (DB) đã điểm danh trong buổi hiện tại
        self.student_status_in_session = {}  # student_id (DB) -> 'present' hoặc 'late'
        
        # Stats
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        print("[SUCCESS] System initialized successfully")
    
    def load_config(self, config_path):
        """Load configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_embeddings(self, model_path):
        """Load trained face embeddings"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"[WARNING]  Model not found: {model_path}")
            print("   Please train the model first!")
            self.embeddings_db = {}
            self.student_name_to_id = {}
            return
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings_db = data['embeddings_db']
        
        # Create mapping: name -> (database id, student_id/MSSV)
        self.student_name_to_id = {}
        self.student_name_to_mssv = {}
        for student_name in self.embeddings_db.keys():
            student = self.db.get_student(name=student_name)
            if student:
                self.student_name_to_id[student_name] = student.id
                self.student_name_to_mssv[student_name] = student.student_id  # MSSV
            else:
                print(f"[WARNING]  Student '{student_name}' not found in database")
        
        print(f"[SUCCESS] Loaded embeddings for {len(self.embeddings_db)} students")
    
    def initialize_camera(self):
        """Khởi tạo camera"""
        try:
            if self.use_webcam:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Cannot open webcam")
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                print("[SUCCESS] Webcam initialized")
                return True
            else:
                self.pipeline = rs.pipeline()
                config = rs.config()
                
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                
                profile = self.pipeline.start(config)
                
                align_to = rs.stream.color
                self.align = rs.align(align_to)
                
                print("[SUCCESS] RealSense D435i camera initialized")
                return True
        except Exception as e:
            print(f"[FAIL] Camera initialization failed: {e}")
            return False
    
    def get_current_schedule(self):
        """Lấy lịch học hiện tại"""
        now = datetime.now()
        day_of_week = now.strftime('%A').lower()
        current_time = now.strftime('%H:%M')
        
        # Get schedules for today
        schedules = self.db.get_schedule_for_day(day_of_week)
        
        # Find current schedule
        for schedule in schedules:
            # Add check-in window before class starts
            check_in_window = self.schedule_config['attendance']['check_in_window']
            
            start_parts = schedule.start_time.split(':')
            start_time = datetime.combine(now.date(), datetime.min.time().replace(
                hour=int(start_parts[0]), minute=int(start_parts[1])
            ))
            
            # Allow check-in 15 minutes before class
            check_in_start = start_time - timedelta(minutes=check_in_window)
            
            end_parts = schedule.end_time.split(':')
            end_time = datetime.combine(now.date(), datetime.min.time().replace(
                hour=int(end_parts[0]), minute=int(end_parts[1])
            ))
            
            if check_in_start <= now <= end_time:
                return schedule
        
        return None

    def get_late_threshold_for_schedule(self, schedule):
        """Lấy ngưỡng đi trễ (phút) cho một lịch học.

        Ưu tiên cấu hình theo từng môn trong configs/schedule.yaml:
        schedule:
          monday:
            - subject: ABC
              start_time: "10:00"
              end_time: "11:30"
              late_threshold: 10  # cho phép muộn 10p

        Nếu không khai báo trong từng môn thì dùng attendance.late_threshold.
        Nếu giá trị là None (hoặc không có), coi như không giới hạn, ai điểm danh
        trong suốt thời gian học đều được tính là present.
        """
        # Mặc định từ phần attendance
        attendance_cfg = self.schedule_config.get('attendance', {}) if self.schedule_config else {}
        default_threshold = attendance_cfg.get('late_threshold', None)

        try:
            schedule_cfg = self.schedule_config.get('schedule', {})
        except Exception:
            return default_threshold

        day = (schedule.day_of_week or '').lower()
        day_entries = schedule_cfg.get(day, []) or []

        for entry in day_entries:
            if (
                entry.get('subject') == schedule.subject and
                entry.get('start_time') == schedule.start_time and
                entry.get('end_time') == schedule.end_time
            ):
                # Cho phép late_threshold = None tại từng môn
                return entry.get('late_threshold', default_threshold)

        return default_threshold

    def get_attendance_status(self, check_in_time, schedule):
        """Xác định trạng thái điểm danh 'present' hoặc 'late' cho 1 lần check-in.

        - Nếu late_threshold là None: luôn trả về 'present' (điểm danh bất kỳ lúc nào).
        - Nếu có late_threshold (phút):
            + check_in_time <= start_time + late_threshold  => 'present'
            + ngược lại                                      => 'late'
        """
        late_threshold = self.get_late_threshold_for_schedule(schedule)

        if late_threshold is None:
            return 'present'

        # Tính mốc thời gian cho phép đi trễ
        start_parts = schedule.start_time.split(':')
        schedule_start = datetime.combine(
            check_in_time.date(),
            datetime.min.time().replace(hour=int(start_parts[0]), minute=int(start_parts[1]))
        )

        late_deadline = schedule_start + timedelta(minutes=int(late_threshold))
        return 'present' if check_in_time <= late_deadline else 'late'
    
    def start_session(self, schedule):
        """Bắt đầu buổi học mới"""
        # Check if session already exists
        today = datetime.now().date()
        existing_sessions = self.db.get_session_by_date(today)
        
        for session in existing_sessions:
            if session.schedule_id == schedule.id and session.status == 'ongoing':
                self.current_session = session
                print(f"[SUCCESS] Resumed existing session: {schedule.subject}")
                # Khởi tạo lại thông tin đã điểm danh cho buổi hiện tại
                self.checked_in_students.clear()
                self.student_status_in_session.clear()
                existing_att = self.db.get_attendance_by_session(session.id)
                for att in existing_att:
                    self.checked_in_students.add(att.student_id)
                    self.student_status_in_session[att.student_id] = att.status
                return session
        
        # Create new session
        self.current_session = self.db.create_attendance_session(schedule.id)
        # Reset trạng thái điểm danh cho buổi mới
        self.checked_in_students.clear()
        self.student_status_in_session.clear()
        print(f"[SUCCESS] Started new session: {schedule.subject}")
        return self.current_session
    
    def recognize_face(self, face_tensor):
        """
        Nhận diện khuôn mặt
        
        Args:
            face_tensor: Tensor của khuôn mặt (160x160)
        
        Returns:
            (student_name, confidence, distance)
        """
        if len(self.embeddings_db) == 0:
            return None, 0.0, float('inf')
        
        with torch.no_grad():
            # Get embedding
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            embedding = self.face_model(face_tensor).cpu().numpy()[0]
            
            # Find best match
            min_distance = float('inf')
            best_match = None
            
            for student_name, data in self.embeddings_db.items():
                mean_emb = data['mean_embedding']
                distance = np.linalg.norm(embedding - mean_emb)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = student_name
            
            # Calculate confidence from distance (FaceNet embeddings are L2-normalized,
            # so max distance ~= 2.0; scale accordingly)
            confidence = max(0.0, 1.0 - (min_distance / 2.0))
            
            # Threshold
            threshold = self.schedule_config['attendance']['confidence_threshold']
            
            if confidence >= threshold:
                return best_match, confidence, min_distance
            else:
                return None, confidence, min_distance
    
    def process_frame(self, color_image):
        """
        Xử lý frame và nhận diện khuôn mặt
        
        Returns:
            (processed_image, detections)
        """
        detections = []
        
        # Detect faces bằng YOLO hoặc MTCNN tùy cấu hình
        if self.detection_method == "yolo" and self.yolo_model is not None:
            try:
                results = self.yolo_model(color_image, verbose=False)[0]
                if results.boxes is not None:
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0].item())
                        if conf < 0.5:
                            continue

                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(color_image.shape[1], x2), min(color_image.shape[0], y2)

                        face = color_image[y1:y2, x1:x2]

                        try:
                            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                            face_pil = Image.fromarray(face_rgb)
                            face_tensor = self.mtcnn(face_pil)

                            if face_tensor is not None:
                                student_name, confidence, distance = self.recognize_face(face_tensor)

                                detections.append({
                                    'box': (x1, y1, x2, y2),
                                    'landmarks': None,
                                    'student_name': student_name,
                                    'student_id': self.student_name_to_mssv.get(student_name, ''),
                                    'confidence': confidence,
                                    'distance': distance
                                })

                                if student_name and self.current_session:
                                    now = datetime.now()
                                    last_time = self.last_recognition[student_name]
                                    if now - last_time > self.recognition_cooldown:
                                        student_db_id = self.student_name_to_id.get(student_name)
                                        if student_db_id:
                                            # Xác định trạng thái điểm danh dựa trên ngưỡng đi trễ
                                            status = self.get_attendance_status(now, self.current_session.schedule)
                                            attendance = self.db.check_in(
                                                self.current_session.id,
                                                student_db_id,
                                                confidence,
                                                check_in_time=now,
                                                status=status
                                            )
                                            if attendance:
                                                self.checked_in_students.add(student_db_id)
                                                self.student_status_in_session[student_db_id] = attendance.status
                                            self.last_recognition[student_name] = now
                        except Exception as e:
                            print(f"Error processing YOLO face: {e}")
                            continue
            except Exception as e:
                print(f"[WARNING]  YOLO detection failed, falling back to MTCNN: {e}")
                # Nếu YOLO lỗi thì fallback sang MTCNN cho frame hiện tại
                self.detection_method = "mtcnn"

        if self.detection_method == "mtcnn":
            # Detect faces bằng MTCNN trực tiếp trên frame
            boxes, probs, landmarks = self.mtcnn.detect(color_image, landmarks=True)
            
            if boxes is not None:
                for box, prob, landmark in zip(boxes, probs, landmarks):
                    if prob < 0.9:  # Face detection confidence threshold
                        continue
                    
                    # Extract face region
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(color_image.shape[1], x2), min(color_image.shape[0], y2)
                    
                    face = color_image[y1:y2, x1:x2]
                    
                    try:
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_pil = Image.fromarray(face_rgb)
                        face_tensor = self.mtcnn(face_pil)
                        
                        if face_tensor is not None:
                            student_name, confidence, distance = self.recognize_face(face_tensor)
                            
                            detections.append({
                                'box': (x1, y1, x2, y2),
                                'landmarks': landmark,
                                'student_name': student_name,
                                'student_id': self.student_name_to_mssv.get(student_name, ''),
                                'confidence': confidence,
                                'distance': distance
                            })
                            
                            if student_name and self.current_session:
                                now = datetime.now()
                                last_time = self.last_recognition[student_name]
                                
                                if now - last_time > self.recognition_cooldown:
                                    student_db_id = self.student_name_to_id.get(student_name)
                                    if student_db_id:
                                        status = self.get_attendance_status(now, self.current_session.schedule)
                                        attendance = self.db.check_in(
                                            self.current_session.id,
                                            student_db_id,
                                            confidence,
                                            check_in_time=now,
                                            status=status
                                        )
                                        if attendance:
                                            self.checked_in_students.add(student_db_id)
                                            self.student_status_in_session[student_db_id] = attendance.status
                                        self.last_recognition[student_name] = now
                    
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue
        
        return detections
    
    def draw_detections(self, image, detections, schedule_info=None):
        """Vẽ kết quả nhận diện lên image"""
        display_image = image.copy()
        
        # Draw schedule info
        if schedule_info:
            info_text = f"{schedule_info['subject']} | {schedule_info['time']} | {schedule_info['room']}"
            cv2.rectangle(display_image, (0, 0), (display_image.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(display_image, info_text, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['box']
            student_name = det['student_name']
            student_id = det.get('student_id', '')
            confidence = det['confidence']
            student_db_id = self.student_name_to_id.get(student_name) if student_name else None
            already_checked = student_db_id in self.checked_in_students if student_db_id is not None else False
            status = self.student_status_in_session.get(student_db_id) if student_db_id is not None else None
            
            # Choose color & label based on recognition + trạng thái đúng giờ / đi trễ
            if student_name:
                # Đã điểm danh rồi
                if already_checked:
                    if status == 'present':
                        # Đúng giờ: chữ xanh lá + dấu ✓
                        color = (0, 255, 0)
                        label = f"{student_name} - {student_id} (✓)"
                    elif status == 'late':
                        # Đi trễ quá ngưỡng: chữ đỏ + dấu X
                        color = (0, 0, 255)
                        label = f"{student_name} - {student_id} (X)"
                    else:
                        # Phòng trường hợp trạng thái khác
                        color = (0, 255, 0)
                        label = f"{student_name} - {student_id} ({confidence:.2f})"
                else:
                    # Chưa check-in: chỉ mới nhận diện, hiển thị màu xanh + confidence
                    color = (0, 255, 0)
                    label = f"{student_name} - {student_id} ({confidence:.2f})"
            else:
                color = (0, 0, 255)  # Red for unknown
                label = "Unknown"
            
            # Draw bounding box
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(display_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw landmarks
            if det['landmarks'] is not None:
                for point in det['landmarks']:
                    cv2.circle(display_image, tuple(point.astype(int)), 2, color, -1)
        
        # Draw FPS
        cv2.putText(display_image, f"FPS: {self.fps:.1f}", (10, display_image.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return display_image
    
    def run(self):
        """Chạy hệ thống điểm danh"""
        if not self.initialize_camera():
            return
        
        print("\n" + "=" * 60)
        print("[CAMERA] ATTENDANCE SYSTEM RUNNING")
        print("=" * 60)
        print("\nControls:")
        print("  SPACE - Refresh schedule")
        print("  S     - View session statistics")
        print("  ESC   - Exit")
        print("\n" + "=" * 60 + "\n")
        
        try:
            while True:
                # Get frames
                if self.use_webcam:
                    ret, color_image = self.cap.read()
                    if not ret:
                        continue
                    depth_image = None
                else:
                    frames = self.pipeline.wait_for_frames()
                    aligned_frames = self.align.process(frames)
                    
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    
                    if not depth_frame or not color_frame:
                        continue
                    
                    # Convert to numpy
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                
                # Check schedule
                current_schedule = self.get_current_schedule()
                
                if current_schedule and not self.current_session:
                    self.start_session(current_schedule)
                elif not current_schedule and self.current_session:
                    # End session
                    print(f"[SUCCESS] Session ended: {self.current_session.schedule.subject}")
                    self.current_session = None
                
                # Process frame
                detections = []
                if self.current_session:
                    detections = self.process_frame(color_image)
                
                # Prepare schedule info for display
                schedule_info = None
                if current_schedule:
                    schedule_info = {
                        'subject': current_schedule.subject,
                        'time': f"{current_schedule.start_time}-{current_schedule.end_time}",
                        'room': current_schedule.room or 'N/A'
                    }
                
                # Draw results
                display_image = self.draw_detections(color_image, detections, schedule_info)
                
                # Calculate FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    current_time = time.time()
                    self.fps = 30 / (current_time - self.last_fps_time)
                    self.last_fps_time = current_time
                
                # Show
                cv2.imshow('Attendance System', display_image)
                
                # Depth visualization (only for RealSense)
                if not self.use_webcam and depth_image is not None:
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03),
                        cv2.COLORMAP_JET
                    )
                    cv2.imshow('Depth', depth_colormap)
                
                # Key handling
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # SPACE - refresh schedule
                    print("\nrotate Refreshing schedule...")
                    current_schedule = self.get_current_schedule()
                    if current_schedule:
                        print(f"   Current: {current_schedule.subject}")
                    else:
                        print("   No class scheduled")
                elif key == ord('s') or key == ord('S'):  # Statistics
                    self.show_statistics()
        
        except KeyboardInterrupt:
            print("\n[WARNING]  Interrupted by user")
        
        finally:
            self.cleanup()
    
    def show_statistics(self):
        """Hiển thị thống kê buổi học hiện tại"""
        if not self.current_session:
            print("\n[WARNING]  No active session")
            return
        
        attendances = self.db.get_attendance_by_session(self.current_session.id)
        
        print("\n" + "=" * 60)
        print(f"[STATS] SESSION STATISTICS")
        print("=" * 60)
        print(f"Subject: {self.current_session.schedule.subject}")
        print(f"Date: {self.current_session.date}")
        print(f"Total check-ins: {len(attendances)}")
        print(f"\nAttendance List:")
        
        for att in attendances:
            status_icon = "[SUCCESS]" if att.status == "present" else "" if att.status == "late" else "[FAIL]"
            print(f"  {status_icon} {att.student.name} - {att.status.upper()} - {att.check_in_time.strftime('%H:%M:%S')}")
        
        print("=" * 60 + "\n")
    
    def cleanup(self):
        """Dọn dẹp resources"""
        if self.use_webcam and self.cap:
            self.cap.release()
        elif self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()
        print("\n[SUCCESS] System shutdown complete")


def main():
    """Main entry point"""
    print("=" * 60)
    print("[SYSTEM] ATTENDANCE SYSTEM")
    print("=" * 60)
    print("\nSelect camera type:")
    print("1. RealSense D435i (depth camera)")
    print("2. Webcam (regular camera)")

    camera_choice = input("\nCamera type (1 or 2, default=2): ").strip() or "2"
    use_webcam = (camera_choice == "2")

    print("\nSelect detection method:")
    print("1. MTCNN (current system)")
    print("2. YOLO (fine-tuned face detector)")

    det_choice = input("\nDetection method (1 or 2, default=1): ").strip() or "1"
    detection_method = "yolo" if det_choice == "2" else "mtcnn"

    print(f"\n[CONFIG] Using camera: {'Webcam' if use_webcam else 'RealSense D435i'}")
    print(f"[CONFIG] Detection method: {detection_method.upper()}")

    system = AttendanceSystem(use_webcam=use_webcam, detection_method=detection_method)
    system.run()


if __name__ == "__main__":
    main()

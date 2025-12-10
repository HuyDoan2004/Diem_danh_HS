"""
Real-time Attendance System with Schedule Management
Hệ thống điểm danh tự động với quản lý lịch học
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
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
                 use_webcam=False):
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
            keep_all=True
        )
        
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
    
    def start_session(self, schedule):
        """Bắt đầu buổi học mới"""
        # Check if session already exists
        today = datetime.now().date()
        existing_sessions = self.db.get_session_by_date(today)
        
        for session in existing_sessions:
            if session.schedule_id == schedule.id and session.status == 'ongoing':
                self.current_session = session
                print(f"[SUCCESS] Resumed existing session: {schedule.subject}")
                return session
        
        # Create new session
        self.current_session = self.db.create_attendance_session(schedule.id)
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
            
            # Calculate confidence (inverse of distance)
            confidence = max(0, 1 - min_distance)
            
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
        
        # Detect faces
        boxes, probs, landmarks = self.mtcnn.detect(color_image, landmarks=True)
        
        if boxes is not None:
            for box, prob, landmark in zip(boxes, probs, landmarks):
                if prob < 0.9:  # Face detection confidence threshold
                    continue
                
                # Extract face region
                x1, y1, x2, y2 = [int(coord) for coord in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(color_image.shape[1], x2), min(color_image.shape[0], y2)
                
                # Crop and process face
                face = color_image[y1:y2, x1:x2]
                
                try:
                    # Convert to tensor
                    face_pil = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_tensor = self.mtcnn.forward(face_pil, return_prob=False)
                    
                    if face_tensor is not None:
                        # Recognize
                        student_name, confidence, distance = self.recognize_face(face_tensor)
                        
                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'landmarks': landmark,
                            'student_name': student_name,
                            'student_id': self.student_name_to_mssv.get(student_name, ''),
                            'confidence': confidence,
                            'distance': distance
                        })
                        
                        # Auto check-in if recognized and session active
                        if student_name and self.current_session:
                            now = datetime.now()
                            last_time = self.last_recognition[student_name]
                            
                            # Check cooldown
                            if now - last_time > self.recognition_cooldown:
                                student_db_id = self.student_name_to_id.get(student_name)
                                if student_db_id:
                                    self.db.check_in(
                                        self.current_session.id,
                                        student_db_id,
                                        confidence
                                    )
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
            
            # Choose color based on recognition
            if student_name:
                color = (0, 255, 0)  # Green for recognized
                # Display: Name - MSSV (confidence)
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
    
    system = AttendanceSystem(use_webcam=use_webcam)
    system.run()


if __name__ == "__main__":
    main()

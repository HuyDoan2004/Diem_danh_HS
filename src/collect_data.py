"""
Video Data Collection Module
Thu thập video khuôn mặt từ RealSense D435i camera để training model
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
from pathlib import Path
import yaml


class VideoCollector:
    """Thu thập video khuôn mặt cho việc training"""
    
    def __init__(self, config_path="configs/camera.yaml", use_webcam=False):
        """Khởi tạo video collector
        
        Args:
            config_path: Đường dẫn config file
            use_webcam: True = dùng webcam thường, False = dùng RealSense D435i
        """
        self.config = self.load_config(config_path)
        self.use_webcam = use_webcam
        self.pipeline = None
        self.align = None
        self.cap = None
        
    def load_config(self, config_path):
        """Load camera configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def initialize_camera(self):
        """Khởi tạo camera (RealSense hoặc webcam)"""
        try:
            if self.use_webcam:
                # Use regular webcam
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Cannot open webcam")
                
                # Set resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                print("[SUCCESS] Webcam initialized successfully")
                return True
            else:
                # Use RealSense D435i
                self.pipeline = rs.pipeline()
                config = rs.config()
                
                # Enable streams
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                
                # Start streaming
                profile = self.pipeline.start(config)
                
                # Create an align object
                align_to = rs.stream.color
                self.align = rs.align(align_to)
                
                print("[SUCCESS] RealSense D435i camera initialized successfully")
                return True
            
        except Exception as e:
            print(f"[FAIL] Error initializing camera: {e}")
            return False
    
    def collect_student_video(self, student_name, duration=30, output_dir="data/enroll"):
        """
        Thu thập video khuôn mặt của học sinh
        
        Args:
            student_name: Tên học sinh
            duration: Thời gian thu thập (giây)
            output_dir: Thư mục lưu video
        """
        if not self.pipeline:
            if not self.initialize_camera():
                return False
        
        # Tạo thư mục cho học sinh
        student_dir = Path(output_dir) / student_name
        student_dir.mkdir(parents=True, exist_ok=True)
        
        # Tên file video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = student_dir / f"video_{timestamp}.avi"
        depth_video_path = student_dir / f"depth_{timestamp}.avi"
        
        # Video writer cho RGB và Depth
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        rgb_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
        depth_writer = cv2.VideoWriter(str(depth_video_path), fourcc, 30.0, (640, 480))
        
        print(f"\n[CAMERA] Bắt đầu thu thập video cho: {student_name}")
        print(f"⏱  Thời gian: {duration} giây")
        print(" Hướng dẫn:")
        print("   - Di chuyển đầu từ trái sang phải")
        print("   - Nhìn lên, nhìn xuống")
        print("   - Xoay đầu nhẹ")
        print("   - Thay đổi biểu cảm")
        print("\nNhấn SPACE để bắt đầu, ESC để thoát...\n")
        
        start_recording = False
        start_time = None
        frame_count = 0
        
        try:
            while True:
                # Get frames based on camera type
                if self.use_webcam:
                    # Webcam mode
                    ret, color_image = self.cap.read()
                    if not ret:
                        continue
                    
                    # Create fake depth (gray version)
                    depth_colormap = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                    depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_GRAY2BGR)
                else:
                    # RealSense mode
                    frames = self.pipeline.wait_for_frames()
                    
                    # Align depth frame to color frame
                    aligned_frames = self.align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    
                    if not depth_frame or not color_frame:
                        continue
                    
                    # Convert images to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Apply colormap on depth image
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03), 
                        cv2.COLORMAP_JET
                    )
                
                # Detect faces for guidance
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                # Draw rectangles around faces
                display_image = color_image.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Recording logic
                if start_recording:
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    remaining_time = duration - elapsed_time
                    
                    if remaining_time > 0:
                        # Save frames
                        rgb_writer.write(color_image)
                        depth_writer.write(depth_colormap)
                        frame_count += 1
                        
                        # Display recording info
                        cv2.putText(display_image, f"RECORDING: {remaining_time:.1f}s", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(display_image, f"Frames: {frame_count}", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        print(f"\n[SUCCESS] Hoàn thành! Đã lưu {frame_count} frames")
                        print(f"[FOLDER] RGB Video: {video_path}")
                        print(f"[FOLDER] Depth Video: {depth_video_path}")
                        break
                else:
                    cv2.putText(display_image, "Press SPACE to start recording", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(display_image, f"Student: {student_name}", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Stack images horizontally
                images = np.hstack((display_image, depth_colormap))
                
                # Show images
                cv2.imshow('Video Collection (RGB | Depth)', images)
                
                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and not start_recording:
                    start_recording = True
                    start_time = datetime.now()
                    print("[RED] Recording started...")
                elif key == 27:  # ESC
                    print("\n[WARNING]  Cancelled by user")
                    break
                    
        except Exception as e:
            print(f"[FAIL] Error during recording: {e}")
            return False
            
        finally:
            rgb_writer.release()
            depth_writer.release()
            cv2.destroyAllWindows()
        
        return True
    
    def extract_frames_from_video(self, video_path, output_dir, max_frames=100):
        """
        Trích xuất frames từ video đã thu thập
        
        Args:
            video_path: Đường dẫn video
            output_dir: Thư mục lưu frames
            max_frames: Số frames tối đa cần trích xuất
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Tính interval để lấy đều các frames
        interval = max(1, total_frames // max_frames)
        
        print(f"\n[STATS] Extracting frames from: {video_path}")
        print(f"   Total frames: {total_frames}, Extracting: {min(max_frames, total_frames)}")
        
        frame_idx = 0
        saved_count = 0
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        while cap.isOpened() and saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Chỉ lưu frames theo interval
            if frame_idx % interval == 0:
                # Detect face để chỉ lưu frames có khuôn mặt
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    frame_path = output_dir / f"frame_{saved_count:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    saved_count += 1
            
            frame_idx += 1
        
        cap.release()
        print(f"[SUCCESS] Extracted {saved_count} frames with faces")
        return saved_count
    
    def batch_collect(self, student_list, duration=30):
        """
        Thu thập video cho nhiều học sinh
        
        Args:
            student_list: Danh sách tên học sinh
            duration: Thời gian thu thập mỗi học sinh (giây)
        """
        print(f"\n[LIST] Batch Collection for {len(student_list)} students")
        print("=" * 50)
        
        for idx, student_name in enumerate(student_list, 1):
            print(f"\n[{idx}/{len(student_list)}] Student: {student_name}")
            success = self.collect_student_video(student_name, duration)
            
            if not success:
                print(f"[WARNING]  Skipped {student_name}")
                continue
            
            # Trích xuất frames ngay sau khi thu thập
            video_dir = Path("data/enroll") / student_name
            videos = list(video_dir.glob("video_*.avi"))
            if videos:
                latest_video = max(videos, key=lambda x: x.stat().st_mtime)
                frames_dir = video_dir / "frames"
                self.extract_frames_from_video(latest_video, frames_dir)
        
        print("\n" + "=" * 50)
        print("[SUCCESS] Batch collection completed!")
    
    def cleanup(self):
        """Dọn dẹp resources"""
        if self.use_webcam and self.cap:
            self.cap.release()
            print("[SUCCESS] Webcam stopped")
        elif self.pipeline:
            self.pipeline.stop()
            print("[SUCCESS] RealSense camera stopped")


def main():
    """Main function để test collection"""
    print("=" * 60)
    print(" VIDEO DATA COLLECTION SYSTEM")
    print("=" * 60)
    
    print("\nSelect camera type:")
    print("1. RealSense D435i (depth camera)")
    print("2. Webcam (regular camera)")
    camera_choice = input("\nCamera type (1 or 2): ").strip()
    
    use_webcam = (camera_choice == "2")
    collector = VideoCollector(use_webcam=use_webcam)
    
    if use_webcam:
        print("\n[SUCCESS] Using webcam mode")
    else:
        print("\n[SUCCESS] Using RealSense D435i mode")
    
    print("\n1. Single student collection")
    print("2. Batch collection")
    print("3. Extract frames from existing video")
    print("0. Exit")
    
    choice = input("\nSelect option: ").strip()
    
    if choice == "1":
        student_name = input("Enter student name: ").strip()
        if student_name:
            duration = int(input("Duration (seconds, default 30): ") or "30")
            collector.collect_student_video(student_name, duration)
    
    elif choice == "2":
        students_input = input("Enter student names (comma separated): ").strip()
        if students_input:
            student_list = [s.strip() for s in students_input.split(",")]
            duration = int(input("Duration per student (seconds, default 30): ") or "30")
            collector.batch_collect(student_list, duration)
    
    elif choice == "3":
        video_path = input("Enter video path: ").strip()
        output_dir = input("Enter output directory: ").strip()
        if video_path and output_dir:
            max_frames = int(input("Max frames to extract (default 100): ") or "100")
            collector.extract_frames_from_video(video_path, output_dir, max_frames)
    
    collector.cleanup()


if __name__ == "__main__":
    main()

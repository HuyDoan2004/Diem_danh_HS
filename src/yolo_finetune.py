"""
YOLO Fine-tuning Module
Fine-tune YOLOv8 cho face detection trong môi trường lớp học
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import shutil


class YOLOFineTuner:
    """Fine-tune YOLO model cho face detection"""
    
    def __init__(self, base_model="yolov8n.pt", project_dir="yolo_training"):
        """
        Args:
            base_model: Pre-trained YOLO model (n, s, m, l, x)
            project_dir: Thư mục lưu kết quả training
        """
        self.base_model = base_model
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_dir = self.project_dir / "dataset"
        self.model = None
        
        print(f"[SUCCESS] Initialized YOLO Fine-tuner")
        print(f"  Base model: {base_model}")
        print(f"  Project dir: {project_dir}")
    
    def prepare_dataset_from_enrollment(self, enroll_dir="data/enroll", train_split=0.8):
        """
        Chuẩn bị dataset từ dữ liệu enrollment
        
        Args:
            enroll_dir: Thư mục chứa dữ liệu enrollment
            train_split: Tỷ lệ train/val split
        """
        print("\n" + "="*60)
        print(" PREPARING YOLO DATASET")
        print("="*60)
        
        enroll_path = Path(enroll_dir)
        
        # Create dataset structure
        train_images = self.dataset_dir / "images" / "train"
        train_labels = self.dataset_dir / "labels" / "train"
        val_images = self.dataset_dir / "images" / "val"
        val_labels = self.dataset_dir / "labels" / "val"
        
        for path in [train_images, train_labels, val_images, val_labels]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Face detector for auto-labeling
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        total_images = 0
        total_faces = 0
        
        # Process each student
        for student_dir in enroll_path.iterdir():
            if not student_dir.is_dir():
                continue
            
            frames_dir = student_dir / "frames"
            if not frames_dir.exists():
                continue
            
            print(f"\nProcessing: {student_dir.name}")
            
            # Get all frames
            frames = list(frames_dir.glob("*.jpg"))
            
            # Split train/val
            n_train = int(len(frames) * train_split)
            train_frames = frames[:n_train]
            val_frames = frames[n_train:]
            
            # Process train frames
            for frame_path in tqdm(train_frames, desc="  Train"):
                self._process_frame(frame_path, train_images, train_labels, face_cascade)
                total_images += 1
            
            # Process val frames
            for frame_path in tqdm(val_frames, desc="  Val"):
                self._process_frame(frame_path, val_images, val_labels, face_cascade)
                total_images += 1
        
        print(f"\n[SUCCESS] Dataset prepared:")
        print(f"  Total images: {total_images}")
        print(f"  Train: {len(list(train_images.glob('*.jpg')))}")
        print(f"  Val: {len(list(val_images.glob('*.jpg')))}")
        
        # Create dataset.yaml
        self._create_dataset_yaml()
        
        return self.dataset_dir / "dataset.yaml"
    
    def _process_frame(self, frame_path, images_dir, labels_dir, face_cascade):
        """Process single frame and create YOLO label"""
        # Read image
        image = cv2.imread(str(frame_path))
        if image is None:
            return
        
        h, w = image.shape[:2]
        
        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return
        
        # Copy image
        new_image_path = images_dir / frame_path.name
        shutil.copy(frame_path, new_image_path)
        
        # Create YOLO label
        label_path = labels_dir / f"{frame_path.stem}.txt"
        
        with open(label_path, 'w') as f:
            for (x, y, fw, fh) in faces:
                # Convert to YOLO format (normalized center x, y, width, height)
                center_x = (x + fw / 2) / w
                center_y = (y + fh / 2) / h
                norm_w = fw / w
                norm_h = fh / h
                
                # Class 0 for face
                f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    
    def _create_dataset_yaml(self):
        """Create dataset.yaml for YOLO"""
        yaml_path = self.dataset_dir / "dataset.yaml"
        
        config = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['face']
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"[SUCCESS] Created: {yaml_path}")
    
    def train(self, epochs=50, imgsz=640, batch=16, patience=10):
        """
        Fine-tune YOLO model
        
        Args:
            epochs: Số epochs training
            imgsz: Kích thước ảnh input
            batch: Batch size
            patience: Early stopping patience
        """
        print("\n" + "="*60)
        print("[START] TRAINING YOLO MODEL")
        print("="*60)
        
        # Check dataset
        dataset_yaml = self.dataset_dir / "dataset.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError("Dataset not prepared! Run prepare_dataset_from_enrollment() first")
        
        # Load base model
        print(f"\n Loading base model: {self.base_model}")
        self.model = YOLO(self.base_model)
        
        # Training configuration
        print(f"\n Training configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {imgsz}")
        print(f"  Batch size: {batch}")
        print(f"  Patience: {patience}")
        
        # Train
        print(f"\n Starting training...\n")
        
        results = self.model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            project=str(self.project_dir),
            name='face_detection',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            plots=True,
            save=True,
            save_period=10,
            cache=False,
            device='',  # Auto-detect GPU/CPU
            workers=8,
            cos_lr=True,
            close_mosaic=10,
            amp=True,
            fraction=1.0,
            profile=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            split='val',
            verbose=True
        )
        
        print(f"\n[OK] Training complete!")
        print(f"[FOLDER] Results saved to: {self.project_dir / 'face_detection'}")
        
        return results
    
    def validate(self):
        """Validate trained model"""
        print("\n" + "="*60)
        print("[STATS] VALIDATING MODEL")
        print("="*60)
        
        if self.model is None:
            # Load best model
            best_model = self.project_dir / "face_detection" / "weights" / "best.pt"
            if not best_model.exists():
                raise FileNotFoundError("No trained model found!")
            
            print(f" Loading model: {best_model}")
            self.model = YOLO(str(best_model))
        
        # Validate
        results = self.model.val()
        
        # Print metrics
        print("\n Validation Metrics:")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")
        print(f"  Precision: {results.box.mp:.4f}")
        print(f"  Recall: {results.box.mr:.4f}")
        
        return results
    
    def plot_training_results(self):
        """Vẽ biểu đồ kết quả training"""
        print("\n" + "="*60)
        print("[STATS] PLOTTING TRAINING RESULTS")
        print("="*60)
        
        results_dir = self.project_dir / "face_detection"
        
        # Check if results exist
        results_csv = results_dir / "results.csv"
        if not results_csv.exists():
            print("[WARNING] No results.csv found. Train the model first.")
            return
        
        # Read results
        import pandas as pd
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('YOLO Training Results', fontsize=16, fontweight='bold')
        
        # 1. Loss curves
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', linewidth=2)
        axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Box Loss')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Class loss
        if 'train/cls_loss' in df.columns:
            axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', linewidth=2)
            axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Class Loss')
            axes[0, 1].set_title('Classification Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. DFL loss
        if 'train/dfl_loss' in df.columns:
            axes[0, 2].plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss', linewidth=2)
            axes[0, 2].plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', linewidth=2)
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('DFL Loss')
            axes[0, 2].set_title('Distribution Focal Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Precision & Recall
        if 'metrics/precision(B)' in df.columns:
            axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color='blue')
            axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Precision & Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. mAP
        if 'metrics/mAP50(B)' in df.columns:
            axes[1, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2, color='orange')
            axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2, color='red')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('mAP')
            axes[1, 1].set_title('Mean Average Precision')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Learning rate
        if 'lr/pg0' in df.columns:
            axes[1, 2].plot(df['epoch'], df['lr/pg0'], linewidth=2, color='purple')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_title('Learning Rate Schedule')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = results_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Saved: {save_path}")
        plt.close()
    
    def test_on_images(self, test_dir="data/enroll", conf=0.25, save_dir=None):
        """
        Test model trên các ảnh
        
        Args:
            test_dir: Thư mục chứa ảnh test
            conf: Confidence threshold
            save_dir: Thư mục lưu kết quả
        """
        print("\n" + "="*60)
        print(" TESTING MODEL")
        print("="*60)
        
        if self.model is None:
            best_model = self.project_dir / "face_detection" / "weights" / "best.pt"
            if not best_model.exists():
                raise FileNotFoundError("No trained model found!")
            
            self.model = YOLO(str(best_model))
        
        if save_dir is None:
            save_dir = self.project_dir / "test_results"
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get test images
        test_path = Path(test_dir)
        image_files = []
        
        for student_dir in test_path.iterdir():
            if not student_dir.is_dir():
                continue
            frames_dir = student_dir / "frames"
            if frames_dir.exists():
                image_files.extend(list(frames_dir.glob("*.jpg"))[:5])  # Take 5 images per student
        
        if not image_files:
            print("[WARNING] No test images found!")
            return
        
        print(f"Testing on {len(image_files)} images...")
        
        # Run inference
        for img_path in tqdm(image_files):
            results = self.model.predict(
                source=str(img_path),
                conf=conf,
                save=False,
                verbose=False
            )
            
            # Draw results
            img = cv2.imread(str(img_path))
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Draw box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"Face {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save
            save_path = save_dir / img_path.name
            cv2.imwrite(str(save_path), img)
        
        print(f"[SUCCESS] Results saved to: {save_dir}")
    
    def export_model(self, format='onnx'):
        """
        Export model sang định dạng khác
        
        Args:
            format: 'onnx', 'torchscript', 'openvino', 'engine', etc.
        """
        print(f"\n Exporting model to {format.upper()}...")
        
        if self.model is None:
            best_model = self.project_dir / "face_detection" / "weights" / "best.pt"
            self.model = YOLO(str(best_model))
        
        export_path = self.model.export(format=format)
        print(f"[SUCCESS] Exported to: {export_path}")
        
        return export_path
    
    def generate_report(self):
        """Tạo báo cáo tổng hợp"""
        print("\n" + "="*60)
        print("[FILE] GENERATING REPORT")
        print("="*60)
        
        results_dir = self.project_dir / "face_detection"
        
        # Validate model
        val_results = self.validate()
        
        report = f"""
# YOLO Face Detection Fine-tuning Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- Base Model: {self.base_model}
- Task: Face Detection
- Dataset: Custom enrollment data

## Training Results

### Validation Metrics
- **mAP@0.5**: {val_results.box.map50:.4f}
- **mAP@0.5:0.95**: {val_results.box.map:.4f}
- **Precision**: {val_results.box.mp:.4f}
- **Recall**: {val_results.box.mr:.4f}

## Model Files
- Best weights: `{results_dir}/weights/best.pt`
- Last weights: `{results_dir}/weights/last.pt`
- Training curves: `{results_dir}/training_curves.png`

## Usage

### Python
```python
from ultralytics import YOLO

model = YOLO('{results_dir}/weights/best.pt')
results = model.predict('image.jpg', conf=0.25)
```

### Command Line
```bash
yolo detect predict model={results_dir}/weights/best.pt source=image.jpg
```

## Recommendations
- mAP@0.5 > 0.8: Excellent [OK]
- mAP@0.5 > 0.6: Good [WARNING]
- mAP@0.5 < 0.6: Needs improvement [ERROR]

Current Status: {'[OK] Excellent' if val_results.box.map50 > 0.8 else '[WARNING] Good' if val_results.box.map50 > 0.6 else '[ERROR] Needs improvement'}
"""
        
        # Save report
        report_path = results_dir / 'training_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[SUCCESS] Report saved: {report_path}")


def main():
    """Main function"""
    print("="*60)
    print("  YOLO FINE-TUNING FOR FACE DETECTION")
    print("="*60)
    
    print("\n1. Prepare dataset and train")
    print("2. Validate existing model")
    print("3. Test on images")
    print("4. Export model")
    print("5. Generate report")
    print("0. Exit")
    
    choice = input("\nSelect option: ").strip()
    
    tuner = YOLOFineTuner(base_model="yolov8n.pt")
    
    if choice == "1":
        # Full pipeline
        print("\n[CONFIG] Starting full training pipeline...")
        
        # Prepare dataset
        dataset_yaml = tuner.prepare_dataset_from_enrollment()
        
        # Train
        epochs = int(input("\nEpochs (default 50): ") or "50")
        tuner.train(epochs=epochs, imgsz=640, batch=16)
        
        # Plot results
        tuner.plot_training_results()
        
        # Validate
        tuner.validate()
        
        # Generate report
        tuner.generate_report()
        
        print("\n[OK] Training pipeline complete!")
    
    elif choice == "2":
        tuner.validate()
    
    elif choice == "3":
        conf = float(input("Confidence threshold (default 0.25): ") or "0.25")
        tuner.test_on_images(conf=conf)
    
    elif choice == "4":
        print("\nAvailable formats:")
        print("  1. ONNX (recommended)")
        print("  2. TorchScript")
        print("  3. OpenVINO")
        
        fmt_choice = input("Select format (1/2/3): ").strip()
        formats = {"1": "onnx", "2": "torchscript", "3": "openvino"}
        fmt = formats.get(fmt_choice, "onnx")
        
        tuner.export_model(format=fmt)
    
    elif choice == "5":
        tuner.plot_training_results()
        tuner.generate_report()


if __name__ == "__main__":
    main()

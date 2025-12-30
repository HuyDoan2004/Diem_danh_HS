"""
Face Recognition Training Module
Training model nhận diện khuôn mặt từ dữ liệu đã thu thập
Sử dụng FaceNet (InceptionResnetV1) để tạo embeddings
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path
import pickle
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


class FaceDataset(Dataset):
    """Dataset cho face recognition training"""
    
    def __init__(self, data_dir, transform=None, mtcnn=None):
        """
        Args:
            data_dir: Thư mục chứa dữ liệu (data/enroll)
            transform: Transformations cho images
            mtcnn: MTCNN detector để crop faces
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.mtcnn = mtcnn
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load tất cả images và labels"""
        student_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        # Tạo mapping: student name -> index
        for idx, student_dir in enumerate(sorted(student_dirs)):
            student_name = student_dir.name
            self.class_to_idx[student_name] = idx
            self.idx_to_class[idx] = student_name
            
            # Load frames từ thư mục frames
            frames_dir = student_dir / "frames"
            if frames_dir.exists():
                for img_path in frames_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), idx))
        
        print(f"[SUCCESS] Loaded {len(self.samples)} samples from {len(self.class_to_idx)} students")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Detect and crop face using MTCNN
        if self.mtcnn is not None:
            try:
                img = self.mtcnn(img)
                if img is None:
                    # Fallback: resize original image
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((160, 160))
                    img = transforms.ToTensor()(img)
            except:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((160, 160))
                img = transforms.ToTensor()(img)
        else:
            if self.transform:
                img = self.transform(img)
        
        return img, label


class FaceRecognitionTrainer:
    """Training pipeline cho face recognition"""
    
    def __init__(self, data_dir="data/enroll", model_dir="models"):
        """
        Args:
            data_dir: Thư mục chứa dữ liệu training
            model_dir: Thư mục lưu models
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[SUCCESS] Using device: {self.device}")
        
        # Initialize MTCNN for face detection and cropping
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )
        
        # Initialize FaceNet model (InceptionResnetV1)
        self.model = InceptionResnetV1(
            classify=False,
            pretrained='vggface2'
        ).to(self.device)
        
        self.model.eval()  # Set to evaluation mode for embedding extraction
        
        self.embeddings_db = {}
        self.class_to_idx = {}
        self.idx_to_class = {}
    
    def extract_embeddings(self, batch_size=32, num_workers=0):
        """
        Trích xuất embeddings cho tất cả students
        
        Returns:
            dict: {student_name: [embeddings_list]}
        """
        print("\n" + "=" * 60)
        print("[STATS] EXTRACTING FACE EMBEDDINGS")
        print("=" * 60)
        
        # Create dataset
        dataset = FaceDataset(
            self.data_dir,
            mtcnn=self.mtcnn
        )
        
        if len(dataset) == 0:
            print("[FAIL] No data found! Please collect data first.")
            return None
        
        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = dataset.idx_to_class
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Extract embeddings
        embeddings_list = []
        labels_list = []
        
        print(f"\nrotate Processing {len(dataset)} images...")
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting embeddings"):
                images = images.to(self.device)
                
                # Get embeddings
                embeddings = self.model(images)
                
                embeddings_list.append(embeddings.cpu().numpy())
                labels_list.append(labels.numpy())
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings_list)
        all_labels = np.concatenate(labels_list)
        
        # Organize embeddings by student
        embeddings_by_student = {}
        for embedding, label in zip(all_embeddings, all_labels):
            student_name = self.idx_to_class[label]
            if student_name not in embeddings_by_student:
                embeddings_by_student[student_name] = []
            embeddings_by_student[student_name].append(embedding)
        
        # Calculate mean embedding for each student
        for student_name in embeddings_by_student:
            embeddings_array = np.array(embeddings_by_student[student_name])
            self.embeddings_db[student_name] = {
                'mean_embedding': np.mean(embeddings_array, axis=0),
                'embeddings': embeddings_array,
                'count': len(embeddings_array)
            }
        
        print(f"\n[SUCCESS] Extracted embeddings for {len(self.embeddings_db)} students")
        for student_name, data in self.embeddings_db.items():
            print(f"   - {student_name}: {data['count']} embeddings")
        
        return self.embeddings_db
    
    def save_model(self, filename="face_recognition_model.pkl"):
        """Lưu embeddings database và metadata"""
        model_path = self.model_dir / filename
        
        save_data = {
            'embeddings_db': self.embeddings_db,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'timestamp': datetime.now().isoformat(),
            'num_students': len(self.embeddings_db),
            'device': str(self.device)
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Save metadata as JSON
        metadata = {
            'num_students': len(self.embeddings_db),
            'students': list(self.embeddings_db.keys()),
            'embeddings_per_student': {
                name: data['count'] 
                for name, data in self.embeddings_db.items()
            },
            'timestamp': datetime.now().isoformat(),
            'model_file': filename
        }
        
        metadata_path = self.model_dir / "model_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUCCESS] Model saved to: {model_path}")
        print(f"[SUCCESS] Metadata saved to: {metadata_path}")
    
    def load_model(self, filename="face_recognition_model.pkl"):
        """Load trained model"""
        model_path = self.model_dir / filename
        
        if not model_path.exists():
            print(f"[FAIL] Model not found: {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.embeddings_db = save_data['embeddings_db']
        self.class_to_idx = save_data['class_to_idx']
        self.idx_to_class = save_data['idx_to_class']
        
        print(f"[SUCCESS] Model loaded from: {model_path}")
        print(f"   Students: {len(self.embeddings_db)}")
        return True
    
    def evaluate_model(self, test_ratio=0.2):
        """
        Đánh giá độ chính xác của model
        
        Args:
            test_ratio: Tỷ lệ dữ liệu để test
        """
        print("\n" + "=" * 60)
        print(" MODEL EVALUATION")
        print("=" * 60)
        
        # Prepare test data
        all_embeddings = []
        all_labels = []
        
        for student_name, data in self.embeddings_db.items():
            embeddings = data['embeddings']
            label = self.class_to_idx[student_name]
            
            for emb in embeddings:
                all_embeddings.append(emb)
                all_labels.append(label)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            all_embeddings, all_labels, 
            test_size=test_ratio, 
            random_state=42,
            stratify=all_labels
        )
        
        # Predict on test set
        predictions = []
        for test_emb in X_test:
            # Find closest match
            min_dist = float('inf')
            predicted_label = -1
            
            for student_name, data in self.embeddings_db.items():
                mean_emb = data['mean_embedding']
                dist = np.linalg.norm(test_emb - mean_emb)
                
                if dist < min_dist:
                    min_dist = dist
                    predicted_label = self.class_to_idx[student_name]
            
            predictions.append(predicted_label)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"\n[SUCCESS] Test Accuracy: {accuracy * 100:.2f}%")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.idx_to_class.values(),
            yticklabels=self.idx_to_class.values()
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plot_path = self.model_dir / 'confusion_matrix.png'
        plt.savefig(plot_path)
        print(f"[SUCCESS] Confusion matrix saved to: {plot_path}")
        plt.close()
        
        # Per-class accuracy
        print("\n[STATS] Per-Student Accuracy:")
        for i, student_name in self.idx_to_class.items():
            mask = np.array(y_test) == i
            if mask.sum() > 0:
                class_acc = (np.array(predictions)[mask] == i).sum() / mask.sum()
                print(f"   - {student_name}: {class_acc * 100:.2f}%")
        
        return accuracy
    
    def visualize_embeddings(self):
        """Visualize embeddings using t-SNE or PCA"""
        from sklearn.manifold import TSNE
        
        print("\n[STATS] Visualizing embeddings...")
        
        all_embeddings = []
        all_labels = []
        all_names = []
        
        for student_name, data in self.embeddings_db.items():
            for emb in data['embeddings']:
                all_embeddings.append(emb)
                all_labels.append(self.class_to_idx[student_name])
                all_names.append(student_name)

        # Guard: no embeddings
        if len(all_embeddings) == 0:
            print("[WARNING]  No embeddings available for visualization")
            return

        # Convert to numpy array for TSNE
        all_embeddings = np.vstack(all_embeddings)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1],
            c=all_labels,
            cmap='tab10',
            alpha=0.6
        )
        plt.colorbar(scatter)
        plt.title('Face Embeddings Visualization (t-SNE)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        # Add legend
        handles, labels = scatter.legend_elements()
        plt.legend(handles, self.idx_to_class.values(), loc='best')
        
        plt.tight_layout()
        plot_path = self.model_dir / 'embeddings_visualization.png'
        plt.savefig(plot_path)
        print(f"[SUCCESS] Visualization saved to: {plot_path}")
        plt.close()


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("[SYSTEM] FACE RECOGNITION TRAINING SYSTEM")
    print("=" * 60)
    
    trainer = FaceRecognitionTrainer()
    
    print("\n1. Train new model")
    print("2. Evaluate existing model")
    print("3. Visualize embeddings")
    print("0. Exit")
    
    choice = input("\nSelect option: ").strip()
    
    if choice == "1":
        # Extract embeddings
        embeddings_db = trainer.extract_embeddings(batch_size=16)
        
        if embeddings_db is not None:
            # Save model
            trainer.save_model()
            
            # Evaluate
            trainer.evaluate_model(test_ratio=0.2)
            
            # Visualize
            trainer.visualize_embeddings()
            
            print("\n" + "=" * 60)
            print("[SUCCESS] TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
    
    elif choice == "2":
        if trainer.load_model():
            trainer.evaluate_model(test_ratio=0.2)
    
    elif choice == "3":
        if trainer.load_model():
            trainer.visualize_embeddings()


if __name__ == "__main__":
    main()

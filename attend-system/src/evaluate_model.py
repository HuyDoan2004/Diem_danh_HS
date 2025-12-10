"""
Model Evaluation Module
Đánh giá chất lượng mô hình nhận diện khuôn mặt với các metrics chi tiết
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Đánh giá toàn diện model nhận diện khuôn mặt"""
    
    def __init__(self, model_path="models/face_recognition_model.pkl", output_dir="models/evaluation"):
        """
        Args:
            model_path: Đường dẫn đến model đã train
            output_dir: Thư mục lưu kết quả đánh giá
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_db = None
        self.class_to_idx = None
        self.idx_to_class = None
        
        # Load model
        self.load_model()
        
        # Set style for plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def load_model(self):
        """Load trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings_db = data['embeddings_db']
        self.class_to_idx = data['class_to_idx']
        self.idx_to_class = data['idx_to_class']
        
        print(f"[SUCCESS] Loaded model with {len(self.embeddings_db)} classes")
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Chuẩn bị dữ liệu train/test
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n[STATS] Preparing data...")
        
        all_embeddings = []
        all_labels = []
        all_names = []
        
        for student_name, data in self.embeddings_db.items():
            embeddings = data['embeddings']
            label = self.class_to_idx[student_name]
            
            for emb in embeddings:
                all_embeddings.append(emb)
                all_labels.append(label)
                all_names.append(student_name)
        
        X = np.array(all_embeddings)
        y = np.array(all_labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"[SUCCESS] Train samples: {len(X_train)}")
        print(f"[SUCCESS] Test samples: {len(X_test)}")
        print(f"[SUCCESS] Classes: {len(self.idx_to_class)}")
        
        return X_train, X_test, y_train, y_test
    
    def predict_with_distance(self, X_test):
        """
        Dự đoán sử dụng distance threshold (giống real system)
        
        Returns:
            predictions, distances, confidences
        """
        predictions = []
        distances = []
        confidences = []
        
        for test_emb in tqdm(X_test, desc="Predicting"):
            min_dist = float('inf')
            predicted_label = -1
            
            for student_name, data in self.embeddings_db.items():
                mean_emb = data['mean_embedding']
                dist = np.linalg.norm(test_emb - mean_emb)
                
                if dist < min_dist:
                    min_dist = dist
                    predicted_label = self.class_to_idx[student_name]
            
            predictions.append(predicted_label)
            distances.append(min_dist)
            confidences.append(max(0, 1 - min_dist))
        
        return np.array(predictions), np.array(distances), np.array(confidences)
    
    def evaluate_basic_metrics(self, y_true, y_pred):
        """Tính các metrics cơ bản"""
        print("\n" + "="*60)
        print(" BASIC METRICS")
        print("="*60)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Print metrics
        print(f"\nAccuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision (Macro):  {metrics['precision_macro']:.4f}")
        print(f"Precision (Weight): {metrics['precision_weighted']:.4f}")
        print(f"Recall (Macro):     {metrics['recall_macro']:.4f}")
        print(f"Recall (Weight):    {metrics['recall_weighted']:.4f}")
        print(f"F1-Score (Macro):   {metrics['f1_macro']:.4f}")
        print(f"F1-Score (Weight):  {metrics['f1_weighted']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False):
        """Vẽ confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=[self.idx_to_class[i] for i in range(len(self.idx_to_class))],
            yticklabels=[self.idx_to_class[i] for i in range(len(self.idx_to_class))],
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save
        filename = 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png'
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Saved: {save_path}")
        plt.close()
    
    def plot_per_class_metrics(self, y_true, y_pred):
        """Vẽ metrics cho từng class"""
        # Calculate per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        precisions = [report[str(i)]['precision'] for i in range(len(classes))]
        recalls = [report[str(i)]['recall'] for i in range(len(classes))]
        f1_scores = [report[str(i)]['f1-score'] for i in range(len(classes))]
        supports = [report[str(i)]['support'] for i in range(len(classes))]
        
        # Create dataframe
        df = pd.DataFrame({
            'Class': classes,
            'Precision': precisions,
            'Recall': recalls,
            'F1-Score': f1_scores,
            'Support': supports
        })
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Per-Class Performance Metrics', fontsize=18, fontweight='bold', y=0.995)
        
        # Precision
        axes[0, 0].barh(df['Class'], df['Precision'], color='#3498db')
        axes[0, 0].set_xlabel('Precision', fontweight='bold')
        axes[0, 0].set_title('Precision by Class', fontweight='bold')
        axes[0, 0].set_xlim([0, 1.1])
        for i, v in enumerate(df['Precision']):
            axes[0, 0].text(v + 0.02, i, f'{v:.3f}', va='center')
        
        # Recall
        axes[0, 1].barh(df['Class'], df['Recall'], color='#2ecc71')
        axes[0, 1].set_xlabel('Recall', fontweight='bold')
        axes[0, 1].set_title('Recall by Class', fontweight='bold')
        axes[0, 1].set_xlim([0, 1.1])
        for i, v in enumerate(df['Recall']):
            axes[0, 1].text(v + 0.02, i, f'{v:.3f}', va='center')
        
        # F1-Score
        axes[1, 0].barh(df['Class'], df['F1-Score'], color='#e74c3c')
        axes[1, 0].set_xlabel('F1-Score', fontweight='bold')
        axes[1, 0].set_title('F1-Score by Class', fontweight='bold')
        axes[1, 0].set_xlim([0, 1.1])
        for i, v in enumerate(df['F1-Score']):
            axes[1, 0].text(v + 0.02, i, f'{v:.3f}', va='center')
        
        # Support
        axes[1, 1].barh(df['Class'], df['Support'], color='#9b59b6')
        axes[1, 1].set_xlabel('Number of Samples', fontweight='bold')
        axes[1, 1].set_title('Support by Class', fontweight='bold')
        for i, v in enumerate(df['Support']):
            axes[1, 1].text(v + 0.5, i, f'{int(v)}', va='center')
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'per_class_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Saved: {save_path}")
        plt.close()
        
        # Save as CSV
        csv_path = self.output_dir / 'per_class_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"[SUCCESS] Saved: {csv_path}")
        
        return df
    
    def plot_distance_distribution(self, distances, y_true, y_pred):
        """Phân bố khoảng cách của predictions"""
        correct_mask = y_true == y_pred
        correct_distances = distances[correct_mask]
        incorrect_distances = distances[~correct_mask]
        
        plt.figure(figsize=(14, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(correct_distances, bins=50, alpha=0.7, label='Correct', color='green', edgecolor='black')
        plt.hist(incorrect_distances, bins=50, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
        plt.xlabel('Distance', fontweight='bold')
        plt.ylabel('Frequency', fontweight='bold')
        plt.title('Distance Distribution (Correct vs Incorrect)', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        data = [correct_distances, incorrect_distances]
        bp = plt.boxplot(data, labels=['Correct', 'Incorrect'], patch_artist=True)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][1].set_facecolor('red')
        plt.ylabel('Distance', fontweight='bold')
        plt.title('Distance Box Plot', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        plt.text(1, correct_distances.max(), 
                f'μ={correct_distances.mean():.3f}\nσ={correct_distances.std():.3f}',
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        plt.text(2, incorrect_distances.max() if len(incorrect_distances) > 0 else 0,
                f'μ={incorrect_distances.mean():.3f}\nσ={incorrect_distances.std():.3f}' if len(incorrect_distances) > 0 else 'N/A',
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'distance_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Saved: {save_path}")
        plt.close()
    
    def plot_confidence_analysis(self, confidences, y_true, y_pred):
        """Phân tích confidence scores"""
        correct_mask = y_true == y_pred
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Confidence Score Analysis', fontsize=18, fontweight='bold')
        
        # 1. Histogram
        axes[0, 0].hist(correct_conf, bins=50, alpha=0.7, label='Correct', color='green', edgecolor='black')
        axes[0, 0].hist(incorrect_conf, bins=50, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
        axes[0, 0].set_xlabel('Confidence', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title('Confidence Distribution', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Threshold analysis
        thresholds = np.linspace(0, 1, 100)
        accuracies = []
        
        for thresh in thresholds:
            valid_mask = confidences >= thresh
            if valid_mask.sum() > 0:
                acc = (y_true[valid_mask] == y_pred[valid_mask]).mean()
            else:
                acc = 0
            accuracies.append(acc)
        
        axes[0, 1].plot(thresholds, accuracies, linewidth=2, color='#3498db')
        axes[0, 1].axvline(x=0.7, color='red', linestyle='--', label='Current threshold (0.7)')
        axes[0, 1].set_xlabel('Confidence Threshold', fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
        axes[0, 1].set_title('Accuracy vs Confidence Threshold', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cumulative distribution
        axes[1, 0].hist(confidences, bins=50, cumulative=True, density=True, 
                       alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_xlabel('Confidence', fontweight='bold')
        axes[1, 0].set_ylabel('Cumulative Probability', fontweight='bold')
        axes[1, 0].set_title('Cumulative Confidence Distribution', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Scatter plot
        sample_indices = np.arange(len(confidences))
        colors = ['green' if c else 'red' for c in correct_mask]
        axes[1, 1].scatter(sample_indices, confidences, c=colors, alpha=0.5, s=10)
        axes[1, 1].axhline(y=0.7, color='blue', linestyle='--', label='Threshold (0.7)')
        axes[1, 1].set_xlabel('Sample Index', fontweight='bold')
        axes[1, 1].set_ylabel('Confidence', fontweight='bold')
        axes[1, 1].set_title('Confidence Scatter (Green=Correct, Red=Incorrect)', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'confidence_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Saved: {save_path}")
        plt.close()
    
    def cross_validation_analysis(self, X, y, cv=5):
        """Cross-validation với nhiều classifiers"""
        print("\n" + "="*60)
        print("rotate CROSS-VALIDATION ANALYSIS")
        print("="*60)
        
        classifiers = {
            'KNN (k=1)': KNeighborsClassifier(n_neighbors=1),
            'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
            'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
            'SVM (Linear)': SVC(kernel='linear'),
            'SVM (RBF)': SVC(kernel='rbf')
        }
        
        results = {}
        
        for name, clf in classifiers.items():
            print(f"\n{name}:")
            kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')
            
            results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            
            print(f"  Scores: {scores}")
            print(f"  Mean: {scores.mean():.4f} ± {scores.std():.4f}")
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        names = list(results.keys())
        means = [results[n]['mean'] for n in names]
        stds = [results[n]['std'] for n in names]
        
        plt.bar(names, means, yerr=stds, capsize=10, color='skyblue', edgecolor='black', alpha=0.7)
        plt.ylabel('Accuracy', fontweight='bold')
        plt.title(f'{cv}-Fold Cross-Validation Results', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, 1.1])
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'cross_validation.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[SUCCESS] Saved: {save_path}")
        plt.close()
        
        return results
    
    def generate_summary_report(self, metrics, df_per_class):
        """Tạo báo cáo tổng hợp"""
        print("\n" + "="*60)
        print("[FILE] GENERATING SUMMARY REPORT")
        print("="*60)
        
        report = f"""
# MODEL EVALUATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- Model Path: {self.model_path}
- Number of Classes: {len(self.idx_to_class)}
- Classes: {', '.join(self.idx_to_class.values())}

## Overall Performance

### Accuracy
- **{metrics['accuracy']:.4f}** ({metrics['accuracy']*100:.2f}%)

### Precision
- Macro: {metrics['precision_macro']:.4f}
- Weighted: {metrics['precision_weighted']:.4f}

### Recall
- Macro: {metrics['recall_macro']:.4f}
- Weighted: {metrics['recall_weighted']:.4f}

### F1-Score
- Macro: {metrics['f1_macro']:.4f}
- Weighted: {metrics['f1_weighted']:.4f}

## Per-Class Performance

{df_per_class.to_markdown(index=False)}

## Recommendations

1. **Model Quality**: {'[SUCCESS] Excellent' if metrics['accuracy'] > 0.95 else ' Needs improvement' if metrics['accuracy'] > 0.85 else '[FAIL] Poor'}
2. **Deployment Ready**: {'Yes' if metrics['accuracy'] > 0.90 else 'No - Collect more data'}
3. **Suggested Actions**:
"""
        
        # Add recommendations based on results
        if metrics['accuracy'] < 0.90:
            report += "\n   - Collect more training data for each student"
        if metrics['precision_macro'] < 0.90:
            report += "\n   - Review false positive cases"
        if metrics['recall_macro'] < 0.90:
            report += "\n   - Review false negative cases"
        
        # Save report
        report_path = self.output_dir / 'evaluation_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[SUCCESS] Saved: {report_path}")
        
        # Also save as TXT
        txt_path = self.output_dir / 'evaluation_report.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"[SUCCESS] Saved: {txt_path}")
    
    def run_full_evaluation(self):
        """Chạy đánh giá đầy đủ"""
        print("\n" + "="*60)
        print("[TARGET] COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Predictions
        print("\n Making predictions...")
        y_pred, distances, confidences = self.predict_with_distance(X_test)
        
        # Basic metrics
        metrics = self.evaluate_basic_metrics(y_test, y_pred)
        
        # Confusion matrices
        print("\n[STATS] Generating confusion matrices...")
        self.plot_confusion_matrix(y_test, y_pred, normalize=False)
        self.plot_confusion_matrix(y_test, y_pred, normalize=True)
        
        # Per-class metrics
        print("\n[STATS] Generating per-class metrics...")
        df_per_class = self.plot_per_class_metrics(y_test, y_pred)
        
        # Distance analysis
        print("\n[STATS] Analyzing distances...")
        self.plot_distance_distribution(distances, y_test, y_pred)
        
        # Confidence analysis
        print("\n[STATS] Analyzing confidence scores...")
        self.plot_confidence_analysis(confidences, y_test, y_pred)
        
        # Cross-validation
        print("\n[STATS] Running cross-validation...")
        X_all = np.vstack([X_train, X_test])
        y_all = np.hstack([y_train, y_test])
        cv_results = self.cross_validation_analysis(X_all, y_all, cv=5)
        
        # Generate report
        self.generate_summary_report(metrics, df_per_class)
        
        print("\n" + "="*60)
        print("[OK] EVALUATION COMPLETE!")
        print("="*60)
        print(f"\n[FOLDER] All results saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  - confusion_matrix.png")
        print("  - confusion_matrix_normalized.png")
        print("  - per_class_metrics.png")
        print("  - per_class_metrics.csv")
        print("  - distance_distribution.png")
        print("  - confidence_analysis.png")
        print("  - cross_validation.png")
        print("  - evaluation_report.md")
        print("  - evaluation_report.txt")


def main():
    """Main function"""
    print("="*60)
    print("  MODEL EVALUATION SYSTEM")
    print("="*60)
    
    try:
        evaluator = ModelEvaluator()
        evaluator.run_full_evaluation()
        
    except FileNotFoundError as e:
        print(f"\n[FAIL] Error: {e}")
        print("\nPlease train the model first:")
        print("  python src/train_model.py")
    
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

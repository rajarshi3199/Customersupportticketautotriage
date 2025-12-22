"""
Evaluation and metrics module for ticket classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import config
import os


class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, y_true, y_pred, y_proba=None, categories=None):
        """
        Initialize evaluator
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            categories: List of category names
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.categories = categories or config.CATEGORIES
        
        # Calculate metrics
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        self.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        self.f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 60)
        print(f"\nOverall Accuracy: {self.accuracy:.4f}")
        print(f"Weighted Precision: {self.precision:.4f}")
        print(f"Weighted Recall: {self.recall:.4f}")
        print(f"Weighted F1-Score: {self.f1:.4f}")
        print("\n" + "=" * 60)
    
    def print_detailed_report(self):
        """Print detailed classification report"""
        print("\n" + "=" * 60)
        print("DETAILED CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=self.categories,
            zero_division=0
        ))
        print("=" * 60)
    
    def plot_confusion_matrix(self, save_path=None, figsize=(10, 8)):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.categories,
            yticklabels=self.categories
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nConfusion matrix saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_class_distribution(self, save_path=None, figsize=(10, 6)):
        """Plot distribution of true vs predicted categories"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # True distribution
        true_counts = pd.Series(self.y_true).value_counts().sort_index()
        ax1.bar(range(len(true_counts)), true_counts.values, color='steelblue')
        ax1.set_xticks(range(len(true_counts)))
        ax1.set_xticklabels(true_counts.index, rotation=45, ha='right')
        ax1.set_title('True Category Distribution', fontweight='bold')
        ax1.set_ylabel('Count')
        
        # Predicted distribution
        pred_counts = pd.Series(self.y_pred).value_counts().sort_index()
        ax2.bar(range(len(pred_counts)), pred_counts.values, color='coral')
        ax2.set_xticks(range(len(pred_counts)))
        ax2.set_xticklabels(pred_counts.index, rotation=45, ha='right')
        ax2.set_title('Predicted Category Distribution', fontweight='bold')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_per_class_metrics(self):
        """Get per-class metrics"""
        report = classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=self.categories,
            output_dict=True,
            zero_division=0
        )
        
        per_class = {}
        for category in self.categories:
            if category in report:
                per_class[category] = {
                    'precision': report[category]['precision'],
                    'recall': report[category]['recall'],
                    'f1-score': report[category]['f1-score'],
                    'support': report[category]['support']
                }
        
        return per_class
    
    def print_per_class_metrics(self):
        """Print per-class metrics"""
        per_class = self.get_per_class_metrics()
        
        print("\n" + "=" * 60)
        print("PER-CLASS METRICS")
        print("=" * 60)
        print(f"{'Category':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 60)
        
        for category, metrics in per_class.items():
            print(f"{category:<25} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1-score']:<12.4f} {metrics['support']:<10}")
        
        print("=" * 60)
    
    def save_evaluation_report(self, output_path=None):
        """Save comprehensive evaluation report to file"""
        output_path = output_path or os.path.join(config.RESULTS_DIR, 'evaluation_report.txt')
        
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("OVERALL METRICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Accuracy: {self.accuracy:.4f}\n")
            f.write(f"Weighted Precision: {self.precision:.4f}\n")
            f.write(f"Weighted Recall: {self.recall:.4f}\n")
            f.write(f"Weighted F1-Score: {self.f1:.4f}\n\n")
            
            f.write("DETAILED CLASSIFICATION REPORT\n")
            f.write("-" * 60 + "\n")
            f.write(classification_report(
                self.y_true, 
                self.y_pred, 
                target_names=self.categories,
                zero_division=0
            ))
            f.write("\n")
            
            f.write("PER-CLASS METRICS\n")
            f.write("-" * 60 + "\n")
            per_class = self.get_per_class_metrics()
            for category, metrics in per_class.items():
                f.write(f"\n{category}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n")
        
        print(f"\nEvaluation report saved to: {output_path}")
    
    def generate_all_visualizations(self, output_dir=None):
        """Generate all evaluation visualizations"""
        output_dir = output_dir or config.RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Confusion matrix
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        self.plot_confusion_matrix(save_path=cm_path)
        
        # Distribution plots
        dist_path = os.path.join(output_dir, 'category_distribution.png')
        self.plot_class_distribution(save_path=dist_path)
        
        print(f"\nAll visualizations saved to: {output_dir}")


def evaluate_model(y_true, y_pred, y_proba=None, save_results=True):
    """
    Comprehensive model evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        save_results: Whether to save results to files
    
    Returns:
        ModelEvaluator: Evaluator object with all metrics
    """
    evaluator = ModelEvaluator(y_true, y_pred, y_proba)
    
    # Print summaries
    evaluator.print_summary()
    evaluator.print_detailed_report()
    evaluator.print_per_class_metrics()
    
    # Generate visualizations
    if save_results:
        evaluator.generate_all_visualizations()
        evaluator.save_evaluation_report()
    
    return evaluator


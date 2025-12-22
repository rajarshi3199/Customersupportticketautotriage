"""
Machine learning models for ticket classification
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import config


class TicketClassifier:
    """Main classifier for support tickets"""
    
    def __init__(self, model_type='ensemble'):
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.is_trained = False
    
    def _create_model(self):
        """Create the specified model"""
        if self.model_type == 'svm':
            return SVC(kernel='rbf', probability=True, random_state=config.RANDOM_STATE)
        
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        
        elif self.model_type == 'naive_bayes':
            return MultinomialNB(alpha=1.0)
        
        elif self.model_type == 'neural_network':
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=config.RANDOM_STATE,
                early_stopping=True
            )
        
        elif self.model_type == 'ensemble':
            # Create an ensemble of multiple classifiers
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
            svm = SVC(kernel='rbf', probability=True, random_state=config.RANDOM_STATE)
            nb = MultinomialNB(alpha=1.0)
            lr = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
            
            return VotingClassifier(
                estimators=[('rf', rf), ('svm', svm), ('nb', nb), ('lr', lr)],
                voting='soft'
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, vectorizer=None):
        """Train the classifier"""
        # Store vectorizer
        self.vectorizer = vectorizer
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print(f"Model ({self.model_type}) trained successfully!")
    
    def predict(self, X):
        """Predict categories for new tickets"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict category probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=config.CATEGORIES))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        if not self.is_trained:
            self.model = self._create_model()
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        print(f"\nCross-Validation Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def save(self, model_path=None, vectorizer_path=None):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_path = model_path or config.MODEL_SAVE_PATH
        vectorizer_path = vectorizer_path or config.VECTORIZER_SAVE_PATH
        
        joblib.dump(self.model, model_path)
        if self.vectorizer:
            joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"Model saved to {model_path}")
        if self.vectorizer:
            print(f"Vectorizer saved to {vectorizer_path}")
    
    def load(self, model_path=None, vectorizer_path=None):
        """Load a trained model"""
        model_path = model_path or config.MODEL_SAVE_PATH
        vectorizer_path = vectorizer_path or config.VECTORIZER_SAVE_PATH
        
        self.model = joblib.load(model_path)
        try:
            self.vectorizer = joblib.load(vectorizer_path)
        except FileNotFoundError:
            print("Warning: Vectorizer file not found")
        
        self.is_trained = True
        print(f"Model loaded from {model_path}")


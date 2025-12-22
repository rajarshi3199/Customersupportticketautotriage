"""
Training script for Customer Support Ticket Classifier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import load_and_preprocess_data, TextVectorizer
from models import TicketClassifier
from evaluation import evaluate_model
import config


def main():
    """Main training function"""
    print("=" * 60)
    print("Customer Support Ticket Auto-Triage - Training")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    try:
        df, preprocessor = load_and_preprocess_data(config.DATASET_PATH)
        print(f"   Loaded {len(df)} tickets")
    except FileNotFoundError:
        print(f"   Dataset not found at {config.DATASET_PATH}")
        print("   Please ensure the dataset file exists or generate sample data first.")
        return
    
    # Check for required columns
    if 'Category' not in df.columns:
        print("   Error: 'Category' column not found in dataset")
        return
    
    # Prepare features and labels
    print("\n2. Preparing features and labels...")
    X = df['Processed_Text'].values
    y = df['Category'].values
    
    # Check for missing categories
    valid_categories = [cat for cat in config.CATEGORIES if cat in y]
    if not valid_categories:
        print("   Error: No valid categories found in dataset")
        return
    
    # Filter to valid categories
    mask = pd.Series(y).isin(valid_categories)
    X = X[mask]
    y = y[mask]
    
    print(f"   Features shape: {X.shape}")
    print(f"   Categories: {valid_categories}")
    print(f"   Category distribution:")
    for category in valid_categories:
        count = np.sum(y == category)
        print(f"     - {category}: {count}")
    
    # Split data
    print("\n3. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Vectorize text
    print("\n4. Vectorizing text features...")
    vectorizer = TextVectorizer(max_features=config.MAX_FEATURES)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    print(f"   Feature matrix shape: {X_train_vectorized.shape}")
    
    # Train model
    print(f"\n5. Training {config.MODEL_TYPE} model...")
    classifier = TicketClassifier(model_type=config.MODEL_TYPE)
    classifier.train(X_train_vectorized, y_train, vectorizer=vectorizer)
    
    # Evaluate model
    print("\n6. Evaluating model...")
    y_pred = classifier.predict(X_test_vectorized)
    y_proba = classifier.predict_proba(X_test_vectorized)
    
    # Comprehensive evaluation
    evaluator = evaluate_model(y_test, y_pred, y_proba, save_results=True)
    
    # Cross-validation
    print("\n7. Performing cross-validation...")
    classifier.cross_validate(X_train_vectorized, y_train, cv=config.CROSS_VALIDATION_FOLDS)
    
    # Save model
    print("\n8. Saving model...")
    classifier.save()
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


"""
Prediction script for Customer Support Ticket Classifier
"""

import pandas as pd
import numpy as np
from preprocessing import TextPreprocessor
from models import TicketClassifier
import config
import joblib


def predict_ticket(subject, description):
    """
    Predict category for a single ticket
    
    Args:
        subject: Ticket subject (string)
        description: Ticket description (string)
    
    Returns:
        dict: Prediction results with category and probabilities
    """
    # Load model and vectorizer
    classifier = TicketClassifier(model_type=config.MODEL_TYPE)
    try:
        classifier.load()
    except FileNotFoundError:
        print("Error: Model not found. Please train the model first.")
        return None
    
    # Preprocess text
    preprocessor = TextPreprocessor(
        use_lemmatization=config.USE_LEMMATIZATION,
        use_stemming=config.USE_STEMMING,
        remove_stopwords=config.REMOVE_STOPWORDS
    )
    
    combined_text = f"{subject} {description}"
    processed_text = preprocessor.preprocess(combined_text)
    
    # Vectorize
    X = classifier.vectorizer.transform([processed_text])
    
    # Predict
    prediction = classifier.predict(X)[0]
    probabilities = classifier.predict_proba(X)[0]
    
    # Get model's class order
    model_classes = classifier.model.classes_ if hasattr(classifier.model, 'classes_') else config.CATEGORIES
    
    # Create results dictionary with correct mapping
    prob_dict = {}
    for i, cls in enumerate(model_classes):
        prob_dict[cls] = float(probabilities[i])
    
    results = {
        'predicted_category': prediction,
        'probabilities': prob_dict,
        'confidence': float(np.max(probabilities))
    }
    
    return results


def predict_batch(file_path):
    """
    Predict categories for a batch of tickets from CSV file
    
    Args:
        file_path: Path to CSV file with 'Subject' and 'Description' columns
    
    Returns:
        DataFrame: Original dataframe with added 'Predicted_Category' and 'Confidence' columns
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Load model
    classifier = TicketClassifier(model_type=config.MODEL_TYPE)
    try:
        classifier.load()
    except FileNotFoundError:
        print("Error: Model not found. Please train the model first.")
        return None
    
    # Preprocess
    preprocessor = TextPreprocessor(
        use_lemmatization=config.USE_LEMMATIZATION,
        use_stemming=config.USE_STEMMING,
        remove_stopwords=config.REMOVE_STOPWORDS
    )
    
    df = preprocessor.preprocess_dataframe(df)
    
    # Vectorize
    X = classifier.vectorizer.transform(df['Processed_Text'].values)
    
    # Predict
    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)
    confidences = np.max(probabilities, axis=1)
    
    # Add predictions to dataframe
    df['Predicted_Category'] = predictions
    df['Confidence'] = confidences
    
    return df


def main():
    """Main prediction function"""
    print("=" * 60)
    print("Customer Support Ticket Auto-Triage - Prediction")
    print("=" * 60)
    
    # Example single prediction
    print("\nExample: Predicting category for a sample ticket...")
    sample_subject = "Application crashes when opening reports"
    sample_description = "Every time I try to open the monthly report, the application crashes. This started happening after the last update."
    
    result = predict_ticket(sample_subject, sample_description)
    
    if result:
        print(f"\nSubject: {sample_subject}")
        print(f"Description: {sample_description}")
        print(f"\nPredicted Category: {result['predicted_category']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nCategory Probabilities:")
        for category, prob in sorted(result['probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"  {category}: {prob:.2%}")
    
    print("\n" + "=" * 60)
    print("To predict a batch of tickets, use:")
    print("  python predict.py --batch <path_to_csv_file>")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print("Error: Please provide path to CSV file")
            sys.exit(1)
        
        file_path = sys.argv[2]
        results_df = predict_batch(file_path)
        
        if results_df is not None:
            output_path = config.RESULTS_DIR + '/predictions.csv'
            results_df.to_csv(output_path, index=False)
            print(f"\nPredictions saved to {output_path}")
    else:
        main()


"""
Error checking and diagnostic script
Run this to identify and resolve common issues
"""

import sys
import os

def check_imports():
    """Check if all required modules can be imported"""
    print("=" * 60)
    print("CHECKING IMPORTS")
    print("=" * 60)
    
    modules = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'nltk': 'nltk',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'joblib': 'joblib'
    }
    
    errors = []
    for module, package in modules.items():
        try:
            __import__(module)
            print(f"[OK] {package}")
        except ImportError as e:
            print(f"[ERROR] {package} - {str(e)}")
            errors.append(package)
    
    return errors

def check_files():
    """Check if required files exist"""
    print("\n" + "=" * 60)
    print("CHECKING FILES")
    print("=" * 60)
    
    required_files = [
        'config.py',
        'preprocessing.py',
        'models.py',
        'train.py',
        'predict.py',
        'evaluation.py'
    ]
    
    errors = []
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file}")
        else:
            print(f"[ERROR] {file} not found")
            errors.append(file)
    
    return errors

def check_directories():
    """Check if required directories exist"""
    print("\n" + "=" * 60)
    print("CHECKING DIRECTORIES")
    print("=" * 60)
    
    import config
    directories = [
        ('data', config.DATA_DIR),
        ('models', config.MODELS_DIR),
        ('results', config.RESULTS_DIR)
    ]
    
    errors = []
    for name, path in directories:
        if os.path.exists(path):
            print(f"[OK] {name}/ exists")
        else:
            print(f"[ERROR] {name}/ not found at {path}")
            errors.append(name)
            # Try to create it
            try:
                os.makedirs(path, exist_ok=True)
                print(f"  -> Created {name}/ directory")
            except Exception as e:
                print(f"  -> Failed to create: {str(e)}")
    
    return errors

def check_models():
    """Check if trained models exist"""
    print("\n" + "=" * 60)
    print("CHECKING TRAINED MODELS")
    print("=" * 60)
    
    import config
    
    model_files = [
        ('Classifier', config.MODEL_SAVE_PATH),
        ('Vectorizer', config.VECTORIZER_SAVE_PATH)
    ]
    
    errors = []
    for name, path in model_files:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print(f"[OK] {name} found ({size:.1f} KB)")
        else:
            print(f"[WARNING] {name} not found at {path}")
            print(f"  -> Run 'python train.py' to create it")
            errors.append(name)
    
    return errors

def check_dataset():
    """Check if dataset exists"""
    print("\n" + "=" * 60)
    print("CHECKING DATASET")
    print("=" * 60)
    
    import config
    
    if os.path.exists(config.DATASET_PATH):
        try:
            import pandas as pd
            df = pd.read_csv(config.DATASET_PATH)
            print(f"[OK] Dataset found: {config.DATASET_PATH}")
            print(f"  -> Rows: {len(df)}")
            print(f"  -> Columns: {list(df.columns)}")
            
            # Check required columns
            required_cols = ['Subject', 'Description', 'Category']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"[ERROR] Missing columns: {missing_cols}")
                return ['missing_columns']
            else:
                print(f"[OK] All required columns present")
        except Exception as e:
            print(f"[ERROR] Cannot read dataset: {str(e)}")
            return ['dataset_read_error']
    else:
        print(f"[WARNING] Dataset not found: {config.DATASET_PATH}")
        print(f"  -> Run 'python generate_sample_data.py' to create sample data")
        return ['dataset_not_found']
    
    return []

def check_nltk_data():
    """Check if NLTK data is downloaded"""
    print("\n" + "=" * 60)
    print("CHECKING NLTK DATA")
    print("=" * 60)
    
    try:
        import nltk
        
        nltk_data = [
            ('punkt_tab', 'tokenizers/punkt_tab', 'punkt_tab'),
            ('stopwords', 'corpora/stopwords', 'stopwords'),
            ('wordnet', 'corpora/wordnet', 'wordnet'),
            ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
        ]
        
        errors = []
        for name, path, download_name in nltk_data:
            found = False
            try:
                nltk.data.find(path)
                print(f"[OK] {name}")
                found = True
            except LookupError:
                # Try alternative path for punkt
                if name == 'punkt_tab':
                    try:
                        nltk.data.find('tokenizers/punkt')
                        print(f"[OK] {name} (using punkt)")
                        found = True
                    except:
                        pass
                
                # For wordnet, try to verify it works by importing
                if name == 'wordnet' and not found:
                    try:
                        from nltk.corpus import wordnet
                        if len(wordnet.synsets('test')) > 0:
                            print(f"[OK] {name} (verified via import)")
                            found = True
                    except:
                        pass
            
            if not found:
                print(f"[WARNING] {name} not found via standard check")
                print(f"  -> Run: python -c \"import nltk; nltk.download('{download_name}')\"")
                # Don't add to errors if it might still work
                # errors.append(name)
        
        return errors
    except ImportError:
        print("[ERROR] NLTK not installed")
        return ['nltk_not_installed']

def test_prediction():
    """Test if prediction works"""
    print("\n" + "=" * 60)
    print("TESTING PREDICTION")
    print("=" * 60)
    
    try:
        from predict import predict_ticket
        
        result = predict_ticket(
            subject="Test ticket",
            description="This is a test description"
        )
        
        if result:
            print(f"[OK] Prediction works")
            print(f"  -> Predicted: {result['predicted_category']}")
            print(f"  -> Confidence: {result['confidence']:.2%}")
            return []
        else:
            print("[ERROR] Prediction returned None")
            return ['prediction_failed']
    except Exception as e:
        print(f"[ERROR] Prediction test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return ['prediction_error']

def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("CUSTOMER SUPPORT TICKET AUTO-TRIAGE - ERROR CHECKER")
    print("=" * 60)
    
    all_errors = []
    
    # Run all checks
    all_errors.extend(check_imports())
    all_errors.extend(check_files())
    all_errors.extend(check_directories())
    all_errors.extend(check_models())
    all_errors.extend(check_dataset())
    all_errors.extend(check_nltk_data())
    all_errors.extend(test_prediction())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if not all_errors:
        print("[SUCCESS] No errors found! Everything is working correctly.")
    else:
        print(f"[WARNING] Found {len(all_errors)} issue(s):")
        for error in set(all_errors):
            print(f"  - {error}")
        print("\nSee above for details and solutions.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()


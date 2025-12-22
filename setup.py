"""
Setup script to verify installation and download required resources
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"Error: Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"[OK] Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        print("\nDownloading NLTK data...")
        
        nltk_data = [
            'punkt',
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger'
        ]
        
        for data in nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data}' if data == 'punkt' 
                              else f'corpora/{data}' if data in ['stopwords', 'wordnet']
                              else f'taggers/{data}')
                print(f"  [OK] {data} already downloaded")
            except LookupError:
                print(f"  Downloading {data}...")
                if data == 'punkt':
                    nltk.download('punkt', quiet=True)
                elif data == 'stopwords':
                    nltk.download('stopwords', quiet=True)
                elif data == 'wordnet':
                    nltk.download('wordnet', quiet=True)
                elif data == 'averaged_perceptron_tagger':
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                print(f"  [OK] {data} downloaded")
        
        return True
    except ImportError:
        print("  Warning: NLTK not installed. Install with: pip install nltk")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'nltk': 'nltk',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    print("\nChecking dependencies...")
    missing = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  [OK] {package} installed")
        except ImportError:
            print(f"  [X] {package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    import config
    print("\nChecking directories...")
    
    directories = [
        config.DATA_DIR,
        config.MODELS_DIR,
        config.RESULTS_DIR
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"  [OK] {os.path.basename(directory)}/ exists")
        else:
            print(f"  [X] {os.path.basename(directory)}/ missing")
    
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("Customer Support Ticket Auto-Triage - Setup Verification")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies before proceeding.")
        sys.exit(1)
    
    # Download NLTK data
    download_nltk_data()
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 60)
    print("Setup verification complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Generate sample data: python generate_sample_data.py")
    print("2. Train the model: python train.py")
    print("3. Make predictions: python predict.py")
    print("=" * 60)

if __name__ == "__main__":
    main()


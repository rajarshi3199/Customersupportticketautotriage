# Customer Support Ticket Auto-Triage System

**An advanced machine learning project focused on revolutionizing customer support through intelligent ticket classification and automated routing systems.**

## 🎯 Project Objective

### Core Mission
To enhance operational efficiency and improve customer satisfaction by automating the initial processing of support tickets, reducing manual effort, and accelerating resolution times.

### Primary Goal
Develop and deploy a robust machine learning model capable of accurately classifying customer support tickets into predefined categories and routing them to the most appropriate team or agent.

## 📋 Project Overview

### Key Ticket Categories

- **Bug Report**: Reporting software defects or errors for immediate action
- **Feature Request**: Gathering user suggestions for new functionalities and enhancements
- **Technical Issue**: Addressing problems requiring specialized technical assistance and troubleshooting
- **Billing Inquiry**: Handling questions and discrepancies related to invoices, payments, and subscriptions
- **Account Management**: Resolving issues regarding user accounts, profiles, and access controls

## 📊 Dataset Structure

The dataset comprises historical customer support tickets, structured as follows:

| Field | Description | Type |
|-------|-------------|------|
| Ticket_ID | Unique identifier for each ticket | Integer |
| Subject | Short summary of the issue | String (Text) |
| Description | Detailed explanation of the problem | String (Long Text) |
| Category | Pre-assigned issue type (target variable) | Categorical String |
| Priority | Urgency level of the ticket | Categorical String |
| Timestamp | Date and time of ticket creation | Datetime |

## 🛠️ Technical Requirements

- **Python**: 3.8+
- **Key Libraries**: scikit-learn, pandas, numpy, NLTK/SpaCy, TensorFlow/PyTorch
- **Version Control**: Git (mandatory for collaboration)

## 📦 Installation

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd "C:\Users\HP\Desktop\Customer Support Ticket"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (if not automatically downloaded):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

5. **Download SpaCy model** (optional, for advanced NLP):
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 🚀 Quick Start

### 1. Generate Sample Data

If you don't have a dataset yet, generate sample data for testing:

```bash
python generate_sample_data.py
```

This will create a sample dataset at `data/sample_tickets.csv` with 500 tickets.

### 2. Configure Dataset Path

Update `config.py` to point to your dataset:

```python
DATASET_PATH = os.path.join(DATA_DIR, 'sample_tickets.csv')  # or your dataset path
```

### 3. Train the Model

Train the classifier on your dataset:

```bash
python train.py
```

The trained model will be saved to `models/ticket_classifier.pkl`.

### 4. Make Predictions

#### Single Ticket Prediction

```python
from predict import predict_ticket

result = predict_ticket(
    subject="Application crashes when opening reports",
    description="Every time I try to open the monthly report, the application crashes."
)

print(f"Predicted Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
```

#### Batch Prediction

```bash
python predict.py --batch path/to/tickets.csv
```

Predictions will be saved to `results/predictions.csv`.

## 📁 Project Structure

```
Customer Support Ticket/
│
├── config.py                 # Configuration settings
├── preprocessing.py          # Text preprocessing module
├── models.py                 # ML model definitions
├── train.py                  # Training script
├── predict.py                # Prediction script
├── generate_sample_data.py   # Sample data generator
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
│
├── data/                    # Data directory
│   ├── support_tickets.csv  # Main dataset (add your data here)
│   └── sample_tickets.csv   # Generated sample data
│
├── models/                  # Saved models
│   ├── ticket_classifier.pkl
│   └── text_vectorizer.pkl
│
└── results/                 # Results and outputs
    └── predictions.csv
```

## 🔧 Configuration

Edit `config.py` to customize:

- **Model Type**: Choose from `'svm'`, `'random_forest'`, `'naive_bayes'`, `'neural_network'`, or `'ensemble'` (default)
- **Text Preprocessing**: Enable/disable lemmatization, stemming, stopword removal
- **Feature Engineering**: Adjust TF-IDF parameters (max_features, ngram_range)
- **Training Parameters**: Test size, cross-validation folds, random state

## 📈 Model Performance

The system supports multiple machine learning algorithms:

1. **Support Vector Machine (SVM)**: Good for high-dimensional text data
2. **Random Forest**: Robust ensemble method with feature importance
3. **Naive Bayes**: Fast and efficient for text classification
4. **Neural Network**: Deep learning approach for complex patterns
5. **Ensemble**: Voting classifier combining multiple models (recommended)

## 🔍 Features

- **Advanced Text Preprocessing**: 
  - URL and email removal
  - Stopword filtering
  - Lemmatization/Stemming
  - Custom text cleaning

- **TF-IDF Vectorization**: 
  - N-gram support (unigrams and bigrams)
  - Feature selection and dimensionality reduction

- **Multiple ML Algorithms**: 
  - Support for various classification algorithms
  - Ensemble methods for improved accuracy

- **Comprehensive Evaluation**: 
  - Accuracy metrics
  - Classification reports
  - Confusion matrices
  - Cross-validation

- **Production Ready**: 
  - Model persistence
  - Batch prediction support
  - Confidence scores

## 📝 Usage Examples

### Training with Custom Dataset

```python
import pandas as pd
from preprocessing import load_and_preprocess_data, TextVectorizer
from models import TicketClassifier
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Preprocess
df, preprocessor = load_and_preprocess_data('your_dataset.csv')

# Prepare features
X = df['Processed_Text'].values
y = df['Category'].values

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
vectorizer = TextVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

classifier = TicketClassifier(model_type='ensemble')
classifier.train(X_train_vec, y_train, vectorizer)
classifier.evaluate(X_test_vec, y_test)
classifier.save()
```

### Making Predictions

```python
from predict import predict_ticket

# Predict single ticket
result = predict_ticket(
    subject="Login button not working",
    description="The login button doesn't respond when clicked."
)

print(result)
# Output:
# {
#     'predicted_category': 'Bug Report',
#     'probabilities': {...},
#     'confidence': 0.95
# }
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available for educational and commercial use.

## 🙏 Acknowledgments

- Built with scikit-learn, NLTK, and other open-source libraries
- Designed for customer support automation and efficiency

## 📧 Support

For issues, questions, or contributions, please open an issue in the repository.

---

**Note**: This is a machine learning project. Model performance depends on the quality and quantity of training data. Ensure you have a representative dataset for best results.


"""
Configuration file for Customer Support Ticket Auto-Triage System
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset configuration
DATASET_PATH = os.path.join(DATA_DIR, 'sample_tickets.csv')  # Using sample data
SAMPLE_DATASET_PATH = os.path.join(DATA_DIR, 'sample_tickets.csv')

# Model configuration
MODEL_TYPE = 'ensemble'  # Options: 'svm', 'random_forest', 'naive_bayes', 'neural_network', 'ensemble'
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'ticket_classifier.pkl')
VECTORIZER_SAVE_PATH = os.path.join(MODELS_DIR, 'text_vectorizer.pkl')

# Text preprocessing configuration
USE_LEMMATIZATION = True
USE_STEMMING = False
REMOVE_STOPWORDS = True
MIN_WORD_LENGTH = 2
MAX_FEATURES = 5000  # For TF-IDF vectorizer

# Model training configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
CROSS_VALIDATION_FOLDS = 5

# Ticket categories
CATEGORIES = [
    'Bug Report',
    'Feature Request',
    'Technical Issue',
    'Billing Inquiry',
    'Account Management'
]

# Priority levels
PRIORITIES = ['Low', 'Medium', 'High', 'Critical']

# NLP model (for SpaCy)
SPACY_MODEL = 'en_core_web_sm'


"""
Text preprocessing module for customer support tickets
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import config

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextPreprocessor:
    """Handles text preprocessing for support tickets"""
    
    def __init__(self, use_lemmatization=True, use_stemming=False, remove_stopwords=True):
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        
        # Initialize NLP tools with error handling
        try:
            self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        except LookupError:
            print("Warning: WordNet not found. Downloading...")
            nltk.download('wordnet', quiet=True)
            self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        
        self.stemmer = PorterStemmer() if use_stemming else None
        
        try:
            self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        except LookupError:
            print("Warning: Stopwords not found. Downloading...")
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
        # Add custom stopwords
        if self.stop_words:
            self.stop_words.update(['ticket', 'support', 'issue', 'problem', 'please', 'thank'])
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep spaces and basic punctuation)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords_from_tokens(self, tokens):
        """Remove stopwords from tokens"""
        if not self.remove_stopwords:
            return tokens
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens):
        """Lemmatize tokens"""
        if not self.use_lemmatization or self.lemmatizer is None:
            return tokens
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem(self, tokens):
        """Stem tokens"""
        if not self.use_stemming or self.stemmer is None:
            return tokens
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Filter short tokens
        tokens = [token for token in tokens if len(token) >= config.MIN_WORD_LENGTH]
        
        # Lemmatize or stem
        if self.use_lemmatization:
            tokens = self.lemmatize(tokens)
        elif self.use_stemming:
            tokens = self.stem(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_columns=['Subject', 'Description']):
        """Preprocess text columns in a dataframe"""
        df = df.copy()
        
        # Combine Subject and Description
        if 'Subject' in df.columns and 'Description' in df.columns:
            df['Combined_Text'] = df['Subject'].fillna('') + ' ' + df['Description'].fillna('')
        elif 'Subject' in df.columns:
            df['Combined_Text'] = df['Subject'].fillna('')
        elif 'Description' in df.columns:
            df['Combined_Text'] = df['Description'].fillna('')
        else:
            raise ValueError("Dataframe must contain 'Subject' or 'Description' column")
        
        # Preprocess combined text
        df['Processed_Text'] = df['Combined_Text'].apply(self.preprocess)
        
        return df


class TextVectorizer:
    """Handles text vectorization using TF-IDF"""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        self.is_fitted = False
    
    def fit(self, texts):
        """Fit the vectorizer on training texts"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
    
    def transform(self, texts):
        """Transform texts to vectors"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transformation")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform texts"""
        result = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return result


def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df['Subject'] = df['Subject'].fillna('')
    df['Description'] = df['Description'].fillna('')
    
    # Preprocess text
    preprocessor = TextPreprocessor(
        use_lemmatization=config.USE_LEMMATIZATION,
        use_stemming=config.USE_STEMMING,
        remove_stopwords=config.REMOVE_STOPWORDS
    )
    
    df = preprocessor.preprocess_dataframe(df)
    
    return df, preprocessor


# How to Run the Customer Support Ticket Auto-Triage Project

This guide provides step-by-step instructions to run the project from scratch.

## Prerequisites

- Python 3.8 or higher installed
- pip (Python package manager)
- Windows/Linux/Mac OS

## Step-by-Step Instructions

### Step 1: Navigate to Project Directory

Open your terminal/command prompt and navigate to the project folder:

```bash
cd "C:\Users\HP\Desktop\Customer Support Ticket"
```

### Step 2: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

**Note:** This may take a few minutes as it installs large packages like TensorFlow and PyTorch.

### Step 3: Verify Setup

Run the setup verification script to ensure everything is configured correctly:

```bash
python setup.py
```

This will:
- Check Python version
- Verify all dependencies are installed
- Download required NLTK data
- Create necessary directories

### Step 4: Prepare Your Dataset (or Generate Sample Data)

#### Option A: Use Your Own Dataset

1. Place your CSV file in the `data/` folder
2. Ensure your CSV has these columns:
   - `Ticket_ID` (Integer)
   - `Subject` (String)
   - `Description` (String)
   - `Category` (Categorical String - target variable)
   - `Priority` (Categorical String)
   - `Timestamp` (Datetime)

3. Update `config.py` to point to your dataset:
   ```python
   DATASET_PATH = os.path.join(DATA_DIR, 'your_dataset.csv')
   ```

#### Option B: Generate Sample Data (for testing)

If you don't have a dataset, generate sample data:

```bash
python generate_sample_data.py
```

This creates 500 sample tickets at `data/sample_tickets.csv`.

### Step 5: Train the Model

Train the machine learning classifier:

```bash
python train.py
```

**What happens:**
- Loads and preprocesses your data
- Splits data into training (80%) and testing (20%) sets
- Trains the ensemble classifier
- Evaluates model performance
- Saves the trained model to `models/ticket_classifier.pkl`
- Saves the text vectorizer to `models/text_vectorizer.pkl`
- Generates evaluation reports and visualizations in `results/`

**Expected output:**
- Model accuracy metrics
- Classification report
- Confusion matrix
- Cross-validation scores
- Saved model files

### Step 6: Make Predictions

#### Option A: Single Ticket Prediction

Run the prediction script for a demo:

```bash
python predict.py
```

This will predict the category for a sample ticket and show:
- Predicted category
- Confidence score
- Probability distribution across all categories

#### Option B: Predict Single Ticket Programmatically

Create a Python script or use Python interactively:

```python
from predict import predict_ticket

result = predict_ticket(
    subject="Application crashes when opening reports",
    description="Every time I try to open the monthly report, the application crashes."
)

print(f"Predicted Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
print("\nCategory Probabilities:")
for category, prob in result['probabilities'].items():
    print(f"  {category}: {prob:.2%}")
```

#### Option C: Batch Prediction

Predict categories for multiple tickets from a CSV file:

```bash
python predict.py --batch data/sample_tickets.csv
```

**Output:**
- Creates `results/predictions.csv` with:
  - Original ticket data
  - `Predicted_Category` column
  - `Confidence` column

## Project Structure

```
Customer Support Ticket/
│
├── config.py                 # Configuration settings
├── preprocessing.py          # Text preprocessing module
├── models.py                 # ML model definitions
├── train.py                  # Training script
├── predict.py                # Prediction script
├── evaluation.py             # Evaluation metrics
├── generate_sample_data.py   # Sample data generator
├── setup.py                  # Setup verification
├── requirements.txt          # Python dependencies
│
├── data/                     # Data directory
│   └── sample_tickets.csv    # Sample dataset
│
├── models/                   # Saved models
│   ├── ticket_classifier.pkl
│   └── text_vectorizer.pkl
│
└── results/                  # Results and outputs
    ├── predictions.csv
    ├── confusion_matrix.png
    ├── category_distribution.png
    └── evaluation_report.txt
```

## Common Use Cases

### 1. Training on New Data

```bash
# 1. Place your dataset in data/ folder
# 2. Update config.py with your dataset path
# 3. Train the model
python train.py
```

### 2. Making Predictions on New Tickets

```bash
# Single ticket
python predict.py

# Batch of tickets
python predict.py --batch path/to/new_tickets.csv
```

### 3. Changing Model Type

Edit `config.py`:

```python
MODEL_TYPE = 'random_forest'  # Options: 'svm', 'random_forest', 'naive_bayes', 'neural_network', 'ensemble'
```

Then retrain:
```bash
python train.py
```

### 4. Viewing Results

Check the `results/` folder for:
- **predictions.csv**: Batch prediction results
- **confusion_matrix.png**: Visual confusion matrix
- **category_distribution.png**: Category distribution plots
- **evaluation_report.txt**: Detailed evaluation metrics

## Troubleshooting

### Issue: "Model not found" error
**Solution:** Train the model first with `python train.py`

### Issue: "Dataset not found" error
**Solution:** 
- Check that your dataset exists at the path specified in `config.py`
- Or generate sample data: `python generate_sample_data.py`

### Issue: NLTK data download errors
**Solution:** Run `python setup.py` to download required NLTK resources

### Issue: Import errors
**Solution:** Install missing packages: `pip install -r requirements.txt`

### Issue: Low accuracy
**Solution:**
- Ensure you have enough training data (recommended: 500+ tickets)
- Check data quality and category balance
- Try different model types in `config.py`
- Adjust preprocessing parameters in `config.py`

## Quick Reference Commands

```bash
# Complete workflow
pip install -r requirements.txt
python setup.py
python generate_sample_data.py
python train.py
python predict.py

# Batch prediction
python predict.py --batch data/your_tickets.csv
```

## Next Steps

- Experiment with different model types
- Tune hyperparameters in `config.py`
- Add your own custom preprocessing
- Integrate with your support ticket system
- Deploy as a web service or API

For more details, see `README.md` and `QUICKSTART.md`.


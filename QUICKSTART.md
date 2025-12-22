# Quick Start Guide

Get up and running with the Customer Support Ticket Auto-Triage system in minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Verify Setup

```bash
python setup.py
```

This will:
- Check Python version
- Verify all dependencies are installed
- Download required NLTK data
- Create necessary directories

## Step 3: Generate Sample Data (Optional)

If you don't have your own dataset:

```bash
python generate_sample_data.py
```

This creates a sample dataset with 500 tickets at `data/sample_tickets.csv`.

## Step 4: Configure Dataset Path

Edit `config.py` and set your dataset path:

```python
DATASET_PATH = os.path.join(DATA_DIR, 'sample_tickets.csv')  # or your dataset path
```

Your dataset should have these columns:
- `Ticket_ID` (Integer)
- `Subject` (String)
- `Description` (String)
- `Category` (Categorical String - target variable)
- `Priority` (Categorical String)
- `Timestamp` (Datetime)

## Step 5: Train the Model

```bash
python train.py
```

This will:
- Load and preprocess your data
- Train the classifier (default: ensemble model)
- Evaluate performance
- Save the trained model to `models/`

## Step 6: Make Predictions

### Single Ticket Prediction

```python
from predict import predict_ticket

result = predict_ticket(
    subject="Application crashes when opening reports",
    description="Every time I try to open the monthly report, the application crashes."
)

print(f"Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Prediction

```bash
python predict.py --batch path/to/tickets.csv
```

Results will be saved to `results/predictions.csv`.

## Model Types

You can change the model type in `config.py`:

- `'ensemble'` - Voting classifier (recommended, default)
- `'svm'` - Support Vector Machine
- `'random_forest'` - Random Forest
- `'naive_bayes'` - Naive Bayes
- `'neural_network'` - Multi-layer Perceptron

## Troubleshooting

### "Model not found" error
Make sure you've trained the model first with `python train.py`

### "Dataset not found" error
Check that your dataset path in `config.py` is correct

### NLTK data download issues
Run `python setup.py` to download required NLTK resources

## Next Steps

- Experiment with different model types
- Tune hyperparameters in `config.py`
- Add your own custom preprocessing
- Integrate with your support ticket system

For more details, see `README.md`.


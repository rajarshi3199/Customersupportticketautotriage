# Dataset Structure Reference

## 📊 Dataset Structure

The Customer Support Ticket dataset must be a CSV file with the following structure:

### Required Columns

| Column Name | Data Type | Description | Required | Example |
|------------|----------|-------------|----------|---------|
| **Ticket_ID** | Integer | Unique identifier for each ticket | ✅ Yes | `1`, `2`, `3` |
| **Subject** | String (Text) | Short summary of the issue | ✅ Yes | `"Application crashes when opening reports"` |
| **Description** | String (Long Text) | Detailed explanation of the problem | ✅ Yes | `"Every time I try to open the monthly report, the application crashes."` |
| **Category** | Categorical String | Pre-assigned issue type (target variable) | ✅ Yes | `"Bug Report"`, `"Feature Request"`, etc. |
| **Priority** | Categorical String | Urgency level of the ticket | ⚠️ Optional | `"Low"`, `"Medium"`, `"High"`, `"Critical"` |
| **Timestamp** | Datetime | Date and time of ticket creation | ⚠️ Optional | `"2024-01-15 10:30:00"` |

### Category Values

The `Category` column must contain one of these exact values:

- `"Bug Report"` - Reporting software defects or errors
- `"Feature Request"` - User suggestions for new functionalities
- `"Technical Issue"` - Problems requiring technical assistance
- `"Billing Inquiry"` - Questions about invoices, payments, subscriptions
- `"Account Management"` - Issues with user accounts, profiles, access

### Priority Values (Optional)

If using Priority column, values should be:
- `"Low"`
- `"Medium"`
- `"High"`
- `"Critical"`

## 📍 Where Dataset Structure is Defined

### 1. **README.md** (Lines 23-34)
   - Main documentation with table format
   - Location: `README.md`

### 2. **generate_sample_data.py** (Lines 100-180)
   - Implementation showing actual structure
   - Creates sample dataset with proper format
   - Location: `generate_sample_data.py`

### 3. **config.py** (Lines 17-19, 35-42)
   - Dataset path configuration
   - Category definitions
   - Location: `config.py`

### 4. **preprocessing.py** (Lines 120-140)
   - Code that processes the dataset
   - Shows which columns are used
   - Location: `preprocessing.py`

## 📝 Example Dataset Format

### CSV Format
```csv
Ticket_ID,Subject,Description,Category,Priority,Timestamp
1,"Application crashes","The app crashes when opening reports","Bug Report","High","2024-01-15 10:30:00"
2,"Add dark mode","Please add dark mode theme option","Feature Request","Low","2024-01-15 11:00:00"
```

### Python DataFrame Format
```python
import pandas as pd

df = pd.DataFrame({
    'Ticket_ID': [1, 2, 3],
    'Subject': ['Application crashes', 'Add dark mode', 'Cannot connect'],
    'Description': ['The app crashes...', 'Please add...', 'Unable to connect...'],
    'Category': ['Bug Report', 'Feature Request', 'Technical Issue'],
    'Priority': ['High', 'Low', 'Medium'],
    'Timestamp': ['2024-01-15 10:30:00', '2024-01-15 11:00:00', '2024-01-15 12:00:00']
})
```

## 🔍 Verify Your Dataset Structure

### Check Dataset Format
```bash
python -c "import pandas as pd; df = pd.read_csv('data/your_dataset.csv'); print('Columns:', list(df.columns)); print('\nSample:'); print(df.head(2))"
```

### Using Error Checker
```bash
python check_errors.py
```

This will verify:
- ✅ Dataset file exists
- ✅ Required columns present
- ✅ Data can be read correctly

## 📂 Dataset File Location

### Default Path
- **Location**: `data/sample_tickets.csv`
- **Configured in**: `config.py` → `DATASET_PATH`

### Change Dataset Path

Edit `config.py`:
```python
DATASET_PATH = os.path.join(DATA_DIR, 'your_dataset.csv')
```

## 🎯 Minimum Required Columns

For the model to work, you **MUST** have:
1. ✅ `Subject` - Used for text classification
2. ✅ `Description` - Used for text classification  
3. ✅ `Category` - Target variable for training

**Optional but recommended:**
- `Ticket_ID` - For tracking
- `Priority` - For additional analysis
- `Timestamp` - For time-based analysis

## 📊 Sample Dataset

Generate a sample dataset with proper structure:

```bash
python generate_sample_data.py
```

This creates `data/sample_tickets.csv` with:
- 500 tickets
- All required columns
- Balanced distribution across categories
- Proper data types

## ⚠️ Common Issues

### Missing Columns
**Error**: `'Category' column not found in dataset`
**Solution**: Ensure your CSV has a `Category` column

### Wrong Category Values
**Error**: `No valid categories found in dataset`
**Solution**: Use exact category names: `"Bug Report"`, `"Feature Request"`, etc.

### Empty Values
**Issue**: Missing Subject or Description
**Solution**: The preprocessing handles empty values, but ensure most tickets have text

### Encoding Issues
**Issue**: Special characters not displaying correctly
**Solution**: Save CSV with UTF-8 encoding

## 📚 Related Files

- **README.md** - Main project documentation
- **HOW_TO_RUN.md** - Step-by-step running instructions
- **QUICKSTART.md** - Quick start guide
- **generate_sample_data.py** - Sample data generator
- **preprocessing.py** - Data preprocessing code
- **config.py** - Configuration and category definitions


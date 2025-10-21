# Tutorial Notebooks

Complete Jupyter notebooks for the ML tutorial series.

## Available Tutorials

### Tutorial 1: ML Fundamentals - Stop Overthinking, Start Building
**File:** `tutorial_01_fundamentals.ipynb`  
**Status:** Placeholder (coming soon)

**Topics:**
- What machine learning actually is
- Supervised vs unsupervised learning
- Building a simple spam classifier
- Train/test splits that matter
- Basic evaluation metrics

---

### Tutorial 2: Data Prep - Where ML Projects Actually Live or Die
**File:** `tutorial_02_data_prep.ipynb`  
**Status:** Complete

**Topics:**
- Loading data from SQL databases
- Handling missing values (the right way)
- Feature scaling and why it matters
- Creating proper train/test/validation splits
- Avoiding data leakage
- Feature engineering basics
- Encoding categorical variables

**Key Learnings:**
- You'll spend 80% of your time on data prep, not modeling
- Always fit scalers on training data only
- Use median (not mean) for imputation
- Stratify splits for imbalanced datasets
- Check for data leakage before training

**Outputs:**
- Processed training and test datasets
- Fitted scaler (saved for production use)
- Complete preprocessing pipeline

---

## Running the Notebooks

### Prerequisites

Install required packages:
```bash
pip install -r ../requirements.txt
```

### Launch Jupyter

From the repository root:
```bash
jupyter notebook
```

Or from this directory:
```bash
cd notebooks
jupyter notebook
```

### Recommended Order

Follow tutorials in sequence:
1. Tutorial 1: Fundamentals
2. Tutorial 2: Data Prep (available now)
3. Tutorial 3: Customer Churn Model (coming soon)
4. Tutorial 4: Feature Engineering (coming soon)
5. Tutorial 5: Model Evaluation (coming soon)
6. Tutorial 6: Employee Attrition (coming soon)
7. Tutorial 7: Hyperparameter Tuning (coming soon)
8. Tutorial 8: Deployment (coming soon)

## Tips for Success

1. **Run cells in order** - Each notebook builds on previous cells
2. **Experiment** - Modify code and see what happens
3. **Take notes** - Add markdown cells with your observations
4. **Save often** - Jupyter can crash, save your work
5. **Check the blog** - Each tutorial has a companion blog post with detailed explanations

## Data Files

The notebooks create and use data files in the `../data/` directory:

**Tutorial 2 creates:**
- `churn_X_train.csv` - Training features (scaled)
- `churn_X_test.csv` - Test features (scaled)
- `churn_y_train.csv` - Training labels
- `churn_y_test.csv` - Test labels
- `scaler.pkl` - Fitted StandardScaler for production

## Common Issues

### Import Errors
```python
ModuleNotFoundError: No module named 'sklearn'
```
**Solution:** Install requirements: `pip install -r requirements.txt`

### File Not Found
```python
FileNotFoundError: [Errno 2] No such file or directory: '../data/...'
```
**Solution:** Run notebooks from the `notebooks/` directory

### Kernel Died
**Solution:** Restart kernel and run all cells from the beginning

## Getting Help

- Check the blog post for the tutorial
- Review the documentation in `src/` directory
- Open an issue on GitHub
- Review the SKILL.md files if available

## Next Steps

After completing Tutorial 2:
- Proceed to Tutorial 3 for model building
- Review `src/data_prep.py` for reusable functions
- Check your saved data files in `data/` directory
- Save your fitted scaler for production use

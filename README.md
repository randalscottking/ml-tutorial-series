# Machine Learning Fundamentals Tutorial Series

A practical, hands-on tutorial series for building real machine learning models using Python and scikit-learn.

## What You'll Learn

Build production-ready ML models for:
- **Customer Churn Prediction** - Identify customers likely to leave
- **Employee Attrition Forecasting** - Predict employee turnover

## Tutorial Series

### ✅ Tutorial 1: ML Fundamentals - Stop Overthinking, Start Building
Learn the basics by building a spam classifier from scratch. Understand supervised vs unsupervised learning, train/test splits, and why you don't need advanced math to get started.

**Topics:** Classification basics, feature extraction, model training, evaluation metrics  
**Status:** Complete

### ✅ Tutorial 2: Data Prep - Where ML Projects Actually Live or Die
Master the critical 80% of ML work that happens before modeling. Handle missing values, scale features, avoid data leakage, and build proper train/test/validation splits.

**Topics:** Data cleaning, feature scaling, handling missing data, SQL data extraction  
**Status:** Complete

### ✅ Tutorial 3: Classification Models - Pick the Right Tool
Compare four classification algorithms (Logistic Regression, Decision Trees, Random Forest, XGBoost) on the same churn dataset. Learn when to use each algorithm and understand the performance vs interpretability tradeoff.

**Topics:** Algorithm selection, logistic regression, decision trees, random forests, XGBoost, model comparison  
**Status:** Complete

### 🚧 Tutorial 4: Regression Models - Predicting Numbers That Matter
Switch from classification to regression. Build models to predict customer lifetime value. Learn regression-specific metrics and how to handle outliers.

**Topics:** Regression algorithms, RMSE, MAE, R², outlier handling  
**Status:** Coming soon

### 🚧 Tutorial 5: Model Evaluation - Beyond Accuracy
Learn what metrics actually matter for business problems. Understand precision vs recall, ROC curves, and when accuracy is a terrible metric.

**Topics:** Confusion matrices, ROC-AUC, precision-recall curves, cross-validation, business metrics  
**Status:** Coming soon

### 🚧 Tutorial 6: Feature Engineering - The Art of Better Inputs
Transform raw data into features that actually help your models learn. Create interaction terms, handle categorical variables, and build time-based features.

**Topics:** Feature creation, encoding techniques, domain knowledge application  
**Status:** Coming soon

### 🚧 Tutorial 7: Hyperparameter Tuning - Making Models Actually Work
Move beyond default parameters. Use grid search and random search to find optimal model settings without overfitting.

**Topics:** Grid search, random search, cross-validation, overfitting prevention  
**Status:** Coming soon

### 🚧 Tutorial 8: Production ML - Getting Models into the Real World
Learn how to deploy models with Streamlit, log predictions, monitor performance, and handle model drift in production environments.

**Topics:** Model persistence, Streamlit deployment, monitoring dashboards, retraining strategies  
**Status:** Coming soon

## Repository Structure

```
ml-tutorial-series/
├── README.md
├── requirements.txt
├── data/
│   ├── customer_churn.csv (coming soon)
│   └── employee_attrition.csv (coming soon)
├── notebooks/
│   ├── tutorial_01_fundamentals.ipynb ✅
│   ├── tutorial_02_data_prep.ipynb ✅
│   ├── tutorial_03_classification_models.md ✅
│   ├── tutorial_04_regression_models.md 🚧
│   ├── tutorial_05_model_evaluation.md 🚧
│   ├── tutorial_06_feature_engineering.md 🚧
│   ├── tutorial_07_hyperparameter_tuning.md 🚧
│   └── tutorial_08_production.md 🚧
├── src/
│   ├── data_prep.py (coming soon)
│   ├── models.py (coming soon)
│   └── evaluation.py (coming soon)
└── sql/
    ├── extract_churn_data.sql (coming soon)
    └── extract_attrition_data.sql (coming soon)
```

## Prerequisites

- Python 3.8+
- Basic Python knowledge
- SQL familiarity
- Jupyter Notebook or code editor

## Installation

```bash
# Clone the repository
git clone https://github.com/randalscottking/ml-tutorial-series.git
cd ml-tutorial-series

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Getting Started

1. Start with Tutorial 1 in the `notebooks/` directory
2. Follow tutorials in order - each builds on previous concepts
3. Complete datasets will be added as tutorials progress
4. SQL scripts for data extraction will be provided

## Requirements

All required packages are listed in `requirements.txt`:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- jupyter
- sqlalchemy (for SQL integration)

## Code Style

- Direct, practical approach
- Real examples, not toy datasets
- SQL integration where relevant
- Production-ready code patterns
- Complete working examples

## Contributing

Found an issue or want to suggest improvements? Open an issue or submit a pull request.

## License

MIT License - See LICENSE file for details

## Questions?

Visit the tutorial series on randalscottking.com for detailed explanations and walkthroughs.

## About

Created by Randal Scott King - Data scientist, engineer, and practitioner focused on practical ML applications.

Website: [randalscottking.com](https://randalscottking.com)

---

**Last Updated:** October 20, 2025  
**Current Progress:** 3 of 8 tutorials complete

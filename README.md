# Machine Learning Fundamentals Tutorial Series

A practical, hands-on tutorial series for building real machine learning models using Python and scikit-learn.

## What You'll Learn

Build production-ready ML models for:
- **Customer Churn Prediction** - Identify customers likely to leave
- **Employee Attrition Forecasting** - Predict employee turnover

## Tutorial Series

### Tutorial 1: ML Fundamentals - Stop Overthinking, Start Building
Learn the basics by building a spam classifier from scratch. Understand supervised vs unsupervised learning, train/test splits, and why you don't need advanced math to get started.

**Topics:** Classification basics, feature extraction, model training, evaluation metrics

### Tutorial 2: Data Prep - Where ML Projects Actually Live or Die
Master the critical 80% of ML work that happens before modeling. Handle missing values, scale features, avoid data leakage, and build proper train/test/validation splits.

**Topics:** Data cleaning, feature scaling, handling missing data, SQL data extraction

### Tutorial 3: Building Your First Real Model - Customer Churn Prediction
Build a complete customer churn prediction model from raw data to deployment-ready code.

**Topics:** Logistic regression, decision trees, random forests, feature importance

### Tutorial 4: Feature Engineering - The Art of Making Better Inputs
Transform raw data into features that actually help your models learn. Create interaction terms, handle categorical variables, and build time-based features.

**Topics:** Feature creation, encoding techniques, domain knowledge application

### Tutorial 5: Model Evaluation - Beyond Accuracy
Learn what metrics actually matter for business problems. Understand precision vs recall, ROC curves, and when accuracy is a terrible metric.

**Topics:** Confusion matrices, ROC-AUC, precision-recall curves, business metrics

### Tutorial 6: Employee Attrition - Applying What You Learned
Take everything from the churn model and apply it to employee attrition. See how the same techniques work across different business problems.

**Topics:** Transfer learning concepts, handling different data types, business context

### Tutorial 7: Hyperparameter Tuning - Making Your Models Better
Move beyond default parameters. Use grid search and random search to find optimal model settings without overfitting.

**Topics:** Grid search, random search, cross-validation, overfitting prevention

### Tutorial 8: Deployment and Monitoring - Getting Models into Production
Learn how to deploy models, log predictions, monitor performance, and handle model drift in production environments.

**Topics:** Model persistence, API deployment, monitoring dashboards, retraining strategies

## Repository Structure

```
ml-tutorial-series/
├── README.md
├── requirements.txt
├── data/
│   ├── customer_churn.csv
│   └── employee_attrition.csv
├── notebooks/
│   ├── tutorial_01_fundamentals.ipynb
│   ├── tutorial_02_data_prep.ipynb
│   ├── tutorial_03_churn_model.ipynb
│   ├── tutorial_04_feature_engineering.ipynb
│   ├── tutorial_05_model_evaluation.ipynb
│   ├── tutorial_06_employee_attrition.ipynb
│   ├── tutorial_07_hyperparameter_tuning.ipynb
│   └── tutorial_08_deployment.ipynb
├── src/
│   ├── data_prep.py
│   ├── models.py
│   ├── evaluation.py
│   └── deployment.py
└── sql/
    ├── extract_churn_data.sql
    └── extract_attrition_data.sql
```

## Prerequisites

- Python 3.8+
- Basic Python knowledge
- SQL familiarity
- Jupyter Notebook

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
3. Complete datasets are in the `data/` directory
4. SQL scripts for data extraction are in `sql/`

## Requirements

All required packages are listed in `requirements.txt`:
- pandas
- numpy
- scikit-learn
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

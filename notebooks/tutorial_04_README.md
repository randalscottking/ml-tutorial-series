# Tutorial 4: Regression Models - Predicting Customer Lifetime Value

**Blog Post:** [Regression Models: When You Need to Predict Actual Numbers](https://randalscottking.com/regression-models-predict-actual-numbers/)

## Overview

This tutorial teaches you when to use regression instead of classification and how to build production-ready models for predicting continuous values. We build a customer lifetime value (CLV) prediction model using three different regression approaches.

## What You'll Learn

- **Classification vs Regression**: When to predict numbers instead of categories
- **Linear Regression**: Fast baseline with interpretable coefficients
- **Ridge Regression**: Handling correlated features with regularization
- **Random Forest Regressor**: Capturing non-linear relationships
- **Regression Metrics**: MAE, RMSE, and R² Score explained
- **Model Selection**: Decision framework for choosing the right algorithm

## Prerequisites

Complete Tutorial 2 (Data Preparation) first. You'll need:
- Cleaned and scaled training/validation/test datasets
- Revenue target variables (y_train_revenue.csv, etc.)

## Files

- `src/tutorial_04_regression_models.py` - Complete regression pipeline
- `notebooks/rf_predictions.png` - Actual vs predicted revenue plot
- `notebooks/regression_comparison.png` - Model performance comparison

## Running the Code

```bash
# From the repository root
cd src
python tutorial_04_regression_models.py
```

## Expected Output

The script will:
1. Train three regression models (Linear, Ridge, Random Forest)
2. Compare performance metrics (MAE, RMSE, R²)
3. Show feature importance/coefficients
4. Generate visualizations
5. Optionally train XGBoost (if installed)

### Typical Performance

On the churn dataset predicting CLV:

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | $120-200 | $180-300 | 0.65-0.75 |
| Ridge Regression | $115-195 | $175-295 | 0.66-0.76 |
| Random Forest | $100-170 | $150-260 | 0.72-0.82 |

## Key Concepts

### When to Use Each Model

**Linear Regression:**
- ✅ Need results in 30 minutes
- ✅ Interpretability is critical
- ✅ Relationships look mostly linear
- ✅ Small dataset (<10K rows)

**Ridge Regression:**
- ✅ Linear regression coefficients are unstable
- ✅ Many correlated features
- ✅ Want regularization without complexity

**Random Forest:**
- ✅ Have 2-4 hours to train properly
- ✅ Performance matters more than interpretability
- ✅ Relationships are non-linear
- ✅ Dataset is 10K-1M rows

### Understanding Regression Metrics

**MAE (Mean Absolute Error):**
- Average dollar amount you're off by
- If MAE = $150, predictions are off by $150 on average
- Easy to explain to stakeholders

**RMSE (Root Mean Squared Error):**
- Penalizes large errors more heavily
- Always higher than MAE if you have outliers
- More mathematically proper, less intuitive

**R² Score:**
- How much variance your model explains (0 to 1)
- 0.80 = model explains 80% of revenue variance
- Higher is better, but context matters

## Real-World Application

### Customer Lifetime Value Prediction

If your Random Forest achieves R² = 0.78 and MAE = $145:

**Business Impact:**
- Segment customers by predicted CLV
- High CLV (>$5K) → VIP treatment
- Medium CLV ($1K-$5K) → Standard retention
- Low CLV (<$1K) → Automated outreach

**ROI Calculation:**
- 10,000 at-risk customers
- Retention call costs $50
- Model identifies top 2,000 highest CLV
- Save: $400K → $100K retention budget
- Better allocation: Focus on customers worth 3x more

## Common Pitfalls

1. **Ignoring outliers** - One $50K customer skews linear models
2. **Not checking residuals** - Plot predicted vs actual for patterns
3. **Using wrong metrics** - MAE for business, RMSE for models, R² for variance
4. **Overfitting** - If training R² = 0.95 but validation = 0.65, you're overfitting

## Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

# Optional but recommended:
pip install xgboost  # For 2-3% better performance
```

## Next Steps

**Tutorial 5:** Model Evaluation - Metrics That Actually Matter  
**Tutorial 6:** Feature Engineering - The Most Important Step

## The Bottom Line

Regression is classification's slightly easier sibling. Same workflow, different target. Start with linear regression, upgrade to random forest if you have time, and only reach for XGBoost when those extra percentage points matter.

Your feature engineering will have a bigger impact than your algorithm choice 90% of the time. A random forest with great features beats XGBoost with mediocre features every single time.

---

**Series:** [Machine Learning Fundamentals Tutorial Series](https://randalscottking.com/machine-learning-fundamentals-tutorial/)  
**Previous:** [Tutorial 3 - Classification Models](https://randalscottking.com/classification-models-pick-the-right-tool/)  
**Blog:** [randalscottking.com](https://randalscottking.com)

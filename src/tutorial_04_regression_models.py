"""
Tutorial 4: Regression Models - Predicting Customer Lifetime Value
===================================================================

This script demonstrates three regression approaches for predicting continuous values:
1. Linear Regression - Fast baseline, interpretable coefficients
2. Ridge Regression - Handles correlated features with regularization
3. Random Forest Regressor - Captures non-linear relationships

Dataset: Customer churn data, predicting total_revenue (CLV)
Blog Post: https://randalscottking.com/regression-models-predict-actual-numbers/
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")
# Load cleaned data from Tutorial 2
X_train = pd.read_csv('../data/X_train_scaled.csv')
X_val = pd.read_csv('../data/X_val_scaled.csv')
X_test = pd.read_csv('../data/X_test_scaled.csv')

# Load revenue targets (instead of churn labels)
y_train = pd.read_csv('../data/y_train_revenue.csv')['total_revenue']
y_val = pd.read_csv('../data/y_val_revenue.csv')['total_revenue']
y_test = pd.read_csv('../data/y_test_revenue.csv')['total_revenue']

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")
print(f"\nRevenue statistics:")
print(f"Mean: ${y_train.mean():.2f}")
print(f"Median: ${y_train.median():.2f}")
print(f"Std Dev: ${y_train.std():.2f}")

# ============================================================================
# MODEL 1: LINEAR REGRESSION
# ============================================================================

print("\n" + "="*70)
print("LINEAR REGRESSION")
print("="*70)

# Train linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr.predict(X_val)

# Evaluate
print("\nLinear Regression Results:")
print(f"MAE: ${mean_absolute_error(y_val, y_pred_lr):.2f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_val, y_pred_lr)):.2f}")
print(f"R² Score: {r2_score(y_val, y_pred_lr):.3f}")

# Feature coefficients (interpretability!)
coefficients = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': lr.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print("\nTop 10 Most Important Features:")
print(coefficients.head(10).to_string(index=False))

# ============================================================================
# MODEL 2: RIDGE REGRESSION
# ============================================================================

print("\n" + "="*70)
print("RIDGE REGRESSION (with Regularization)")
print("="*70)

# Try different regularization strengths
param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
}

ridge = Ridge()
grid_search = GridSearchCV(
    ridge,
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

# Find best alpha
print("Finding optimal regularization strength...")
grid_search.fit(X_train, y_train)
best_ridge = grid_search.best_estimator_

print(f"Best alpha: {grid_search.best_params_['alpha']}")

# Make predictions
y_pred_ridge = best_ridge.predict(X_val)

# Evaluate
print("\nRidge Regression Results:")
print(f"MAE: ${mean_absolute_error(y_val, y_pred_ridge):.2f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_val, y_pred_ridge)):.2f}")
print(f"R² Score: {r2_score(y_val, y_pred_ridge):.3f}")

# Compare coefficients to linear regression
ridge_coefs = pd.DataFrame({
    'feature': X_train.columns,
    'linear_coef': lr.coef_,
    'ridge_coef': best_ridge.coef_,
    'difference': np.abs(lr.coef_ - best_ridge.coef_)
}).sort_values('difference', ascending=False)

print("\nCoefficient Comparison (Top 10 Differences):")
print(ridge_coefs.head(10).to_string(index=False))

# ============================================================================
# MODEL 3: RANDOM FOREST REGRESSOR
# ============================================================================

print("\n" + "="*70)
print("RANDOM FOREST REGRESSOR")
print("="*70)

# Train random forest regressor
print("Training Random Forest (this may take a minute)...")
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_reg.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_reg.predict(X_val)

# Evaluate
print("\nRandom Forest Regressor Results:")
print(f"MAE: ${mean_absolute_error(y_val, y_pred_rf):.2f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_val, y_pred_rf)):.2f}")
print(f"R² Score: {r2_score(y_val, y_pred_rf):.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_reg.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_rf, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], 
         [y_val.min(), y_val.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Revenue ($)', fontsize=12)
plt.ylabel('Predicted Revenue ($)', fontsize=12)
plt.title('Random Forest: Actual vs Predicted Revenue', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../notebooks/rf_predictions.png', dpi=150, bbox_inches='tight')
print("\nSaved plot: notebooks/rf_predictions.png")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

# Create comparison DataFrame
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Random Forest'],
    'MAE': [
        mean_absolute_error(y_val, y_pred_lr),
        mean_absolute_error(y_val, y_pred_ridge),
        mean_absolute_error(y_val, y_pred_rf)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_val, y_pred_lr)),
        np.sqrt(mean_squared_error(y_val, y_pred_ridge)),
        np.sqrt(mean_squared_error(y_val, y_pred_rf))
    ],
    'R²': [
        r2_score(y_val, y_pred_lr),
        r2_score(y_val, y_pred_ridge),
        r2_score(y_val, y_pred_rf)
    ]
})

print("\nFinal Model Comparison:")
print(results.to_string(index=False))

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# MAE comparison
results.plot(x='Model', y='MAE', kind='bar', ax=axes[0], legend=False, color='steelblue')
axes[0].set_title('Mean Absolute Error\n(Lower is Better)', fontweight='bold')
axes[0].set_ylabel('MAE ($)')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=45)

# RMSE comparison
results.plot(x='Model', y='RMSE', kind='bar', ax=axes[1], legend=False, color='coral')
axes[1].set_title('Root Mean Squared Error\n(Lower is Better)', fontweight='bold')
axes[1].set_ylabel('RMSE ($)')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=45)

# R² comparison
results.plot(x='Model', y='R²', kind='bar', ax=axes[2], legend=False, color='seagreen')
axes[2].set_title('R² Score\n(Higher is Better)', fontweight='bold')
axes[2].set_ylabel('R² Score')
axes[2].set_xlabel('')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../notebooks/regression_comparison.png', dpi=150, bbox_inches='tight')
print("Saved plot: notebooks/regression_comparison.png")

# ============================================================================
# BONUS: XGBOOST REGRESSOR (Optional)
# ============================================================================

print("\n" + "="*70)
print("BONUS: XGBOOST REGRESSOR")
print("="*70)

try:
    import xgboost as xgb
    
    print("Training XGBoost (this may take a minute)...")
    xgb_reg = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    xgb_reg.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False)
    
    y_pred_xgb = xgb_reg.predict(X_val)
    
    print("\nXGBoost Regressor Results:")
    print(f"MAE: ${mean_absolute_error(y_val, y_pred_xgb):.2f}")
    print(f"RMSE: ${np.sqrt(mean_squared_error(y_val, y_pred_xgb)):.2f}")
    print(f"R² Score: {r2_score(y_val, y_pred_xgb):.3f}")
    
except ImportError:
    print("\nXGBoost not installed. Install with: pip install xgboost")
    print("Note: XGBoost typically provides 2-3% better R² than Random Forest")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. Linear Regression: Fast, interpretable, good baseline
   - Use when: Relationships are linear, need quick results
   - Training time: < 1 second

2. Ridge Regression: Handles correlated features better
   - Use when: Features are correlated, linear regression unstable
   - Training time: ~5-10 seconds (with cross-validation)

3. Random Forest: Best performance, captures non-linearity
   - Use when: Performance matters, have time to train
   - Training time: 1-2 minutes
   
4. Feature engineering (Tutorial 6) impacts results more than algorithm choice

5. Regression metrics:
   - MAE: Average $ amount off (easy to explain)
   - RMSE: Penalizes large errors more
   - R²: Variance explained (0-1 scale)

Next: Tutorial 5 - Model Evaluation: Metrics That Actually Matter
""")

print("\n✅ Tutorial 4 Complete!")

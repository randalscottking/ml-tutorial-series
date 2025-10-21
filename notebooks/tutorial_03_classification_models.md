# Tutorial 3: Classification Models - Pick the Right Tool

**Part of the Machine Learning Tutorial Series**  
**What You'll Build:** Customer churn prediction model comparing 4 algorithms  
**Time:** ~45 minutes  
**Prerequisites:** Tutorial 1 (Fundamentals) and Tutorial 2 (Data Prep)

---

## The Reality of Algorithm Selection

Here's the thing nobody tells you when you're starting out: the algorithm usually matters less than you think. I've seen data scientists spend weeks optimizing model selection when their real problem was dirty data or bad features.

But—and this is important—*sometimes* the algorithm choice makes a massive difference. Knowing when it matters (and when it doesn't) is what separates people who ship working ML systems from people who endlessly tweak hyperparameters.

In this tutorial, we're going to build the same churn prediction model four different ways. You'll see exactly where each algorithm shines and where it falls flat. No theory. No proofs. Just working code and honest comparisons.

## What We're Building

We're predicting customer churn using the same dataset we cleaned in Tutorial 2. We'll implement:

1. **Logistic Regression** - The workhorse
2. **Decision Tree** - The explainer
3. **Random Forest** - The reliable performer
4. **Gradient Boosting** - The competition winner

By the end, you'll know which one to reach for first.

---

## The Dataset (Quick Recap)

From Tutorial 2, we have a cleaned customer dataset with:
- **Features:** Account age, monthly charges, contract type, support tickets, usage patterns
- **Target:** Churned (1) or Stayed (0)
- **Size:** 7,000 customers after cleaning
- **Split:** 70% train, 15% validation, 15% test

If you haven't done Tutorial 2, go do it. Data prep matters more than this tutorial.

---

## Algorithm 1: Logistic Regression

### Why Start Here?

Logistic regression is where you should start for binary classification. Always. Here's why:

- **Fast to train** - Seconds, not minutes
- **Easy to interpret** - You can explain it to non-technical stakeholders
- **Good baseline** - Shows you what simple relationships exist
- **Robust** - Doesn't overfit easily
- **Works well** - Surprisingly good on clean data

It's not sexy. It won't win Kaggle competitions. But it's shipped in more production systems than any fancy deep learning model.

### The Code

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load our cleaned data from Tutorial 2
X_train = pd.read_csv('data/X_train_scaled.csv')
X_val = pd.read_csv('data/X_val_scaled.csv')
X_test = pd.read_csv('data/X_test_scaled.csv')
y_train = pd.read_csv('data/y_train.csv')['churned']
y_val = pd.read_csv('data/y_val.csv')['churned']
y_test = pd.read_csv('data/y_test.csv')['churned']

# Train logistic regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_val)
y_pred_proba = log_reg.predict_proba(X_val)[:, 1]

# Evaluate
print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.3f}")
print(f"Precision: {precision_score(y_val, y_pred):.3f}")
print(f"Recall: {recall_score(y_val, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_val, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_proba):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))
```

### What to Expect

On a well-prepped churn dataset, logistic regression typically gives:
- **Accuracy:** 75-80%
- **Precision:** 60-70% (of predicted churners, 60-70% actually churn)
- **Recall:** 50-65% (catches 50-65% of actual churners)
- **ROC-AUC:** 0.75-0.82

If you're getting worse than this, your data prep needs work. Go back to Tutorial 2.

### When to Use Logistic Regression

**Use it when:**
- You need to explain predictions to business stakeholders
- Training time matters (millions of rows)
- You want feature importance rankings
- Your problem has mostly linear relationships
- You need a fast baseline

**Don't use it when:**
- You have complex non-linear patterns
- Feature interactions are critical
- You're competing in Kaggle (only half-joking)

---

## Algorithm 2: Decision Tree

### The Explainability Champion

Decision trees ask yes/no questions about your features until they reach a prediction. Think of it like a flowchart:

- "Is monthly charge > $70?" → Yes → "Has support tickets > 3?" → Yes → **CHURN**

This makes them incredibly easy to explain. You can literally draw the decision path for any prediction.

The downside? They overfit like crazy if you're not careful.

### The Code

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Train decision tree with depth limit to prevent overfitting
dt = DecisionTreeClassifier(
    max_depth=8,           # Limit tree depth
    min_samples_split=50,  # Need at least 50 samples to split
    min_samples_leaf=20,   # Each leaf needs 20+ samples
    random_state=42
)
dt.fit(X_train, y_train)

# Predictions
y_pred_dt = dt.predict(X_val)
y_pred_proba_dt = dt.predict_proba(X_val)[:, 1]

# Evaluate
print("Decision Tree Results:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_dt):.3f}")
print(f"Precision: {precision_score(y_val, y_pred_dt):.3f}")
print(f"Recall: {recall_score(y_val, y_pred_dt):.3f}")
print(f"F1 Score: {f1_score(y_val, y_pred_dt):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_proba_dt):.3f}")

# Visualize the tree (optional, but cool)
plt.figure(figsize=(20,10))
tree.plot_tree(dt, 
               feature_names=X_train.columns, 
               class_names=['Stayed', 'Churned'],
               filled=True,
               max_depth=3)  # Only show top 3 levels
plt.savefig('decision_tree_viz.png', dpi=150, bbox_inches='tight')
plt.close()
print("Decision tree visualization saved to decision_tree_viz.png")
```

### What to Expect

A properly constrained decision tree will give:
- **Accuracy:** 72-78%
- **Precision:** 55-65%
- **Recall:** 55-70%
- **ROC-AUC:** 0.70-0.78

Usually slightly worse than logistic regression, but way easier to explain.

### The Overfitting Problem

If you don't constrain your tree (set `max_depth`, `min_samples_split`, etc.), you'll get:
- **Training accuracy:** 99%+
- **Validation accuracy:** 65%

That's overfitting. The tree memorized the training data instead of learning patterns.

### When to Use Decision Trees

**Use them when:**
- You need to explain every single prediction
- You're presenting to non-technical executives
- You want to understand feature interactions
- Interpretability > Performance

**Don't use them when:**
- You want the best possible accuracy
- You have noisy data (they'll overfit)
- You can't tune the constraints properly

---

## Algorithm 3: Random Forest

### The Reliable Workhorse

Random forests are what you get when you train hundreds of decision trees on random subsets of your data, then average their predictions. It's ensemble learning, and it works stupidly well.

**Why random forests are great:**
- Hard to overfit (the randomness prevents it)
- Usually 2-5% more accurate than single trees
- Still somewhat interpretable (feature importance)
- Robust to outliers and missing data
- Works well out-of-the-box with minimal tuning

This is my default algorithm for most classification problems. Not because it's the best, but because it's reliably good with minimal effort.

### The Code

```python
from sklearn.ensemble import RandomForestClassifier

# Train random forest
rf = RandomForestClassifier(
    n_estimators=100,      # 100 trees
    max_depth=10,          # Deeper than single tree
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',   # Randomness in feature selection
    random_state=42,
    n_jobs=-1             # Use all CPU cores
)
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_val)
y_pred_proba_rf = rf.predict_proba(X_val)[:, 1]

# Evaluate
print("Random Forest Results:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_rf):.3f}")
print(f"Precision: {precision_score(y_val, y_pred_rf):.3f}")
print(f"Recall: {recall_score(y_val, y_pred_rf):.3f}")
print(f"F1 Score: {f1_score(y_val, y_pred_rf):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_proba_rf):.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance.head(10)['feature'], 
         feature_importance.head(10)['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Features - Random Forest')
plt.tight_layout()
plt.savefig('feature_importance_rf.png', dpi=150, bbox_inches='tight')
plt.close()
print("Feature importance plot saved")
```

### What to Expect

Random forests typically improve on single decision trees:
- **Accuracy:** 78-83%
- **Precision:** 65-75%
- **Recall:** 60-72%
- **ROC-AUC:** 0.78-0.85

That 3-5% accuracy bump might not sound like much, but on a 100,000-customer dataset, that's 3,000-5,000 more correct predictions.

### Training Time

Random forests are slower than logistic regression. On this dataset:
- **Logistic regression:** 2 seconds
- **Random forest:** 15-30 seconds

That's still fast enough for most use cases.

### When to Use Random Forest

**Use it when:**
- You want good performance without much tuning
- You need feature importance rankings
- You have enough data (1,000+ samples minimum)
- Training time under a few minutes is acceptable
- You're building a production system

**Don't use it when:**
- You need millisecond prediction times (hundreds of trees = slow)
- You need to explain individual predictions in detail
- You're on a tiny dataset (< 500 samples)

---

## Algorithm 4: Gradient Boosting (XGBoost)

### The Competition Winner

Gradient boosting is what wins Kaggle competitions. It's what I use when I need the absolute best performance and I have time to tune it properly.

Instead of training trees in parallel (like random forest), gradient boosting trains them sequentially. Each tree tries to fix the mistakes of the previous trees. It's brilliant, and it works really well.

We'll use XGBoost, which is the most popular gradient boosting library.

### Installation First

```bash
pip install xgboost
```

### The Code

```python
import xgboost as xgb

# Train XGBoost model
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)],
              verbose=False)

# Predictions
y_pred_xgb = xgb_model.predict(X_val)
y_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]

# Evaluate
print("XGBoost Results:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_xgb):.3f}")
print(f"Precision: {precision_score(y_val, y_pred_xgb):.3f}")
print(f"Recall: {recall_score(y_val, y_pred_xgb):.3f}")
print(f"F1 Score: {f1_score(y_val, y_pred_xgb):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_proba_xgb):.3f}")

# Feature importance (XGBoost version)
feature_importance_xgb = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features (XGBoost):")
print(feature_importance_xgb.head(10))
```

### What to Expect

XGBoost usually edges out random forest:
- **Accuracy:** 80-85%
- **Precision:** 68-77%
- **Recall:** 62-75%
- **ROC-AUC:** 0.80-0.87

Those extra few percentage points of accuracy? They come at a cost in complexity and tuning time.

### The Tradeoff

**Pros:**
- Best performance (usually)
- Feature importance
- Handles missing data well
- Can continue training on new data

**Cons:**
- More hyperparameters to tune
- Slower to train than random forest
- Easier to overfit if you're not careful
- Harder to explain to stakeholders

### When to Use XGBoost

**Use it when:**
- You need the best possible performance
- You have time to tune hyperparameters
- You're competing or building a critical system
- Dataset is medium-to-large (10K+ samples)

**Don't use it when:**
- You need results quickly
- Simple interpretability is critical
- Your baseline models (logistic/random forest) already work well enough

---

## Head-to-Head Comparison

Let's compare all four models on the same validation set:

```python
# Create comparison dataframe
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'Accuracy': [
        accuracy_score(y_val, y_pred),
        accuracy_score(y_val, y_pred_dt),
        accuracy_score(y_val, y_pred_rf),
        accuracy_score(y_val, y_pred_xgb)
    ],
    'Precision': [
        precision_score(y_val, y_pred),
        precision_score(y_val, y_pred_dt),
        precision_score(y_val, y_pred_rf),
        precision_score(y_val, y_pred_xgb)
    ],
    'Recall': [
        recall_score(y_val, y_pred),
        recall_score(y_val, y_pred_dt),
        recall_score(y_val, y_pred_rf),
        recall_score(y_val, y_pred_xgb)
    ],
    'F1': [
        f1_score(y_val, y_pred),
        f1_score(y_val, y_pred_dt),
        f1_score(y_val, y_pred_rf),
        f1_score(y_val, y_pred_xgb)
    ],
    'ROC-AUC': [
        roc_auc_score(y_val, y_pred_proba),
        roc_auc_score(y_val, y_pred_proba_dt),
        roc_auc_score(y_val, y_pred_proba_rf),
        roc_auc_score(y_val, y_pred_proba_xgb)
    ]
})

print("\n" + "="*60)
print("MODEL COMPARISON - VALIDATION SET")
print("="*60)
print(results.to_string(index=False))
print("="*60)

# Visualize comparison
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']

for idx, metric in enumerate(metrics):
    axes[idx].bar(results['Model'], results[metric], color='#752763')
    axes[idx].set_title(metric)
    axes[idx].set_ylim(0, 1)
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].axhline(y=results[metric].mean(), color='#fec110', 
                      linestyle='--', label='Average')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nComparison chart saved to model_comparison.png")
```

### Typical Results

On a real churn dataset, here's what you'll usually see:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|--------|---------|
| Logistic Regression | 0.77 | 0.65 | 0.58 | 0.61 | 0.78 |
| Decision Tree | 0.74 | 0.61 | 0.63 | 0.62 | 0.74 |
| Random Forest | 0.81 | 0.71 | 0.67 | 0.69 | 0.83 |
| XGBoost | 0.83 | 0.74 | 0.69 | 0.71 | 0.85 |

**The pattern:**
- Logistic regression: Fast, interpretable, decent performance
- Decision tree: Most interpretable, slightly worse performance
- Random forest: Big performance jump with minimal effort
- XGBoost: Best performance, but diminishing returns

---

## Making the Choice: A Decision Framework

Here's how I actually choose in real projects:

### Start with Logistic Regression if:
- Dataset < 10K rows
- You need results in 30 minutes
- Stakeholders need to understand predictions
- Problem seems mostly linear
- **Time investment:** 30 minutes

### Use Random Forest if:
- You have a few hours for proper training
- Dataset is 10K - 1M rows
- Performance matters more than perfect interpretability
- You want good results without much tuning
- **Time investment:** 2-4 hours

### Use XGBoost if:
- This is a critical production system
- You have days/weeks for proper tuning
- Every percentage point of accuracy matters
- Dataset is 100K+ rows
- **Time investment:** 1-3 days

### Use Decision Tree if:
- You need to show the decision process visually
- Regulations require complete explainability
- You're okay with slightly lower performance
- **Time investment:** 1-2 hours

---

## The Reality Check

Let's be honest about what we just did. We improved from 77% to 83% accuracy by switching from logistic regression to XGBoost. That's a 6% improvement.

**Is that worth it?**

Depends on your business problem:

**If you're predicting which customers to call for retention:**
- 10,000 customers predicted to churn
- 77% accuracy = 7,700 correct predictions
- 83% accuracy = 8,300 correct predictions
- That's 600 more correct predictions

If each saved customer is worth $1,000/year, that's $600K in additional revenue. Suddenly that 6% matters a lot.

**But if you're building an internal demo:**
- Save yourself the time
- Use logistic regression
- Ship it in an afternoon

Context matters. Always.

---

## Testing on Holdout Data

Never, ever, EVER make your final model selection based on validation data. You need a clean holdout test set that you only touch once.

```python
# Choose your best model (let's say Random Forest won)
final_model = rf  # Or whichever performed best

# Evaluate on holdout test set (ONLY ONCE!)
y_test_pred = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE - HOLDOUT TEST SET")
print("="*60)
print(f"Model: {type(final_model).__name__}")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_test_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_test_proba):.3f}")
print("="*60)

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, 
                          target_names=['Stayed', 'Churned']))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_final.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nConfusion matrix saved to confusion_matrix_final.png")
```

If your test performance is significantly worse than validation performance, you probably overfit during model selection. Welcome to ML.

---

## What You Learned

By now, you should be able to:

✅ Train and evaluate four different classification algorithms  
✅ Understand the performance vs interpretability tradeoff  
✅ Know when to use each algorithm  
✅ Compare models using multiple metrics  
✅ Make principled choices based on business context  
✅ Evaluate on proper holdout test data

You're not an expert yet—that takes years—but you can now build real classification models and make informed decisions about which algorithm to use.

---

## Next Up: Tutorial 4

In the next tutorial, we'll switch from classification to **regression** - predicting continuous values instead of categories. We'll build a model to predict customer lifetime value (a number, not a yes/no).

You'll learn:
- When to use regression vs classification
- Different regression algorithms
- Regression-specific evaluation metrics (RMSE, MAE, R²)
- How to handle outliers in regression problems

---

## The Bottom Line

Algorithm selection matters less than people think, but when it matters, it matters a lot. Start with logistic regression, upgrade to random forest if you have time, and only reach for XGBoost when performance is critical.

Your data prep (Tutorial 2) and feature engineering (coming in Tutorial 6) will have bigger impacts than your algorithm choice 90% of the time.

Now go train some models. The code is all here. Use it.

---

**Tutorial 3 Complete**  
**Next:** Tutorial 4 - Regression Models: Predicting Numbers That Matter  
**Previous:** Tutorial 2 - Data Prep: Where ML Projects Live or Die

**Questions?** Find me at [randalscottking.com](https://randalscottking.com)

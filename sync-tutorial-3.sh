#!/bin/bash

# Tutorial 3 Sync Script
# Adds Tutorial 3 to the ml-tutorial-series repository

cd /Users/randalking/Documents/git/ml-tutorial-series

echo "📝 Syncing Tutorial 3: Classification Models..."

# Add all changes
git add .

# Check status
echo ""
echo "📊 Git Status:"
git status

# Commit
echo ""
echo "💾 Committing changes..."
git commit -m "Add Tutorial 3: Classification Models - Pick the Right Tool

- Complete tutorial covering 4 classification algorithms
- Logistic Regression, Decision Trees, Random Forest, XGBoost
- Head-to-head performance comparison
- Decision framework for algorithm selection
- Updated README with progress tracker
- Added xgboost to requirements.txt

Tutorial 3 is complete and ready for students."

# Push to GitHub
echo ""
echo "🚀 Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Tutorial 3 synced successfully!"
echo "🔗 View at: https://github.com/randalscottking/ml-tutorial-series"

#!/bin/bash

# Navigate to repository
cd /Users/randalking/Documents/git/ml-tutorial-series

# Initialize git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: ML tutorial series structure and supporting code"

# Add remote repository
git remote add origin https://github.com/randalscottking/ml-tutorial-series.git

# Set branch to main
git branch -M main

# Push to GitHub
git push -u origin main

echo "Repository synced to GitHub successfully!"

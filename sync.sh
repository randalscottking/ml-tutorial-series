#!/bin/bash

# ML Tutorial Series - Git Sync Script
# This script syncs your local repository with GitHub

echo "=================================================="
echo "ML Tutorial Series - GitHub Sync"
echo "=================================================="

# Change to repository directory
cd /Users/randalking/Documents/git/ml-tutorial-series || exit 1

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git branch -M main
fi

# Check for remote
if ! git remote | grep -q origin; then
    echo "Adding remote repository..."
    git remote add origin https://github.com/randalscottking/ml-tutorial-series.git
fi

# Check git status
echo ""
echo "Git Status:"
git status --short

# Add all files
echo ""
echo "Adding files to git..."
git add .

# Show what will be committed
echo ""
echo "Files to be committed:"
git status --short

# Prompt for commit message
echo ""
read -p "Enter commit message (or press Enter for default): " commit_msg

if [ -z "$commit_msg" ]; then
    commit_msg="Update ML tutorial series"
fi

# Commit changes
echo ""
echo "Committing changes..."
git commit -m "$commit_msg"

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "=================================================="
echo "Sync complete!"
echo "=================================================="
echo ""
echo "View your repository at:"
echo "https://github.com/randalscottking/ml-tutorial-series"

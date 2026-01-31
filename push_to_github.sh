#!/bin/bash
# Script to push code review to GitHub

echo "=========================================="
echo "Pushing Code Review to GitHub"
echo "=========================================="
echo ""
echo "Repository: https://github.com/alazkiyai09/21Days_Project.git"
echo ""
echo "Step 1: Check git status..."
git status

echo ""
echo "Step 2: Check current branch..."
git branch

echo ""
echo "Step 3: Verify remote..."
git remote -v

echo ""
echo "Step 4: Attempting to push..."
echo ""
echo "If authentication fails, please run one of these commands:"
echo ""
echo "Option A: Using Personal Access Token (Recommended)"
echo "  git push -u origin main"
echo "  Username: alazkiyai09"
echo "  Password: <your GitHub Personal Access Token>"
echo ""
echo "Option B: Using SSH"
echo "  git remote set-url origin git@github.com:alazkiyai09/21Days_Project.git"
echo "  git push -u origin main"
echo ""
echo "Option C: Using GitHub CLI"
echo "  gh auth login"
echo "  git push -u origin main"
echo ""

# Try the push
git push -u origin main

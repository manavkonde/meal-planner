@echo off
REM GitHub Push Script for Meal Planner Project
REM Run this after creating the repository on GitHub

cd /d "d:\Manav Konde\PROJECT\meal-planner"

echo.
echo ========================================
echo AI Meal Planner - GitHub Push
echo ========================================
echo.
echo Pushing Week 1 implementation to GitHub...
echo Remote: https://github.com/manavkonde/meal-planner.git
echo.

git push -u origin master

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo ✅ SUCCESS! Code pushed to GitHub
    echo ========================================
    echo.
    echo Repository: https://github.com/manavkonde/meal-planner
    echo Branch: master
    echo.
    echo Next steps:
    echo 1. Open GitHub repo and verify files are there
    echo 2. Add repo link to your portfolio/resume
    echo 3. Prepare Week 2 training (next session)
    echo.
    pause
) else (
    echo.
    echo ========================================
    echo ❌ Push failed - Check the error above
    echo ========================================
    echo.
    echo Common issues:
    echo - Repository not created on GitHub yet
    echo - Authentication not set up
    echo - Wrong GitHub username
    echo.
    pause
)

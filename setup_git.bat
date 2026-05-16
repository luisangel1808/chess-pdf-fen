@echo off
cd /d "%~dp0"
echo === Initialising git repo ===
git init
git add .
git commit -m "Initial commit: chess PDF to FEN extractor"
git branch -M main
git remote add origin https://github.com/luisangel1808/chess-pdf-fen.git
git push -u origin main
echo === Done ===
pause

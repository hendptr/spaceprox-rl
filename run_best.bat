@echo off
setlocal
cd /d "%~dp0RL"
python play_final.py --model "..\models\spaceprox_best.zip"


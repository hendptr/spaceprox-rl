@echo off
setlocal
cd /d "%~dp0RL"
for /f "tokens=1-4 delims=/ " %%a in ("%date%") do set D=%%a%%b%%c
for /f "tokens=1-3 delims=:,. " %%a in ("%time%") do set T=%%a%%b%%c
python train_final.py --minutes 5 --seed 41 --run-name "spaceprox_5min_%D%_%T%"
pause


@echo off
rem Double-click to start the TrustDerm-Net benign/malignant demo.
cd /d "%~dp0.."
python app\trustderm_binary_gui.py %*
if errorlevel 1 pause

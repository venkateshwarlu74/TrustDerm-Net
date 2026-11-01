@echo off
rem Double-click to start the TrustDerm-Net desktop demo.
cd /d "%~dp0.."
python app\trustderm_gui.py %*
if errorlevel 1 pause

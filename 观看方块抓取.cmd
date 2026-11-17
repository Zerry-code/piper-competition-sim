@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-console.ps1" -Mode l1
if errorlevel 1 pause

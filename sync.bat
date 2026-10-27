@echo off
REM Sinkronisasi script dari folder repo sumber (Windows). Contoh: sync.bat --dry-run
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (py -3 sync.py %*) else (python sync.py %*)

@echo off
REM ===========================================================================
REM  start.bat - cukup klik dua kali di Windows Explorer.
REM   - pertama kali: menjalankan setup.bat otomatis (membuat .venv, beberapa menit)
REM   - menjalankan portal + Data Mining UI di latar belakang (tanpa jendela)
REM   - membuka portal di browser, lalu jendela ini tertutup sendiri
REM  Tanpa jendela sama sekali: klik dua kali "start-silent.vbs".
REM  Menghentikan: klik dua kali stop.bat (atau stop-silent.vbs).
REM ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title Portal Simulator dan ML - start
set "VPY=%~dp0.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
set "SILENT=0"
if /i "%~1"=="/silent" set "SILENT=1"

if exist "%VPY%" goto run
echo Instalasi pertama kali: menjalankan setup.bat (beberapa menit) ...
echo.
call "%~dp0setup.bat"
if errorlevel 1 goto fail

:run
"%VPY%" -m portal.winctl start
if errorlevel 1 goto fail
"%VPY%" -m portal.winctl open portal
if "%SILENT%"=="1" exit /b 0
echo.
echo Layanan berjalan di latar belakang. Jendela ini tertutup otomatis ...
ping -n 5 127.0.0.1 >nul
exit /b 0

:fail
if "%SILENT%"=="1" exit /b 1
echo.
echo GAGAL menjalankan layanan. Periksa pesan di atas atau folder logs\
pause
exit /b 1

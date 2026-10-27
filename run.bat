@echo off
REM ===========================================================================
REM  run.bat - menjalankan layanan di latar belakang (tanpa jendela) di Windows.
REM  Padanan run.sh. Portal memakai waitress (gunicorn tidak mendukung Windows).
REM
REM     run.bat start   [portal|datamining]   jalankan (default: semua)
REM     run.bat stop    [portal|datamining]   hentikan
REM     run.bat restart [portal|datamining]
REM     run.bat status                         cek proses & health check
REM     run.bat logs    [portal|datamining]   ikuti log (Ctrl+C keluar)
REM     run.bat open    [portal|datamining]   buka di browser
REM     run.bat fg      portal|datamining     jalan di jendela ini (debug)
REM     run.bat autostart-on                   jalankan otomatis saat login Windows
REM     run.bat autostart-off                  matikan autostart
REM
REM  Port & opsi diatur di config.env (dibuat oleh setup.bat).
REM ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "VPY=%~dp0.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
set "TASK=multiple_python_web"

if "%~1"=="" goto help
if /i "%~1"=="help" goto help
if /i "%~1"=="-h" goto help
if /i "%~1"=="autostart-on" goto auto_on
if /i "%~1"=="autostart-off" goto auto_off

if not exist "%VPY%" (
  echo Virtual environment belum ada. Jalankan dulu: setup.bat
  exit /b 1
)
"%VPY%" -m portal.winctl %*
exit /b %errorlevel%

:auto_on
schtasks /create /f /tn "%TASK%" /sc onlogon /rl limited /tr "\"%~f0\" start" >nul
if errorlevel 1 ( echo Gagal membuat Scheduled Task & exit /b 1 )
echo Autostart aktif: layanan dijalankan setiap Anda login (Task Scheduler: %TASK%)
exit /b 0

:auto_off
schtasks /delete /f /tn "%TASK%" >nul 2>&1
echo Autostart dimatikan.
exit /b 0

:help
for /f "usebackq tokens=1,* delims= " %%A in ("%~f0") do (
  if /i "%%A"=="REM" echo(%%B
)
exit /b 1

@echo off
REM ===========================================================================
REM  stop.bat - menghentikan layanan portal dan Data Mining UI (Windows).
REM
REM     stop.bat                  hentikan semua layanan
REM     stop.bat portal           hentikan portal saja
REM     stop.bat datamining       hentikan Data Mining UI saja
REM     stop.bat /silent          tanpa jeda di akhir (dipakai stop-silent.vbs)
REM  Bisa juga diklik dua kali dari Windows Explorer.
REM
REM  Langkah: (1) hentikan lewat PID di run\ (sama dengan "run.bat stop"),
REM  (2) bila port masih dipakai proses python milik paket ini (mis. file PID
REM  hilang atau dijalankan dengan "run.bat fg"), proses itu ikut dihentikan.
REM  Proses selain python yang memakai port yang sama TIDAK disentuh.
REM ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "VPY=%~dp0.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"

set "TARGET="
set "SILENT=0"
:args
if "%~1"=="" goto argsdone
if /i "%~1"=="/silent" ( set "SILENT=1" ) else ( set "TARGET=%~1" )
shift
goto args
:argsdone
if "%TARGET%"=="" set "TARGET=all"
if /i "%TARGET%"=="-h" goto help
if /i "%TARGET%"=="--help" goto help
if /i not "%TARGET%"=="all" if /i not "%TARGET%"=="portal" if /i not "%TARGET%"=="datamining" (
  echo Layanan tidak dikenal: %TARGET%   ^(pilihan: portal, datamining^)
  exit /b 1
)

REM ---- baca port dari config.env (default 8000 / 8501)
set "PORTAL_PORT=8000"
set "STREAMLIT_PORT=8501"
if exist config.env (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in ("config.env") do (
    if /i "%%A"=="PORTAL_PORT" if not "%%B"=="" set "PORTAL_PORT=%%B"
    if /i "%%A"=="STREAMLIT_PORT" if not "%%B"=="" set "STREAMLIT_PORT=%%B"
  )
)

if /i "%TARGET%"=="all" (
  call :stop_service portal %PORTAL_PORT%
  call :stop_service datamining %STREAMLIT_PORT%
)
if /i "%TARGET%"=="portal" call :stop_service portal %PORTAL_PORT%
if /i "%TARGET%"=="datamining" call :stop_service datamining %STREAMLIT_PORT%
if "%SILENT%"=="0" (
  echo.
  echo Selesai. Jendela ini tertutup otomatis ...
  ping -n 4 127.0.0.1 >nul
)
endlocal
exit /b 0

REM ------------------------------------------------------------------
REM  :stop_service <nama> <port>
:stop_service
set "SVC=%~1"
set "PORT=%~2"
REM (1) lewat PID file
if exist "%VPY%" (
  "%VPY%" -m portal.winctl stop %SVC%
) else if exist "run\%SVC%.pid" (
  for /f "usebackq delims=" %%P in ("run\%SVC%.pid") do (
    echo Menghentikan %SVC% ^(pid %%P^) ...
    taskkill /PID %%P /T /F >nul 2>&1
  )
  del /q "run\%SVC%.pid" >nul 2>&1
)
REM (2) sisa proses python yang masih LISTEN di port (PowerShell: tidak bergantung bahasa Windows)
set "PSCMD=Get-NetTCPConnection -State Listen -LocalPort %PORT% -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique"
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "%PSCMD%"`) do call :kill_if_python %%P %PORT% %SVC%
ping -n 2 127.0.0.1 >nul
set "LEFT="
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "%PSCMD%"`) do set "LEFT=%%P"
if defined LEFT (
  echo   ! port %PORT% masih dipakai pid %LEFT%
) else (
  echo %SVC% berhenti ^(port %PORT% bebas^)
)
exit /b 0

REM  :kill_if_python <pid> <port> <nama>
:kill_if_python
if "%~1"=="0" exit /b 0
tasklist /FI "PID eq %~1" /NH 2>nul | findstr /i "python" >nul
if errorlevel 1 (
  echo   ! port %~2 dipakai proses lain ^(pid %~1^) - tidak dihentikan
  exit /b 0
)
echo Menghentikan sisa proses %~3 di port %~2 ^(pid %~1^)
taskkill /PID %~1 /T /F >nul 2>&1
exit /b 0

:help
echo Pemakaian: stop.bat [portal^|datamining]
echo   tanpa argumen = hentikan semua layanan
exit /b 0

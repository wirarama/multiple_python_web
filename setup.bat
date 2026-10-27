@echo off
REM ===========================================================================
REM  setup.bat - membuat virtual environment Python (.venv) dan memasang semua
REM  dependensi di Windows (padanan setup.sh untuk Linux).
REM
REM  Pemakaian (Command Prompt / PowerShell, di folder repo):
REM     setup.bat                   setup baru
REM     run.bat start               jalankan semua layanan
REM
REM  Opsi:
REM     --python C:\Python311\python.exe   pilih interpreter tertentu
REM     --recreate                         hapus .venv lama lalu buat ulang
REM     --sync                             salin ulang script dari folder sumber (..\software_defined_network, dst.)
REM     --sync-git                         salin ulang script dari GitHub sumber
REM     --no-datamining                    lewati paket Streamlit/Data Mining
REM     -h, --help                         bantuan
REM ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "PYCMD="
set "RECREATE=0"
set "WITH_DM=1"
set "DO_SYNC="

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--python" (
  set PYCMD="%~2"
  shift
  shift
  goto parse
)
if /i "%~1"=="--recreate" ( set "RECREATE=1" & shift & goto parse )
if /i "%~1"=="--no-datamining" ( set "WITH_DM=0" & shift & goto parse )
if /i "%~1"=="--sync" ( set "DO_SYNC=local" & shift & goto parse )
if /i "%~1"=="--sync-git" ( set "DO_SYNC=git" & shift & goto parse )
if /i "%~1"=="-h" goto help
if /i "%~1"=="--help" goto help
echo [ERR] Opsi tidak dikenal: %~1   (lihat: setup.bat --help)
exit /b 1

:help
echo Pemakaian: setup.bat [opsi]
echo   --python C:\Python311\python.exe   pilih interpreter tertentu
echo   --recreate                         hapus .venv lama lalu buat ulang
echo   --sync                             salin ulang script dari folder sumber
echo   --sync-git                         salin ulang script dari GitHub sumber
echo   --no-datamining                    lewati paket Streamlit/Data Mining
exit /b 0

:parsed
REM ------------------------------------------------------------- 1. Python
if defined PYCMD goto checkpy
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if not errorlevel 1 ( set "PYCMD=py -3" & goto checkpy )
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if not errorlevel 1 ( set "PYCMD=python" & goto checkpy )
echo [ERR] Python 3.9 atau lebih baru tidak ditemukan.
echo       Pasang dari https://www.python.org/downloads/windows/ dan centang "Add python.exe to PATH".
exit /b 1

:checkpy
%PYCMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if errorlevel 1 (
  echo [ERR] %PYCMD% tidak bisa dijalankan atau versinya di bawah 3.9
  exit /b 1
)
echo ==^> Memakai interpreter: %PYCMD%
%PYCMD% -c "import sys; print('    Python', sys.version.split()[0], '-', sys.executable)"

REM ------------------------------------------------------- 2. sync (opsional)
if "%DO_SYNC%"=="local" (
  echo ==^> Sinkronisasi script dari folder sumber
  %PYCMD% sync.py || exit /b 1
)
if "%DO_SYNC%"=="git" (
  echo ==^> Sinkronisasi script dari GitHub sumber
  %PYCMD% sync.py --source git || exit /b 1
)
for %%D in (sdn smarthome rlfl pyspark datamining) do (
  if not exist "apps\%%D\app.py" echo  !! apps\%%D\app.py tidak ada - jalankan: sync.bat
)

REM ------------------------------------------------------------------ 3. venv
if "%RECREATE%"=="1" if exist "%VENV%" (
  echo ==^> Menghapus .venv lama ^(--recreate^)
  rmdir /s /q "%VENV%"
)
if exist "%VPY%" (
  "%VPY%" -c "import sys" >nul 2>&1 || (
    echo  !! .venv rusak / dibuat di komputer lain - dibuat ulang
    rmdir /s /q "%VENV%"
  )
)
if exist "%VPY%" (
  echo ==^> .venv sudah ada - memperbarui paket ^(pakai --recreate untuk setup bersih^)
) else (
  echo ==^> Membuat virtual environment di .venv
  %PYCMD% -m venv "%VENV%"
  if errorlevel 1 ( echo [ERR] Gagal membuat venv & exit /b 1 )
)

echo ==^> Memperbarui pip
"%VPY%" -m pip install --upgrade --quiet pip setuptools wheel
if errorlevel 1 ( echo [ERR] Gagal memperbarui pip & exit /b 1 )

set "REQ=requirements.txt"
if "%WITH_DM%"=="0" set "REQ=requirements-web.txt"
echo ==^> Memasang dependensi dari %REQ% (bisa beberapa menit)
"%VPY%" -m pip install --quiet -r "%REQ%"
if errorlevel 1 ( echo [ERR] Gagal memasang dependensi & exit /b 1 )
echo  OK Dependensi terpasang

REM ----------------------------------------------------------- 4. konfigurasi
if not exist logs mkdir logs
if not exist run mkdir run
if not exist config.env (
  copy /y config.env.example config.env >nul
  echo  OK config.env dibuat dari config.env.example ^(ubah port di sini bila perlu^)
)
if "%WITH_DM%"=="0" "%VPY%" -c "import re,pathlib; p=pathlib.Path('config.env'); s=p.read_text(); s=re.sub(r'(?m)^ENABLE_DATAMINING=.*$', 'ENABLE_DATAMINING=0', s); p.write_text(s)"

REM ------------------------------------------------------------- 5. uji cepat
echo ==^> Uji impor aplikasi Flask
set "MPLBACKEND=Agg"
"%VPY%" -c "import sys; from portal.gateway import build, FLASK_APPS; gw = build(); print('   dimuat :', ', '.join(gw.apps)); sys.exit(0 if len(gw.apps) == len(FLASK_APPS) else 1)"
if errorlevel 1 ( echo [ERR] Ada aplikasi yang gagal dimuat ^(lihat pesan GAGAL di atas^) & exit /b 1 )
echo  OK Semua aplikasi Flask dapat dimuat
"%VPY%" -c "import waitress" >nul 2>&1 || ( echo [ERR] waitress belum terpasang & exit /b 1 )
if "%WITH_DM%"=="1" (
  pushd apps\datamining
  "%VPY%" -c "import streamlit, sklearn, statsmodels, mlxtend, scipy; print('   streamlit', streamlit.__version__)"
  if errorlevel 1 ( popd & echo [ERR] Paket Data Mining belum lengkap & exit /b 1 )
  popd
  echo  OK Paket Data Mining UI siap
)

echo.
echo Setup selesai.
echo   Jalankan  : run.bat start
echo   Buka      : run.bat open
echo   Status    : run.bat status
echo   Hentikan  : run.bat stop
echo Portal (default) di http://localhost:8000/  - port diatur di config.env
endlocal
exit /b 0

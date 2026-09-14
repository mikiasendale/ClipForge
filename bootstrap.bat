@echo off
REM ClipForge bootstrap (Windows). Installs ffmpeg + Python 3.12 if missing,
REM creates .venv, installs pinned deps, fetches Scrapling browser, seeds .env.
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "VENV=.venv"

echo(
echo === ClipForge bootstrap ===

REM ---- detect winget
where winget >nul 2>nul
set "HAS_WINGET=%errorlevel%"
if "%HAS_WINGET%"=="0" (set "WINGET=winget") else (set "WINGET=")

REM ---- ffmpeg ----
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg not found - installing...
  if defined WINGET (
    winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
  ) else (
    echo winget missing - downloading static build to tools\ffmpeg
    call :download_ffmpeg
  )
) else (
  echo ffmpeg present.
)

REM ---- python 3.12 ----
set "PY="
py -3.12 --version >nul 2>nul && set "PY=py -3.12"
if not defined PY (
  python --version 2>&1 | findstr /R "3\.12\." >nul && set "PY=python"
)
if not defined PY (
  echo python 3.12 not found - trying winget...
  if defined WINGET (
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    py -3.12 --version >nul 2>nul && set "PY=py -3.12"
  )
)
if not defined PY (
  echo ERROR: Python 3.12 is required. Install it manually, then re-run:
  echo   https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe
  echo   ^(check "Add python.exe to PATH" during install^)
  exit /b 1
)
echo python: %PY%

REM ---- venv + deps ----
if not exist "%VENV%" (
  echo creating virtualenv...
  %PY% -m venv %VENV%
)
echo installing requirements...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip >nul
"%VENV%\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 ( echo ERROR: pip install failed & exit /b 1 )

echo fetching Scrapling browser (best-effort)...
"%VENV%\Scripts\scrapling.exe" install || echo WARN: scrapling install failed - enrichment will degrade

REM ---- .env ----
if not exist ".env" copy ".env.example" ".env" >nul

echo(
echo Done - run run.bat
exit /b 0

:download_ffmpeg
where curl >nul 2>nul || (echo curl missing, cannot auto-download ffmpeg & exit /b 1)
if not exist tools mkdir tools
echo downloading gyan.dev static build...
curl -L -o tools\ffmpeg.zip https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
if errorlevel 1 exit /b 1
powershell -NoProfile -Command "Expand-Archive tools\ffmpeg.zip -DestinationPath tools\ffmpeg -Force"
for /d %%d in (tools\ffmpeg\ffmpeg-*\bin) do set "FFDIR=%%d"
if defined FFDIR set "PATH=%FFDIR%;%PATH%"
exit /b 0

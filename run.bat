@echo off
REM ClipForge launcher (Windows). Activates .venv, ensures ffmpeg, starts the UI
REM (or runs the daily job headless with --auto).
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "VENV=.venv"
set "HOST=127.0.0.1"
set "PORT=8000"

if not exist "%VENV%\Scripts\python.exe" (
  echo No .venv found. Run bootstrap.bat first.
  exit /b 1
)
call "%VENV%\Scripts\activate.bat"

REM ---- resolve ffmpeg; record bundled path in .env ----
where ffmpeg >nul 2>nul
if errorlevel 1 (
  if exist "tools\ffmpeg" for /d %%d in (tools\ffmpeg\ffmpeg-*\bin) do (
    set "FFDIR=%%d"
  )
  if defined FFDIR (
    set "PATH=!FFDIR!;!PATH!"
    findstr /B "FFMPEG_PATH=" .env >nul 2>nul || echo FFMPEG_PATH=!FFDIR!\ffmpeg.exe>> .env
  )
)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  if not exist "!FFDIR!\ffmpeg.exe" (
    echo ffmpeg not found. Run bootstrap.bat first.
    exit /b 1
  )
)

if /I "%~1"=="--auto" (
  echo [ClipForge] running daily job headless...
  python -m app.main --auto
  exit /b %errorlevel%
)

start "" "http://%HOST%:%PORT%"
echo [ClipForge] starting http://%HOST%:%PORT%
uvicorn app.main:app --host %HOST% --port %PORT%
endlocal

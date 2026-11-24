@echo off
rem Start the compression comparison page (this computer only) and open it in the browser.
rem Closing this window stops the server.
rem Python: %HEBZIP_PYTHON% if set, else the repository's own .venv (two folders up), else "python" on the PATH.
rem It must have hebzip's dependencies (numpy, numba, zstandard, pyppmd).
chcp 65001 >nul
cd /d "%~dp0"
title hebzip - comparison page (close this window to stop)
set PYTHONDONTWRITEBYTECODE=1
set "PY=python"
if exist "%~dp0..\..\.venv\Scripts\python.exe" set "PY=%~dp0..\..\.venv\Scripts\python.exe"
if defined HEBZIP_PYTHON set "PY=%HEBZIP_PYTHON%"
"%PY%" -m atar.ui.server %*
echo.
echo The server stopped. You can close this window.
pause >nul

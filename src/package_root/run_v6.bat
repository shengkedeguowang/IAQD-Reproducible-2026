@echo off
setlocal
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"
set "PYTHONPATH=%~dp0src"
"%PYTHON_EXE%" "%~dp0run_all_v6.py" %*
exit /b %ERRORLEVEL%


@echo off
REM NEXUS-Engine launcher for cmd.exe: nexus setup / quickstart / serve / all
setlocal
set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%nexus.ps1" %*
endlocal

@echo off
setlocal
call "%~dp0sparkdeck.cmd" start
exit /b %ERRORLEVEL%

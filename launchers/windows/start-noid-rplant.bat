@echo off
REM X-Miner - Parano1d (NOID) on rplant.xyz (1% pool fee)
REM Edit WALLET (your o1... NOID address) and WORKER below, then double-click this file.
cd /d "%~dp0"

set WALLET=YOUR_NOID_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=na.rplant.xyz:17185

:loop
x-miner.exe --algorithm noid --pool %POOL% --wallet %WALLET% --worker %WORKER%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
